#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/cfd_result.hpp"
#include "aero_cfd/cfd_solver.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__global__ void check_status_kernel(
    const int* d_failed,
    const float* d_l2_sum,
    int nvar_ncells,
    float convergence_tol,
    int* d_converged,
    float* d_residual_history_slot) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (*d_failed != 0) {
        *d_converged = 1;
        *d_residual_history_slot = -1.0f;
        return;
    }
    float l2 = sqrtf(*d_l2_sum / static_cast<float>(nvar_ncells));
    *d_residual_history_slot = l2;
    if (l2 < convergence_tol) {
        *d_converged = 1;
    }
}

} // namespace

static void solve_gpu_free(int* d_failed, float* d_min_dt, float* d_l2_sum, float* d_forces,
    int* d_converged, float* d_residual_history) {
    cudaFree(d_failed); cudaFree(d_min_dt); cudaFree(d_l2_sum); cudaFree(d_forces);
    cudaFree(d_converged); cudaFree(d_residual_history);
}

static CfdSolveSummary solve_gpu_impl(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    float* d_min_dt,
    float* d_l2_sum,
    float* d_forces,
    int* d_converged,
    float* d_residual_history,
    bool owned_buffers,
    std::string* error) {
    CfdSolveSummary summary;
    std::vector<float> host_residual_history;
    int host_failed = 0;
    int host_converged = 0;

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    int nvar_ncells = DeviceMesh::NVAR * static_cast<int>(d_mesh.cell_count());

    if (!cuda_check(cudaMemset(d_converged, 0, sizeof(int)), "cudaMemset converged", error)) goto fail;

    for (int iter = 0; iter < config.max_iter; ++iter) {
        if (!compute_timestep_gpu(d_mesh, config.gamma, config.cfl, d_min_dt)) {
            if (error) *error = "timestep kernel failed";
            goto fail;
        }

        if (!launch_euler_residual_kernel(d_mesh, w_inf, config.gamma, d_failed, nullptr, error)) {
            goto fail;
        }

        if (!compute_update_gpu(d_mesh, d_min_dt, config.gamma, d_l2_sum, d_failed)) {
            if (error) *error = "update kernel failed";
            goto fail;
        }

        check_status_kernel<<<1, 1>>>(
            d_failed, d_l2_sum, nvar_ncells,
            config.convergence_tol,
            d_converged, d_residual_history + iter);
        if (!cuda_check(cudaGetLastError(), "check_status kernel launch", error)) goto fail;
    }

    if (!cuda_check(cudaDeviceSynchronize(), "post-loop sync", error)) goto fail;

    if (!cuda_check(cudaMemcpy(&host_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "read d_failed", error)) goto fail;
    if (!cuda_check(cudaMemcpy(&host_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost), "read d_converged", error)) goto fail;

    if (host_failed != 0) {
        if (error) *error = "GPU solver failed during iteration";
        summary.failed = true;
        goto cleanup;
    }

    host_residual_history.assign(config.max_iter, 0.0f);
    if (!cuda_check(cudaMemcpy(host_residual_history.data(), d_residual_history,
            config.max_iter * sizeof(float), cudaMemcpyDeviceToHost), "read residual history", error)) goto fail;

    {
        int valid_iters = 0;
        for (int i = 0; i < config.max_iter; ++i) {
            if (host_residual_history[i] < 0.0f) break;
            summary.residual_history.push_back(host_residual_history[i]);
            valid_iters++;
        }
        if (valid_iters > 0 && host_residual_history[valid_iters - 1] < config.convergence_tol) {
            summary.converged = true;
        }
    }

    if (!compute_wall_forces_gpu(d_mesh, config.gamma, d_forces)) {
        if (error) *error = "wall force kernel failed";
        goto fail;
    }
    if (!cuda_check(cudaDeviceSynchronize(), "wall force sync", error)) goto fail;

    {
        float forces[6];
        if (!cuda_check(cudaMemcpy(forces, d_forces, 6 * sizeof(float), cudaMemcpyDeviceToHost), "read d_forces", error)) goto fail;

        float q_inf = 0.5f * condition.mach * condition.mach;
        float inv_force_ref = 1.0f / fmaxf(q_inf * config.ref_area, 1e-30f);
        summary.forces.CX = forces[0] * inv_force_ref;
        summary.forces.CY = forces[1] * inv_force_ref;
        summary.forces.CZ = forces[2] * inv_force_ref;
        summary.forces.Cl = forces[3] / fmaxf(q_inf * config.ref_area * config.ref_span, 1e-30f);
        summary.forces.Cm = forces[4] / fmaxf(q_inf * config.ref_area * config.ref_length, 1e-30f);
        summary.forces.Cn = forces[5] / fmaxf(q_inf * config.ref_area * config.ref_span, 1e-30f);

        constexpr float kPi = 3.14159265358979323846f;
        float alpha = condition.alpha_deg * kPi / 180.0f;
        float beta = condition.beta_deg * kPi / 180.0f;
        float ca = cosf(alpha);
        float sa = sinf(alpha);
        float cb = cosf(beta);
        float sb = sinf(beta);
        float fsx = summary.forces.CX * ca * cb + summary.forces.CY * sb + summary.forces.CZ * sa * cb;
        float fsz = -summary.forces.CX * sa + summary.forces.CZ * ca;
        summary.forces.CD = -fsx;
        summary.forces.CL = -fsz;

        summary.forces.iterations = static_cast<int>(summary.residual_history.size());
        summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
    }

    goto cleanup;

fail:
    summary.failed = true;

cleanup:
    if (owned_buffers) solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history);
    return summary;
}

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error) {
    if (d_mesh.cell_count() == 0 || d_mesh.face_count() == 0) {
        CfdSolveSummary s;
        if (error) *error = "DeviceMesh is empty";
        s.failed = true;
        return s;
    }

    int* d_failed = nullptr;
    float* d_min_dt = nullptr;
    float* d_l2_sum = nullptr;
    float* d_forces = nullptr;
    int* d_converged = nullptr;
    float* d_residual_history = nullptr;

    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_min_dt, sizeof(float)), "cudaMalloc d_min_dt", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_l2_sum, sizeof(float)), "cudaMalloc d_l2_sum", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_forces, 6 * sizeof(float)), "cudaMalloc d_forces", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_converged, sizeof(int)), "cudaMalloc d_converged", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(float)), "cudaMalloc d_residual_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_converged, d_residual_history); CfdSolveSummary s; s.failed = true; return s; }

    return solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_converged, d_residual_history, true, error);
}

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    float* d_min_dt,
    float* d_l2_sum,
    float* d_forces,
    std::string* error) {
    if (d_mesh.cell_count() == 0 || d_mesh.face_count() == 0) {
        CfdSolveSummary s;
        if (error) *error = "DeviceMesh is empty";
        s.failed = true;
        return s;
    }

    int* d_converged = nullptr;
    float* d_residual_history = nullptr;
    if (!cuda_check(cudaMalloc(&d_converged, sizeof(int)), "cudaMalloc d_converged", error)) { cudaFree(d_converged); cudaFree(d_residual_history); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(float)), "cudaMalloc d_residual_history", error)) { cudaFree(d_converged); cudaFree(d_residual_history); CfdSolveSummary s; s.failed = true; return s; }

    CfdSolveSummary result = solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_converged, d_residual_history, false, error);
    cudaFree(d_converged);
    cudaFree(d_residual_history);
    return result;
}

} // namespace Cfd
} // namespace AeroSim
