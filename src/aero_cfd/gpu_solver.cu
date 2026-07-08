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
#include "aero_cfd/diagnostics.hpp"

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
    float* d_residual_history_slot) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (*d_failed != 0) {
        *d_residual_history_slot = -1.0f;
        return;
    }
    float l2 = sqrtf(*d_l2_sum / static_cast<float>(nvar_ncells));
    *d_residual_history_slot = l2;
}

} // namespace

static void solve_gpu_free(int* d_failed, float* d_min_dt, float* d_l2_sum, float* d_forces,
    float* d_residual_history, float* d_state_bounds_history, int* d_failure_cell, float* d_failure_state) {
    cudaFree(d_failed); cudaFree(d_min_dt); cudaFree(d_l2_sum); cudaFree(d_forces);
    cudaFree(d_residual_history); cudaFree(d_state_bounds_history);
    cudaFree(d_failure_cell); cudaFree(d_failure_state);
}

static CfdSolveSummary solve_gpu_impl(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    float* d_min_dt,
    float* d_l2_sum,
    float* d_forces,
    float* d_residual_history,
    float* d_state_bounds_history,
    int* d_failure_cell,
    float* d_failure_state,
    bool owned_buffers,
    std::string* error) {
    CfdSolveSummary summary;
    std::vector<float> host_residual_history;
    int host_failed = 0;
    bool diagnostics_enabled = config.diagnostic_level != DiagnosticLevel::Off;

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    int nvar_ncells = DeviceMesh::NVAR * static_cast<int>(d_mesh.cell_count());

    for (int iter = 0; iter < config.max_iter; ++iter) {
        if (config.reconstruction_order == 2) {
            if (!compute_gradients_gpu(d_mesh, config.gamma, error)) goto fail;
            if (!compute_limiters_gpu(d_mesh, config.gamma, error)) goto fail;
            if (!apply_limiter_gpu(d_mesh, false, error)) goto fail;
        }

        if (!compute_timestep_gpu(d_mesh, config.gamma, config.cfl, d_min_dt)) {
            if (error) *error = "timestep kernel failed";
            goto fail;
        }

        if (!launch_euler_residual_kernel(d_mesh, w_inf, config.gamma, d_failed, nullptr, error,
                config.reconstruction_order)) {
            goto fail;
        }

        if (!compute_update_gpu(d_mesh, d_min_dt, config.gamma, d_l2_sum, d_failed,
                d_failure_cell, d_failure_state)) {
            if (error) *error = "update kernel failed";
            goto fail;
        }

        if (diagnostics_enabled) {
            if (!compute_state_bounds_gpu(d_mesh, config.gamma, d_state_bounds_history + iter * 6)) {
                if (error) *error = "state bounds kernel failed";
                goto fail;
            }
        }

        check_status_kernel<<<1, 1>>>(
            d_failed, d_l2_sum, nvar_ncells,
            config.convergence_tol, d_residual_history + iter);
        if (!cuda_check(cudaGetLastError(), "check_status kernel launch", error)) goto fail;
    }

    if (!cuda_check(cudaDeviceSynchronize(), "post-loop sync", error)) goto fail;

    if (!cuda_check(cudaMemcpy(&host_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "read d_failed", error)) goto fail;

    host_residual_history.assign(config.max_iter, 0.0f);
    if (!cuda_check(cudaMemcpy(host_residual_history.data(), d_residual_history,
            config.max_iter * sizeof(float), cudaMemcpyDeviceToHost), "read residual history", error)) goto fail;

    if (host_failed != 0) {
        if (error) *error = "GPU solver failed during iteration";
        summary.failed = true;
    }

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
        int valid_iters = 0;
        for (int i = 0; i < config.max_iter; ++i) {
            if (host_residual_history[i] < 0.0f) break;
            summary.residual_history.push_back(host_residual_history[i]);
            valid_iters++;
        }
        if (!summary.failed && valid_iters > 0 && host_residual_history[valid_iters - 1] < config.convergence_tol) {
            summary.converged = true;
        }
    }

    if (diagnostics_enabled) {
        std::vector<float> bounds_host(config.max_iter * 6);
        if (!cuda_check(cudaMemcpy(bounds_host.data(), d_state_bounds_history,
                config.max_iter * 6 * sizeof(float), cudaMemcpyDeviceToHost), "read state bounds history", error)) goto fail;

        for (int i = 0; i < config.max_iter; ++i) {
            StateBounds sb;
            sb.min_rho = bounds_host[i * 6 + 0];
            sb.max_rho = bounds_host[i * 6 + 1];
            sb.min_p = bounds_host[i * 6 + 2];
            sb.max_p = bounds_host[i * 6 + 3];
            sb.min_mach = bounds_host[i * 6 + 4];
            sb.max_mach = bounds_host[i * 6 + 5];
            sb.valid = true;
            summary.diagnostics.state_bounds_history.push_back(sb);
        }

        if (host_failed != 0 && d_failure_cell) {
            int host_failure_cell = -1;
            float host_failure_state[5] = {0.0f};
            cudaMemcpy(&host_failure_cell, d_failure_cell, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_failure_state, d_failure_state, 5 * sizeof(float), cudaMemcpyDeviceToHost);

            if (host_failure_cell >= 0) {
                int fail_iter = 0;
                for (int i = 0; i < config.max_iter; ++i) {
                    if (host_residual_history[i] < 0.0f) { fail_iter = i; break; }
                }
                ConservativeState fail_q;
                fail_q.rho = host_failure_state[0];
                fail_q.rho_u = host_failure_state[1];
                fail_q.rho_v = host_failure_state[2];
                fail_q.rho_w = host_failure_state[3];
                fail_q.rho_E = host_failure_state[4];
                summary.diagnostics.failure = make_failure_snapshot(
                    fail_iter + 1, host_failure_cell, "NaN/non-positive state", fail_q, config.gamma);
            }
        }
    }

    if (!summary.failed) {
        if (!compute_wall_forces_gpu(d_mesh, config.gamma, d_forces)) {
            if (error) *error = "wall force kernel failed";
            goto fail;
        }
        if (!cuda_check(cudaDeviceSynchronize(), "wall force sync", error)) goto fail;

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
    if (owned_buffers) solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history,
        d_state_bounds_history, d_failure_cell, d_failure_state);
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
    float* d_residual_history = nullptr;
    float* d_state_bounds_history = nullptr;
    int* d_failure_cell = nullptr;
    float* d_failure_state = nullptr;

    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_min_dt, sizeof(float)), "cudaMalloc d_min_dt", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_l2_sum, sizeof(float)), "cudaMalloc d_l2_sum", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_forces, 6 * sizeof(float)), "cudaMalloc d_forces", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(float)), "cudaMalloc d_residual_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }

    bool diag = config.diagnostic_level != DiagnosticLevel::Off;
    if (diag) {
        if (!cuda_check(cudaMalloc(&d_state_bounds_history, config.max_iter * 6 * sizeof(float)), "cudaMalloc d_state_bounds_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_cell, sizeof(int)), "cudaMalloc d_failure_cell", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_state, 5 * sizeof(float)), "cudaMalloc d_failure_state", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMemset(d_failure_cell, 0xFF, sizeof(int)), "init d_failure_cell", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    }

    return solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state, true, error);
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

    float* d_residual_history = nullptr;
    float* d_state_bounds_history = nullptr;
    int* d_failure_cell = nullptr;
    float* d_failure_state = nullptr;

    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(float)), "cudaMalloc d_residual_history", error)) { cudaFree(d_residual_history); CfdSolveSummary s; s.failed = true; return s; }

    bool diag = config.diagnostic_level != DiagnosticLevel::Off;
    if (diag) {
        if (!cuda_check(cudaMalloc(&d_state_bounds_history, config.max_iter * 6 * sizeof(float)), "cudaMalloc d_state_bounds_history", error)) { cudaFree(d_residual_history); cudaFree(d_state_bounds_history); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_cell, sizeof(int)), "cudaMalloc d_failure_cell", error)) { cudaFree(d_residual_history); cudaFree(d_state_bounds_history); cudaFree(d_failure_cell); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_state, 5 * sizeof(float)), "cudaMalloc d_failure_state", error)) { cudaFree(d_residual_history); cudaFree(d_state_bounds_history); cudaFree(d_failure_cell); cudaFree(d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMemset(d_failure_cell, 0xFF, sizeof(int)), "init d_failure_cell", error)) { cudaFree(d_residual_history); cudaFree(d_state_bounds_history); cudaFree(d_failure_cell); cudaFree(d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    }

    CfdSolveSummary result = solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state, false, error);
    cudaFree(d_residual_history);
    cudaFree(d_state_bounds_history);
    cudaFree(d_failure_cell);
    cudaFree(d_failure_state);
    return result;
}

} // namespace Cfd
} // namespace AeroSim
