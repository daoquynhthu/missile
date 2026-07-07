#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/cfd_result.hpp"
#include "aero_cfd/cfd_solver.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver.hpp"

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

// Forward declarations for helper functions in sibling .cu files
bool compute_timestep_gpu(DeviceMesh& mesh, float gamma, float cfl, float* d_min_dt);
bool compute_update_gpu(DeviceMesh& mesh, float min_dt, float gamma,
    float* d_l2_sum, int* d_failed);
bool compute_wall_forces_gpu(DeviceMesh& mesh, float gamma, float* d_forces);

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error) {
    CfdSolveSummary summary;
    if (d_mesh.cell_count() <= 0 || d_mesh.face_count() <= 0) {
        if (error) *error = "DeviceMesh is empty";
        summary.failed = true;
        return summary;
    }

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);

    int* d_failed = nullptr;
    float* d_min_dt = nullptr;
    float* d_l2_sum = nullptr;
    float* d_forces = nullptr;

    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", error)) goto fail;
    if (!cuda_check(cudaMalloc(&d_min_dt, sizeof(float)), "cudaMalloc d_min_dt", error)) goto fail;
    if (!cuda_check(cudaMalloc(&d_l2_sum, sizeof(float)), "cudaMalloc d_l2_sum", error)) goto fail;
    if (!cuda_check(cudaMalloc(&d_forces, 6 * sizeof(float)), "cudaMalloc d_forces", error)) goto fail;

    for (int iter = 0; iter < config.max_iter; ++iter) {
        if (!compute_timestep_gpu(d_mesh, config.gamma, config.cfl, d_min_dt)) {
            if (error) *error = "timestep kernel failed";
            goto fail;
        }

        if (!launch_euler_residual_kernel(d_mesh, w_inf, config.gamma, d_failed, error)) {
            goto fail;
        }

        if (!cuda_check(cudaDeviceSynchronize(), "residual kernel sync", error)) goto fail;

        int residual_failed = 0;
        if (!cuda_check(cudaMemcpy(&residual_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "read d_failed", error)) goto fail;
        if (residual_failed != 0) {
            if (error) *error = "GPU residual encountered invalid state";
            summary.failed = true;
            goto cleanup;
        }

        float min_dt = 0.0f;
        if (!cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(float), cudaMemcpyDeviceToHost), "read d_min_dt", error)) goto fail;

        if (!compute_update_gpu(d_mesh, min_dt, config.gamma, d_l2_sum, d_failed)) {
            if (error) *error = "update kernel failed";
            goto fail;
        }

        if (!cuda_check(cudaDeviceSynchronize(), "update kernel sync", error)) goto fail;

        int update_failed = 0;
        if (!cuda_check(cudaMemcpy(&update_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "read update d_failed", error)) goto fail;
        if (update_failed != 0) {
            if (error) *error = "GPU update produced invalid state";
            summary.failed = true;
            goto cleanup;
        }

        float l2 = 0.0f;
        if (!cuda_check(cudaMemcpy(&l2, d_l2_sum, sizeof(float), cudaMemcpyDeviceToHost), "read d_l2", error)) goto fail;
        float residual_l2 = std::sqrt(l2 / (5.0f * static_cast<float>(d_mesh.cell_count())));
        summary.residual_history.push_back(residual_l2);

        if (residual_l2 < config.convergence_tol) {
            summary.converged = true;
            break;
        }
    }

    if (!cuda_check(cudaDeviceSynchronize(), "post-loop sync", error)) goto fail;

    if (!compute_wall_forces_gpu(d_mesh, config.gamma, d_forces)) {
        if (error) *error = "wall force kernel failed";
        goto fail;
    }
    if (!cuda_check(cudaDeviceSynchronize(), "wall force sync", error)) goto fail;

    {
        float forces[6];
        if (!cuda_check(cudaMemcpy(forces, d_forces, 6 * sizeof(float), cudaMemcpyDeviceToHost), "read d_forces", error)) goto fail;

        float q_inf = 0.5f * condition.mach * condition.mach;
        float inv_force_ref = 1.0f / std::max(q_inf * config.ref_area, 1e-30f);
        summary.forces.CX = forces[0] * inv_force_ref;
        summary.forces.CY = forces[1] * inv_force_ref;
        summary.forces.CZ = forces[2] * inv_force_ref;
        summary.forces.Cl = forces[3] / std::max(q_inf * config.ref_area * config.ref_span, 1e-30f);
        summary.forces.Cm = forces[4] / std::max(q_inf * config.ref_area * config.ref_length, 1e-30f);
        summary.forces.Cn = forces[5] / std::max(q_inf * config.ref_area * config.ref_span, 1e-30f);

        float alpha = condition.alpha_deg * 3.14159265358979323846f / 180.0f;
        float beta = condition.beta_deg * 3.14159265358979323846f / 180.0f;
        float ca = std::cos(alpha);
        float sa = std::sin(alpha);
        float cb = std::cos(beta);
        float sb = std::sin(beta);
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
    cudaFree(d_failed);
    cudaFree(d_min_dt);
    cudaFree(d_l2_sum);
    cudaFree(d_forces);
    return summary;
}

} // namespace Cfd
} // namespace AeroSim
