#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/cfd_result.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/diagnostics.hpp"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
namespace AeroSim {
namespace Cfd {

namespace {

__global__ void check_status_kernel(
    const int* d_failed,
    const Real* d_l2_sum,
    int nvar_ncells,
    Real convergence_tol,
    Real* d_residual_history_slot) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (*d_failed != 0 || !real_isfinite(*d_l2_sum)) {
        *d_residual_history_slot = -1.0f;
        return;
    }
    Real l2 = real_sqrt(*d_l2_sum / static_cast<Real>(nvar_ncells));
    *d_residual_history_slot = l2;
}

} // namespace

static void solve_gpu_free(int* d_failed, Real* d_min_dt, Real* d_l2_sum, Real* d_forces,
    Real* d_residual_history, Real* d_state_bounds_history, int* d_failure_cell, Real* d_failure_state) {
    cuda_free_safe(d_failed); cuda_free_safe(d_min_dt); cuda_free_safe(d_l2_sum); cuda_free_safe(d_forces);
    cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history);
    cuda_free_safe(d_failure_cell); cuda_free_safe(d_failure_state);
}

static CfdSolveSummary solve_gpu_impl(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    Real* d_min_dt,
    Real* d_l2_sum,
    Real* d_forces,
    Real* d_residual_history,
    Real* d_state_bounds_history,
    int* d_failure_cell,
    Real* d_failure_state,
    bool owned_buffers,
    std::string* error) {
    CfdSolveSummary summary;
    std::vector<Real> host_residual_history;
    int host_failed = 0;
    bool diagnostics_enabled = config.diagnostic_level != DiagnosticLevel::Off;

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * real_sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        w_inf.nu_tilde = condition.nu_tilde_ratio * mu_inf / w_inf.rho;
    }
    int nvar_ncells = DeviceMesh::NVAR * static_cast<int>(d_mesh.cell_count());

#ifdef MPI_ENABLED
    cudaStream_t stream_comp, stream_comm;
    cudaStreamCreate(&stream_comp);
    cudaStreamCreate(&stream_comm);
#endif

    for (int iter = 0; iter < config.max_iter; ++iter) {
#ifdef MPI_ENABLED
        if (d_mesh.has_halo()) {
            // exchange_halo_kernel<<<grid, 1, 0, stream_comm>>>(d_mesh.halo_send_device(), d_mesh.halo_recv_device());
        }
#endif
        if (config.reconstruction_order == 2) {
            if (!compute_gradients_gpu(d_mesh, config.gamma, error, d_failed)) goto fail;
            if (!compute_limiters_gpu(d_mesh, config.gamma, error, d_failed)) goto fail;
            if (!apply_limiter_gpu(d_mesh, false, error)) goto fail;
        }

        if (config.viscous && d_mesh.mu_device() == nullptr) {
            if (!d_mesh.allocate_viscous()) {
                if (error) *error = "allocate_viscous failed";
                goto fail;
            }
        }

        if (!compute_timestep_gpu(d_mesh, config.gamma, config.cfl, d_min_dt,
                config.viscous, d_mesh.mu_device(), config.Re)) {
            if (error) *error = "timestep kernel failed";
            goto fail;
        }

        if (!launch_euler_residual_kernel(d_mesh, w_inf, config.gamma, d_failed, nullptr, error,
                config.reconstruction_order)) {
            goto fail;
        }

if (config.viscous) {
            if (!compute_viscous_flux_gpu(d_mesh, config.gamma, config.prandtl,
                    config.mu_ref, config.T_ref, config.sutherland_T,
                    config.Re, config.turbulence ? 1 : 0, d_failed)) {
                if (error) *error = "viscous flux kernel failed";
                goto fail;
            }
        }

        if (config.turbulence) {
            if (!compute_rans_source_gpu(d_mesh, config.gamma, config.Re, d_failed, error)) {
                if (error && error->empty()) *error = "RANS source kernel failed";
                goto fail;
            }
            if (!apply_rans_implicit_gpu(d_mesh, config.Re, d_min_dt, error)) {
                if (error && error->empty()) *error = "RANS implicit kernel failed";
                goto fail;
            }
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

#ifdef MPI_ENABLED
        if (d_mesh.has_halo()) {
            cudaStreamSynchronize(stream_comm);
        }
#endif
    }

    if (!cuda_check(cudaDeviceSynchronize(), "post-loop sync", error)) goto fail;

    if (!cuda_check(cudaMemcpy(&host_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "read d_failed", error)) goto fail;

    host_residual_history.assign(config.max_iter, 0.0f);
    if (!cuda_check(cudaMemcpy(host_residual_history.data(), d_residual_history,
            config.max_iter * sizeof(Real), cudaMemcpyDeviceToHost), "read residual history", error)) goto fail;

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

    if (diagnostics_enabled) {
        std::vector<Real> bounds_host(config.max_iter * 6);
        if (!cuda_check(cudaMemcpy(bounds_host.data(), d_state_bounds_history,
                config.max_iter * 6 * sizeof(Real), cudaMemcpyDeviceToHost), "read state bounds history", error)) goto fail;

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
            Real host_failure_state[5] = {0.0f};
            if (!cuda_check(cudaMemcpy(&host_failure_cell, d_failure_cell, sizeof(int), cudaMemcpyDeviceToHost), "read d_failure_cell", error)) goto fail;
            if (!cuda_check(cudaMemcpy(host_failure_state, d_failure_state, 5 * sizeof(Real), cudaMemcpyDeviceToHost), "read d_failure_state", error)) goto fail;

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
        if (!compute_wall_forces_gpu(d_mesh, config.gamma, d_forces,
                config.viscous, config.prandtl, config.mu_ref, config.T_ref,
                config.sutherland_T, config.Re, config.wall_temperature)) {
            if (error) *error = "wall force kernel failed";
            goto fail;
        }
        if (!cuda_check(cudaDeviceSynchronize(), "wall force sync", error)) goto fail;

        Real forces[6];
        if (!cuda_check(cudaMemcpy(forces, d_forces, 6 * sizeof(Real), cudaMemcpyDeviceToHost), "read d_forces", error)) goto fail;

        Real q_inf = 0.5f * condition.mach * condition.mach;
        Real inv_force_ref = 1.0f / real_fmax(q_inf * config.ref_area, 1e-30f);
        summary.forces.CX = forces[0] * inv_force_ref;
        summary.forces.CY = forces[1] * inv_force_ref;
        summary.forces.CZ = forces[2] * inv_force_ref;
        summary.forces.Cl = forces[3] / real_fmax(q_inf * config.ref_area * config.ref_span, 1e-30f);
        summary.forces.Cm = forces[4] / real_fmax(q_inf * config.ref_area * config.ref_length, 1e-30f);
        summary.forces.Cn = forces[5] / real_fmax(q_inf * config.ref_area * config.ref_span, 1e-30f);

        constexpr Real kPi = 3.14159265358979323846;
        Real alpha = condition.alpha_deg * kPi / 180.0f;
        Real beta = condition.beta_deg * kPi / 180.0f;
        Real ca = real_cos(alpha);
        Real sa = real_sin(alpha);
        Real cb = real_cos(beta);
        Real sb = real_sin(beta);
        Real fsx = summary.forces.CX * ca * cb + summary.forces.CY * sb + summary.forces.CZ * sa * cb;
        Real fsz = -summary.forces.CX * sa + summary.forces.CZ * ca;
        summary.forces.CD = -fsx;
        summary.forces.CL = -fsz;

summary.forces.iterations = static_cast<int>(summary.residual_history.size());
        summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
        summary.forces.turbulence_model = config.turbulence ? "rans-sa" : "laminar";
        summary.forces.fidelity = "cfd-gpu";
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
    Real* d_min_dt = nullptr;
    Real* d_l2_sum = nullptr;
    Real* d_forces = nullptr;
    Real* d_residual_history = nullptr;
    Real* d_state_bounds_history = nullptr;
    int* d_failure_cell = nullptr;
    Real* d_failure_state = nullptr;

    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_min_dt, sizeof(Real)), "cudaMalloc d_min_dt", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_l2_sum, sizeof(Real)), "cudaMalloc d_l2_sum", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_forces, 6 * sizeof(Real)), "cudaMalloc d_forces", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(Real)), "cudaMalloc d_residual_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }

    bool diag = config.diagnostic_level != DiagnosticLevel::Off;
    if (diag) {
        if (!cuda_check(cudaMalloc(&d_state_bounds_history, config.max_iter * 6 * sizeof(Real)), "cudaMalloc d_state_bounds_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_cell, sizeof(int)), "cudaMalloc d_failure_cell", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_state, 5 * sizeof(Real)), "cudaMalloc d_failure_state", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
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
    Real* d_min_dt,
    Real* d_l2_sum,
    Real* d_forces,
    std::string* error) {
    if (d_mesh.cell_count() == 0 || d_mesh.face_count() == 0) {
        CfdSolveSummary s;
        if (error) *error = "DeviceMesh is empty";
        s.failed = true;
        return s;
    }

    Real* d_residual_history = nullptr;
    Real* d_state_bounds_history = nullptr;
    int* d_failure_cell = nullptr;
    Real* d_failure_state = nullptr;

    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(Real)), "cudaMalloc d_residual_history", error)) { cuda_free_safe(d_residual_history); CfdSolveSummary s; s.failed = true; return s; }

    bool diag = config.diagnostic_level != DiagnosticLevel::Off;
    if (diag) {
        if (!cuda_check(cudaMalloc(&d_state_bounds_history, config.max_iter * 6 * sizeof(Real)), "cudaMalloc d_state_bounds_history", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_cell, sizeof(int)), "cudaMalloc d_failure_cell", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); cuda_free_safe(d_failure_cell); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_state, 5 * sizeof(Real)), "cudaMalloc d_failure_state", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); cuda_free_safe(d_failure_cell); cuda_free_safe(d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMemset(d_failure_cell, 0xFF, sizeof(int)), "init d_failure_cell", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); cuda_free_safe(d_failure_cell); cuda_free_safe(d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    }

    CfdSolveSummary result = solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state, false, error);
    cuda_free_safe(d_residual_history);
    cuda_free_safe(d_state_bounds_history);
    cuda_free_safe(d_failure_cell);
    cuda_free_safe(d_failure_state);
    return result;
}

} // namespace Cfd
} // namespace AeroSim




