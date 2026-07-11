#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/real.hpp"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__global__ void perturb_kernel(Real* d_q_pert, const Real* d_q,
    const Real* d_v, Real eps, int n, int nvar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * nvar) return;
    d_q_pert[idx] = d_q[idx] + eps * d_v[idx];
}

__global__ void jfv_result_kernel(Real* d_result, const Real* d_residual_pert,
    const Real* d_residual, Real inv_eps, int n, int nvar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * nvar) return;
    d_result[idx] = (d_residual_pert[idx] - d_residual[idx]) * inv_eps;
}

} // namespace

bool compute_jfv_product(DeviceMesh& mesh, const Real* d_v, Real* d_result,
    const Real* d_residual, Real epsilon, const CfdConfig& config,
    const PrimitiveState& w_inf, Real* d_scratch, std::string* error) {
    int n = static_cast<int>(mesh.cell_count());
    int nvar = DeviceMesh::NVAR;
    int nvar_cells = n * nvar;

    int block = 128;
    int grid = (nvar_cells + block - 1) / block;
    if (grid < 1) grid = 1;

    Real* d_q_pert = d_scratch;

    perturb_kernel<<<grid, block>>>(d_q_pert, mesh.state_device(), d_v, epsilon, n, nvar);
    if (!cuda_check(cudaGetLastError(), "perturb kernel", error)) return false;

    Real* d_q_saved = mesh.state_device();
    Real* d_q_orig = d_scratch + nvar_cells;

    if (!cuda_check(cudaMemcpy(d_q_orig, d_q_saved, nvar_cells * sizeof(Real),
            cudaMemcpyDeviceToDevice), "save q", error)) return false;

    if (!cuda_check(cudaMemcpy(d_q_saved, d_q_pert, nvar_cells * sizeof(Real),
            cudaMemcpyDeviceToDevice), "set q = q_pert", error)) return false;

    if (!launch_euler_residual_kernel(mesh, w_inf, config.gamma, nullptr, nullptr, error,
            config.reconstruction_order)) {
        cudaMemcpy(d_q_saved, d_q_orig, nvar_cells * sizeof(Real), cudaMemcpyDeviceToDevice);
        return false;
    }

    if (config.viscous) {
        if (!compute_viscous_flux_gpu(mesh, config.gamma, config.prandtl,
                config.mu_ref, config.T_ref, config.sutherland_T,
                config.Re, config.wall_temperature, config.turbulence ? 1 : 0, nullptr)) {
            cudaMemcpy(d_q_saved, d_q_orig, nvar_cells * sizeof(Real), cudaMemcpyDeviceToDevice);
            if (error) *error = "JFV viscous flux failed";
            return false;
        }
    }

    Real* d_residual_pert = mesh.residual_device();

    Real inv_eps = 1.0f / epsilon;
    jfv_result_kernel<<<grid, block>>>(d_result, d_residual_pert, d_residual, inv_eps, n, nvar);
    if (!cuda_check(cudaGetLastError(), "jfv result kernel", error)) {
        cudaMemcpy(d_q_saved, d_q_orig, nvar_cells * sizeof(Real), cudaMemcpyDeviceToDevice);
        return false;
    }

    if (!cuda_check(cudaMemcpy(d_q_saved, d_q_orig, nvar_cells * sizeof(Real),
            cudaMemcpyDeviceToDevice), "restore q", error)) return false;

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
