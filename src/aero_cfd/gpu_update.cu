#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/real.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"
#include <cmath>
#include <cuda_runtime.h>
namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_float_zero_kernel(Real* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *ptr = 0.0f;
}

__global__ void init_int_zero_kernel(int* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *ptr = 0;
}

__global__ void update_and_l2_kernel(
    Real* d_q,
    const Real* d_residual,
    const Real* d_volume,
    int n_cells, int nvar, const Real* d_min_dt, Real gamma,
    Real* d_l2_sum,
    int* d_failed,
    int* d_failure_cell, Real* d_failure_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real min_dt = *d_min_dt;

    Real old_rho = d_q[idx * nvar + 0];
    Real old_rhou = d_q[idx * nvar + 1];
    Real old_rhov = d_q[idx * nvar + 2];
    Real old_rhow = d_q[idx * nvar + 3];
    Real old_rhoE = d_q[idx * nvar + 4];

    Real scale = min_dt / d_volume[idx];

    Real new_rho = old_rho + scale * d_residual[idx * nvar + 0];
    Real new_rhou = old_rhou + scale * d_residual[idx * nvar + 1];
    Real new_rhov = old_rhov + scale * d_residual[idx * nvar + 2];
    Real new_rhow = old_rhow + scale * d_residual[idx * nvar + 3];
    Real new_rhoE = old_rhoE + scale * d_residual[idx * nvar + 4];

    if (!real_isfinite(new_rho) || new_rho <= 0.0f) {
        int old = atomicCAS(d_failed, 0, 1);
        if (old == 0 && d_failure_cell) {
            *d_failure_cell = idx;
            d_failure_state[0] = new_rho;
            d_failure_state[1] = new_rhou;
            d_failure_state[2] = new_rhov;
            d_failure_state[3] = new_rhow;
            d_failure_state[4] = new_rhoE;
        }
        return;
    }
    Real inv_rho = 1.0f / new_rho;
    Real u = new_rhou * inv_rho;
    Real v = new_rhov * inv_rho;
    Real w = new_rhow * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (new_rhoE - new_rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) {
        int old = atomicCAS(d_failed, 0, 1);
        if (old == 0 && d_failure_cell) {
            *d_failure_cell = idx;
            d_failure_state[0] = new_rho;
            d_failure_state[1] = new_rhou;
            d_failure_state[2] = new_rhov;
            d_failure_state[3] = new_rhow;
            d_failure_state[4] = new_rhoE;
        }
        return;
    }

    Real dr = new_rho - old_rho;
    Real d1 = new_rhou - old_rhou;
    Real d2 = new_rhov - old_rhov;
    Real d3 = new_rhow - old_rhow;
    Real d4 = new_rhoE - old_rhoE;
    Real cell_l2 = dr*dr + d1*d1 + d2*d2 + d3*d3 + d4*d4;
    real_atomic_add(d_l2_sum, cell_l2);

    d_q[idx * nvar + 0] = new_rho;
    d_q[idx * nvar + 1] = new_rhou;
    d_q[idx * nvar + 2] = new_rhov;
    d_q[idx * nvar + 3] = new_rhow;
    d_q[idx * nvar + 4] = new_rhoE;
}

} // namespace

bool compute_update_gpu(DeviceMesh& mesh, const Real* d_min_dt, Real gamma,
    Real* d_l2_sum, int* d_failed,
    int* d_failure_cell, Real* d_failure_state) {
    init_float_zero_kernel<<<1, 1>>>(d_l2_sum);
    if (!cuda_check(cudaGetLastError(), "init_l2 kernel launch")) return false;
    init_int_zero_kernel<<<1, 1>>>(d_failed);
    if (!cuda_check(cudaGetLastError(), "init_failed kernel launch")) return false;
    if (d_failure_cell) {
        if (!cuda_check(cudaMemset(d_failure_cell, 0xFF, sizeof(int)), "init_failure_cell memset")) return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    update_and_l2_kernel<<<grid, block>>>(
        mesh.state_device(), mesh.residual_device(), cd.volume,
        nc, DeviceMesh::NVAR, d_min_dt, gamma,
        d_l2_sum, d_failed,
        d_failure_cell, d_failure_state);
    if (!cuda_check(cudaGetLastError(), "update kernel launch")) return false;
    return true;
}

} // namespace Cfd
} // namespace AeroSim




