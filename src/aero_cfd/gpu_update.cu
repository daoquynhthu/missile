#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"

#include <cmath>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_float_zero_kernel(float* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *ptr = 0.0f;
}

__global__ void init_int_zero_kernel(int* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *ptr = 0;
}

__global__ void update_and_l2_kernel(
    float* d_q,
    const float* d_residual,
    const float* d_volume,
    int n_cells, int nvar, const float* d_min_dt, float gamma,
    float* d_l2_sum,
    int* d_failed,
    int* d_failure_cell, float* d_failure_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    float min_dt = *d_min_dt;

    float old_rho = d_q[idx * nvar + 0];
    float old_rhou = d_q[idx * nvar + 1];
    float old_rhov = d_q[idx * nvar + 2];
    float old_rhow = d_q[idx * nvar + 3];
    float old_rhoE = d_q[idx * nvar + 4];

    float scale = min_dt / d_volume[idx];

    float new_rho = old_rho + scale * d_residual[idx * nvar + 0];
    float new_rhou = old_rhou + scale * d_residual[idx * nvar + 1];
    float new_rhov = old_rhov + scale * d_residual[idx * nvar + 2];
    float new_rhow = old_rhow + scale * d_residual[idx * nvar + 3];
    float new_rhoE = old_rhoE + scale * d_residual[idx * nvar + 4];

    if (!__finitef(new_rho) || new_rho <= 0.0f) {
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
    float inv_rho = 1.0f / new_rho;
    float u = new_rhou * inv_rho;
    float v = new_rhov * inv_rho;
    float w = new_rhow * inv_rho;
    float kinetic = 0.5f * (u*u + v*v + w*w);
    float p = (gamma - 1.0f) * (new_rhoE - new_rho * kinetic);
    if (!__finitef(p) || p <= 0.0f) {
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

    float dr = new_rho - old_rho;
    float d1 = new_rhou - old_rhou;
    float d2 = new_rhov - old_rhov;
    float d3 = new_rhow - old_rhow;
    float d4 = new_rhoE - old_rhoE;
    float cell_l2 = dr*dr + d1*d1 + d2*d2 + d3*d3 + d4*d4;
    atomicAdd(d_l2_sum, cell_l2);

    d_q[idx * nvar + 0] = new_rho;
    d_q[idx * nvar + 1] = new_rhou;
    d_q[idx * nvar + 2] = new_rhov;
    d_q[idx * nvar + 3] = new_rhow;
    d_q[idx * nvar + 4] = new_rhoE;
}

} // namespace

bool compute_update_gpu(DeviceMesh& mesh, const float* d_min_dt, float gamma,
    float* d_l2_sum, int* d_failed,
    int* d_failure_cell, float* d_failure_state) {
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
