#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__device__ float atomic_min_float(float* addr, float val) {
    unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_int;
    unsigned int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomic_max_float(float* addr, float val) {
    unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_int;
    unsigned int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void init_bounds_slot_kernel(float* slot) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        slot[0] = FLT_MAX;
        slot[1] = -FLT_MAX;
        slot[2] = FLT_MAX;
        slot[3] = -FLT_MAX;
        slot[4] = FLT_MAX;
        slot[5] = -FLT_MAX;
    }
}

__global__ void state_bounds_kernel(
    const float* d_q, int n_cells, int nvar, float gamma,
    float* d_min_rho, float* d_max_rho,
    float* d_min_p, float* d_max_p,
    float* d_min_mach, float* d_max_mach) {
    __shared__ float s_min_rho[128], s_max_rho[128];
    __shared__ float s_min_p[128], s_max_p[128];
    __shared__ float s_min_mach[128], s_max_mach[128];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float l_min_rho = FLT_MAX, l_max_rho = -FLT_MAX;
    float l_min_p = FLT_MAX, l_max_p = -FLT_MAX;
    float l_min_mach = FLT_MAX, l_max_mach = -FLT_MAX;

    if (idx < n_cells) {
        float rho = d_q[idx * nvar + 0];
        float rhou = d_q[idx * nvar + 1];
        float rhov = d_q[idx * nvar + 2];
        float rhow = d_q[idx * nvar + 3];
        float rhoE = d_q[idx * nvar + 4];

        if (__finitef(rho) && rho > 0.0f) {
            l_min_rho = rho; l_max_rho = rho;

            float inv_rho = 1.0f / rho;
            float u = rhou * inv_rho;
            float v = rhov * inv_rho;
            float w = rhow * inv_rho;
            float ke = 0.5f * (u*u + v*v + w*w);
            float p = (gamma - 1.0f) * (rhoE - rho * ke);

            if (__finitef(p) && p > 0.0f) {
                l_min_p = p; l_max_p = p;
                float a = sqrtf(gamma * p * inv_rho);
                float vm = sqrtf(u*u + v*v + w*w);
                float mach = vm / fmaxf(a, 1e-30f);
                l_min_mach = mach; l_max_mach = mach;
            }
        }
    }

    s_min_rho[tid] = l_min_rho; s_max_rho[tid] = l_max_rho;
    s_min_p[tid] = l_min_p; s_max_p[tid] = l_max_p;
    s_min_mach[tid] = l_min_mach; s_max_mach[tid] = l_max_mach;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_rho[tid] = fminf(s_min_rho[tid], s_min_rho[tid + s]);
            s_max_rho[tid] = fmaxf(s_max_rho[tid], s_max_rho[tid + s]);
            s_min_p[tid] = fminf(s_min_p[tid], s_min_p[tid + s]);
            s_max_p[tid] = fmaxf(s_max_p[tid], s_max_p[tid + s]);
            s_min_mach[tid] = fminf(s_min_mach[tid], s_min_mach[tid + s]);
            s_max_mach[tid] = fmaxf(s_max_mach[tid], s_max_mach[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_min_float(d_min_rho, s_min_rho[0]);
        atomic_max_float(d_max_rho, s_max_rho[0]);
        atomic_min_float(d_min_p, s_min_p[0]);
        atomic_max_float(d_max_p, s_max_p[0]);
        atomic_min_float(d_min_mach, s_min_mach[0]);
        atomic_max_float(d_max_mach, s_max_mach[0]);
    }
}

__global__ void failure_snapshot_kernel(
    const float* d_q, int n_cells, int nvar, float gamma,
    int* d_failure_cell, float* d_failure_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    float rho = d_q[idx * nvar + 0];
    float rhou = d_q[idx * nvar + 1];
    float rhov = d_q[idx * nvar + 2];
    float rhow = d_q[idx * nvar + 3];
    float rhoE = d_q[idx * nvar + 4];

    if (!__finitef(rho) || rho <= 0.0f) {
        int old = atomicCAS(d_failure_cell, -1, idx);
        if (old == -1) {
            d_failure_state[0] = rho;
            d_failure_state[1] = rhou;
            d_failure_state[2] = rhov;
            d_failure_state[3] = rhow;
            d_failure_state[4] = rhoE;
        }
        return;
    }
    float inv_rho = 1.0f / rho;
    float u = rhou * inv_rho;
    float v = rhov * inv_rho;
    float w = rhow * inv_rho;
    float ke = 0.5f * (u*u + v*v + w*w);
    float p = (gamma - 1.0f) * (rhoE - rho * ke);
    if (!__finitef(p) || p <= 0.0f) {
        int old = atomicCAS(d_failure_cell, -1, idx);
        if (old == -1) {
            d_failure_state[0] = rho;
            d_failure_state[1] = rhou;
            d_failure_state[2] = rhov;
            d_failure_state[3] = rhow;
            d_failure_state[4] = rhoE;
        }
    }
}

} // namespace

bool compute_state_bounds_gpu(DeviceMesh& mesh, float gamma, float* d_bounds_slot) {
    init_bounds_slot_kernel<<<1, 1>>>(d_bounds_slot);
    if (!cuda_check(cudaGetLastError(), "init_bounds_slot launch")) return false;

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    state_bounds_kernel<<<grid, block>>>(
        mesh.state_device(), nc, DeviceMesh::NVAR, gamma,
        d_bounds_slot + 0, d_bounds_slot + 1,
        d_bounds_slot + 2, d_bounds_slot + 3,
        d_bounds_slot + 4, d_bounds_slot + 5);
    if (!cuda_check(cudaGetLastError(), "state_bounds_kernel launch")) return false;

    return true;
}

bool compute_failure_snapshot_gpu(DeviceMesh& mesh, float gamma,
    int* d_failure_cell, float* d_failure_state) {
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    failure_snapshot_kernel<<<grid, block>>>(
        mesh.state_device(), nc, DeviceMesh::NVAR, gamma,
        d_failure_cell, d_failure_state);
    if (!cuda_check(cudaGetLastError(), "failure_snapshot_kernel launch")) return false;

    return true;
}

} // namespace Cfd
} // namespace AeroSim
