#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"

#include <cfloat>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_float_max_kernel(float* ptr) {
    *ptr = FLT_MAX;
}

__global__ void timestep_kernel(
    const float* d_q, int n_cells, int nvar, float gamma, float cfl,
    const float* d_volume, const float* d_h_min,
    float* d_min_dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    float rho = d_q[idx * nvar + 0];
    if (!__finitef(rho) || rho <= 0.0f) return;
    float inv_rho = 1.0f / rho;
    float u = d_q[idx * nvar + 1] * inv_rho;
    float v = d_q[idx * nvar + 2] * inv_rho;
    float w = d_q[idx * nvar + 3] * inv_rho;
    float E = d_q[idx * nvar + 4];
    float kinetic = 0.5f * (u*u + v*v + w*w);
    float p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!__finitef(p) || p <= 0.0f) return;

    float vmag = sqrtf(u*u + v*v + w*w);
    float a = sqrtf(gamma * p / rho);
    float denom = vmag + a;
    float dt = cfl * d_h_min[idx] / (denom > 1e-30f ? denom : 1e-30f);
    unsigned int candidate = __float_as_int(dt);
    unsigned int* ptr = reinterpret_cast<unsigned int*>(d_min_dt);
    unsigned int old = atomicCAS(ptr, __float_as_int(FLT_MAX), candidate);
    while (old != __float_as_int(FLT_MAX) && candidate < old) {
        unsigned int prev = atomicCAS(ptr, old, candidate);
        if (prev == old) break;
        old = prev;
    }
}

} // namespace

bool compute_timestep_gpu(DeviceMesh& mesh, float gamma, float cfl, float* d_min_dt) {
    init_float_max_kernel<<<1, 1>>>(d_min_dt);
    if (!cuda_check(cudaGetLastError(), "init_float_max kernel launch")) return false;

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    timestep_kernel<<<grid, block>>>(
        mesh.state_device(), nc, DeviceMesh::NVAR, gamma, cfl,
        cd.volume, cd.h_min, d_min_dt);
    if (!cuda_check(cudaGetLastError(), "timestep kernel launch")) return false;
    return true;
}

} // namespace Cfd
} // namespace AeroSim
