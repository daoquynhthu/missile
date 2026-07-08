#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/real.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
#include <limits>
namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_max_kernel(Real* ptr) {
    *ptr = std::numeric_limits<Real>::max();
}

__global__ void timestep_kernel(
    const Real* d_q, int n_cells, int nvar, Real gamma, Real cfl,
    const Real* d_volume, const Real* d_h_min,
    Real* d_min_dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (!real_isfinite(rho) || rho <= 0.0f) return;
    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real E = d_q[idx * nvar + 4];
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) return;

    Real vmag = real_sqrt(u*u + v*v + w*w);
    Real a = real_sqrt(gamma * p / rho);
    Real denom = vmag + a;
    Real dt = cfl * d_h_min[idx] / (denom > 1e-30f ? denom : 1e-30f);
    real_atomic_min(d_min_dt, dt);
}

} // namespace

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt) {
    init_max_kernel<<<1, 1>>>(d_min_dt);
    if (!cuda_check(cudaGetLastError(), "init_max kernel launch")) return false;

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




