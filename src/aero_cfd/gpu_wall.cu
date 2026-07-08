#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/real.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_float6_zero_kernel(Real* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 6; ++i) ptr[i] = 0.0f;
    }
}

__global__ void wall_force_kernel(
    const Real* d_q, int nvar, Real gamma,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_left_cell,
    const int* d_boundary,
    const Real* d_cx, const Real* d_cy, const Real* d_cz,
    int face_count, int n_cells,
    Real* d_forces,
    const Real* d_gradients,
    const Real* d_cell_cx, const Real* d_cell_cy, const Real* d_cell_cz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= face_count) return;

    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::SlipWall) &&
        bnd != static_cast<int>(BoundaryKind::NoSlipWall)) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    Real rho = d_q[left * nvar + 0];
    if (!real_isfinite(rho) || rho <= 0.0f) return;
    Real inv_rho = 1.0f / rho;
    Real u = d_q[left * nvar + 1] * inv_rho;
    Real v = d_q[left * nvar + 2] * inv_rho;
    Real w = d_q[left * nvar + 3] * inv_rho;
    Real E = d_q[left * nvar + 4];
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) return;

    if (d_gradients) {
        Real dr = d_cx[idx] - d_cell_cx[left];
        Real ds = d_cy[idx] - d_cell_cy[left];
        Real dt = d_cz[idx] - d_cell_cz[left];
        int g_base = left * DeviceMesh::NGRAD;
        p += d_gradients[g_base + 12] * dr;
        p += d_gradients[g_base + 13] * ds;
        p += d_gradients[g_base + 14] * dt;
        if (!real_isfinite(p) || p <= 0.0f) p = (gamma - 1.0f) * (E - rho * kinetic);
    }

    Real nx = d_nx[idx];
    Real ny = d_ny[idx];
    Real nz = d_nz[idx];
    Real area = d_area[idx];

    Real px = -p * nx * area;
    Real py = -p * ny * area;
    Real pz = -p * nz * area;

    Real fcx = d_cx[idx];
    Real fcy = d_cy[idx];
    Real fcz = d_cz[idx];

    real_atomic_add(&d_forces[0], px);
    real_atomic_add(&d_forces[1], py);
    real_atomic_add(&d_forces[2], pz);
    real_atomic_add(&d_forces[3], fcy * pz - fcz * py);
    real_atomic_add(&d_forces[4], fcz * px - fcx * pz);
    real_atomic_add(&d_forces[5], fcx * py - fcy * px);
}

} // namespace

bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces) {
    init_float6_zero_kernel<<<1, 1>>>(d_forces);
    if (!cuda_check(cudaGetLastError(), "init_forces kernel launch")) return false;

    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int grid = (nf + block - 1) / block;
    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();

    int nc = static_cast<int>(mesh.cell_count());
    wall_force_kernel<<<grid, block>>>(
        mesh.state_device(), DeviceMesh::NVAR, gamma,
        fd.nx, fd.ny, fd.nz, fd.area,
        fd.left_cell, fd.boundary,
        fd.cx, fd.cy, fd.cz,
        nf, nc, d_forces,
        mesh.gradients_device(),
        cd.cx, cd.cy, cd.cz);
    if (!cuda_check(cudaGetLastError(), "wall_force kernel launch")) return false;
    return true;
}

} // namespace Cfd
} // namespace AeroSim




