#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_float6_zero_kernel(float* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 6; ++i) ptr[i] = 0.0f;
    }
}

__global__ void wall_force_kernel(
    const float* d_q, int nvar, float gamma,
    const float* d_nx, const float* d_ny, const float* d_nz,
    const float* d_area,
    const int* d_left_cell,
    const int* d_boundary,
    const float* d_cx, const float* d_cy, const float* d_cz,
    int face_count,
    float* d_forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= face_count) return;

    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::SlipWall) &&
        bnd != static_cast<int>(BoundaryKind::NoSlipWall)) return;

    int left = d_left_cell[idx];
    float rho = d_q[left * nvar + 0];
    if (rho <= 0.0f) return;
    float inv_rho = 1.0f / rho;
    float u = d_q[left * nvar + 1] * inv_rho;
    float v = d_q[left * nvar + 2] * inv_rho;
    float w = d_q[left * nvar + 3] * inv_rho;
    float E = d_q[left * nvar + 4];
    float kinetic = 0.5f * (u*u + v*v + w*w);
    float p = (gamma - 1.0f) * (E - rho * kinetic);
    if (p <= 0.0f) return;

    float nx = d_nx[idx];
    float ny = d_ny[idx];
    float nz = d_nz[idx];
    float area = d_area[idx];

    float px = -p * nx * area;
    float py = -p * ny * area;
    float pz = -p * nz * area;

    float fcx = d_cx[idx];
    float fcy = d_cy[idx];
    float fcz = d_cz[idx];

    atomicAdd(&d_forces[0], px);
    atomicAdd(&d_forces[1], py);
    atomicAdd(&d_forces[2], pz);
    atomicAdd(&d_forces[3], fcy * pz - fcz * py);
    atomicAdd(&d_forces[4], fcz * px - fcx * pz);
    atomicAdd(&d_forces[5], fcx * py - fcy * px);
}

} // namespace

bool compute_wall_forces_gpu(DeviceMesh& mesh, float gamma, float* d_forces) {
    init_float6_zero_kernel<<<1, 1>>>(d_forces);
    if (!cuda_check(cudaGetLastError(), "init_forces kernel launch")) return false;

    int block = 128;
    int grid = (mesh.face_count() + block - 1) / block;
    DeviceFaceData fd = mesh.face_data();

    wall_force_kernel<<<grid, block>>>(
        mesh.state_device(), DeviceMesh::NVAR, gamma,
        fd.nx, fd.ny, fd.nz, fd.area,
        fd.left_cell, fd.boundary,
        fd.cx, fd.cy, fd.cz,
        mesh.face_count(), d_forces);
    if (!cuda_check(cudaGetLastError(), "wall_force kernel launch")) return false;
    return true;
}

} // namespace Cfd
} // namespace AeroSim
