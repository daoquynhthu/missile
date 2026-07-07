#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/cuda_utils.hpp"

#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__device__ bool d_conservative_to_primitive(const float* q, int cell, int nvar, float gamma, float& rho, float& u, float& v, float& w, float& p) {
    rho = q[cell * nvar + 0];
    if (rho <= 0.0f || !isfinite(rho)) return false;
    float inv_rho = 1.0f / rho;
    u = q[cell * nvar + 1] * inv_rho;
    v = q[cell * nvar + 2] * inv_rho;
    w = q[cell * nvar + 3] * inv_rho;
    float kinetic = 0.5f * (u*u + v*v + w*w);
    p = (gamma - 1.0f) * (q[cell * nvar + 4] - rho * kinetic);
    return isfinite(rho) && isfinite(u) && isfinite(v) && isfinite(w) && isfinite(p) && p > 0.0f;
}

__device__ float d_speed_of_sound(float rho, float p, float gamma) {
    return sqrtf(gamma * p / rho);
}

__device__ void d_physical_flux(float rho, float u, float v, float w, float p, float gamma,
    float nx, float ny, float nz, float& mass, float& mom_x, float& mom_y, float& mom_z, float& energy) {
    float vn = u*nx + v*ny + w*nz;
    float kinetic = 0.5f * (u*u + v*v + w*w);
    float rho_E = p / (gamma - 1.0f) + rho * kinetic;
    mass = rho * vn;
    mom_x = rho * u * vn + p * nx;
    mom_y = rho * v * vn + p * ny;
    mom_z = rho * w * vn + p * nz;
    energy = (rho_E + p) * vn;
}

__device__ void d_slip_wall_flux(float p, float nx, float ny, float nz,
    float& mass, float& mom_x, float& mom_y, float& mom_z, float& energy) {
    mass = 0.0f;
    mom_x = p * nx;
    mom_y = p * ny;
    mom_z = p * nz;
    energy = 0.0f;
}

__device__ void d_farfield_ghost_state(float /*left_rho*/, float left_u, float left_v, float left_w, float /*left_p*/,
    float /*left_a*/, float inf_u, float inf_v, float inf_w, float inf_a,
    float nx, float ny, float nz,
    float& ghost_u, float& ghost_v, float& ghost_w) {
    float vn_inf = inf_u*nx + inf_v*ny + inf_w*nz;
    if (vn_inf >= inf_a) {
        ghost_u = left_u;
        ghost_v = left_v;
        ghost_w = left_w;
    } else {
        ghost_u = inf_u;
        ghost_v = inf_v;
        ghost_w = inf_w;
    }
}

__device__ void d_hllc_flux(
    float rhoL, float uL, float vL, float wL, float pL,
    float rhoR, float uR, float vR, float wR, float pR,
    float gamma, float nx, float ny, float nz,
    float& mass, float& mom_x, float& mom_y, float& mom_z, float& energy) {
    float vn_l = uL*nx + vL*ny + wL*nz;
    float vn_r = uR*nx + vR*ny + wR*nz;
    float a_l = d_speed_of_sound(rhoL, pL, gamma);
    float a_r = d_speed_of_sound(rhoR, pR, gamma);
    float s_l = fminf(vn_l - a_l, vn_r - a_r);
    float s_r = fmaxf(vn_l + a_l, vn_r + a_r);

    float fL_mass, fL_mx, fL_my, fL_mz, fL_en;
    float fR_mass, fR_mx, fR_my, fR_mz, fR_en;
    d_physical_flux(rhoL, uL, vL, wL, pL, gamma, nx, ny, nz, fL_mass, fL_mx, fL_my, fL_mz, fL_en);
    d_physical_flux(rhoR, uR, vR, wR, pR, gamma, nx, ny, nz, fR_mass, fR_mx, fR_my, fR_mz, fR_en);

    if (s_l >= 0.0f) { mass = fL_mass; mom_x = fL_mx; mom_y = fL_my; mom_z = fL_mz; energy = fL_en; return; }
    if (s_r <= 0.0f) { mass = fR_mass; mom_x = fR_mx; mom_y = fR_my; mom_z = fR_mz; energy = fR_en; return; }

    float denom = rhoL * (s_l - vn_l) - rhoR * (s_r - vn_r);
    float s_m = (pR - pL + rhoL*vn_l*(s_l - vn_l) - rhoR*vn_r*(s_r - vn_r)) / denom;

    if (s_m >= 0.0f) {
        float rho_star = rhoL * (s_l - vn_l) / (s_l - s_m);
        float kineticL = 0.5f * (uL*uL + vL*vL + wL*wL);
        float e_l = pL / ((gamma - 1.0f) * rhoL) + kineticL;
        float e_star = e_l + (s_m - vn_l) * (s_m + pL / (rhoL * (s_l - vn_l)));
        float qL_rho = rhoL;
        float qL_rhou = rhoL * uL;
        float qL_rhov = rhoL * vL;
        float qL_rhow = rhoL * wL;
        float qL_rhoE = pL / (gamma - 1.0f) + rhoL * kineticL;

        float qs_rho = rho_star;
        float qs_rhou = rho_star * (uL + (s_m - vn_l) * nx);
        float qs_rhov = rho_star * (vL + (s_m - vn_l) * ny);
        float qs_rhow = rho_star * (wL + (s_m - vn_l) * nz);
        float qs_rhoE = rho_star * e_star;

        mass = fL_mass + s_l * (qs_rho - qL_rho);
        mom_x = fL_mx + s_l * (qs_rhou - qL_rhou);
        mom_y = fL_my + s_l * (qs_rhov - qL_rhov);
        mom_z = fL_mz + s_l * (qs_rhow - qL_rhow);
        energy = fL_en + s_l * (qs_rhoE - qL_rhoE);
    } else {
        float rho_star = rhoR * (s_r - vn_r) / (s_r - s_m);
        float kineticR = 0.5f * (uR*uR + vR*vR + wR*wR);
        float e_r = pR / ((gamma - 1.0f) * rhoR) + kineticR;
        float e_star = e_r + (s_m - vn_r) * (s_m + pR / (rhoR * (s_r - vn_r)));
        float qR_rho = rhoR;
        float qR_rhou = rhoR * uR;
        float qR_rhov = rhoR * vR;
        float qR_rhow = rhoR * wR;
        float qR_rhoE = pR / (gamma - 1.0f) + rhoR * kineticR;

        float qs_rho = rho_star;
        float qs_rhou = rho_star * (uR + (s_m - vn_r) * nx);
        float qs_rhov = rho_star * (vR + (s_m - vn_r) * ny);
        float qs_rhow = rho_star * (wR + (s_m - vn_r) * nz);
        float qs_rhoE = rho_star * e_star;

        mass = fR_mass + s_r * (qs_rho - qR_rho);
        mom_x = fR_mx + s_r * (qs_rhou - qR_rhou);
        mom_y = fR_my + s_r * (qs_rhov - qR_rhov);
        mom_z = fR_mz + s_r * (qs_rhow - qR_rhow);
        energy = fR_en + s_r * (qs_rhoE - qR_rhoE);
    }
}

__global__ void euler_residual_kernel(
    const float* d_nx, const float* d_ny, const float* d_nz,
    const float* d_area,
    const int* d_left_cell, const int* d_right_cell,
    const int* d_boundary,
    const float* d_q,
    int face_count, int nvar,
    float gamma,
    float inf_u, float inf_v, float inf_w, float inf_a,
    float* d_residual,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= face_count) return;

    int left = d_left_cell[idx];
    int bnd = d_boundary[idx];
    float nx = d_nx[idx];
    float ny = d_ny[idx];
    float nz = d_nz[idx];
    float area = d_area[idx];

    float rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        atomicExch(d_failed, 1);
        return;
    }

    float mass, mom_x, mom_y, mom_z, energy;

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        float rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            atomicExch(d_failed, 1);
            return;
        }
        d_hllc_flux(rhoL, uL, vL, wL, pL, rhoR, uR, vR, wR, pR, gamma, nx, ny, nz,
            mass, mom_x, mom_y, mom_z, energy);
    } else if (bnd == static_cast<int>(BoundaryKind::SlipWall) || bnd == static_cast<int>(BoundaryKind::NoSlipWall)) {
        d_slip_wall_flux(pL, nx, ny, nz, mass, mom_x, mom_y, mom_z, energy);
    } else {
        float aL = d_speed_of_sound(rhoL, pL, gamma);
        float ghu, ghv, ghw;
        d_farfield_ghost_state(rhoL, uL, vL, wL, pL, aL, inf_u, inf_v, inf_w, inf_a, nx, ny, nz, ghu, ghv, ghw);
        d_hllc_flux(rhoL, uL, vL, wL, pL, 1.0f, ghu, ghv, ghw, 1.0f / gamma, gamma, nx, ny, nz,
            mass, mom_x, mom_y, mom_z, energy);
    }

    float fmass = mass * area;
    float fmx = mom_x * area;
    float fmy = mom_y * area;
    float fmz = mom_z * area;
    float fen = energy * area;

    atomicAdd(&d_residual[left * nvar + 0], -fmass);
    atomicAdd(&d_residual[left * nvar + 1], -fmx);
    atomicAdd(&d_residual[left * nvar + 2], -fmy);
    atomicAdd(&d_residual[left * nvar + 3], -fmz);
    atomicAdd(&d_residual[left * nvar + 4], -fen);

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        atomicAdd(&d_residual[right * nvar + 0], fmass);
        atomicAdd(&d_residual[right * nvar + 1], fmx);
        atomicAdd(&d_residual[right * nvar + 2], fmy);
        atomicAdd(&d_residual[right * nvar + 3], fmz);
        atomicAdd(&d_residual[right * nvar + 4], fen);
    }
}

bool launch_euler_residual_kernel(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    float gamma,
    int* d_failed,
    std::string* error) {
    if (!mesh.clear_residual(error)) return false;
    if (!cuda_check(cudaMemset(d_failed, 0, sizeof(int)), "cudaMemset failed", error)) return false;

    DeviceFaceData fd = mesh.face_data();
    float a_inf = speed_of_sound(freestream, gamma);

    int block = 128;
    int grid = (mesh.face_count() + block - 1) / block;
    euler_residual_kernel<<<grid, block>>>(
        fd.nx, fd.ny, fd.nz, fd.area,
        fd.left_cell, fd.right_cell, fd.boundary,
        mesh.state_device(),
        mesh.face_count(), DeviceMesh::NVAR,
        gamma,
        freestream.u, freestream.v, freestream.w, a_inf,
        mesh.residual_device(),
        d_failed);
    if (!cuda_check(cudaGetLastError(), "euler_residual_kernel launch", error)) return false;
    return true;
}

bool read_kernel_failed_flag(int* d_failed, std::string* error) {
    int failed = 0;
    if (!cuda_check(cudaMemcpy(&failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed", error)) return false;
    if (failed != 0) {
        if (error) *error = "GPU residual encountered invalid state";
        return false;
    }
    return true;
}

} // namespace

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    float gamma,
    std::string* error) {
    if (mesh.cell_count() <= 0 || mesh.face_count() <= 0) {
        if (error) *error = "DeviceMesh is not ready";
        return false;
    }

    int* d_failed = nullptr;
    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc failed", error)) return false;
    if (!launch_euler_residual_kernel(mesh, freestream, gamma, d_failed, error)) {
        cudaFree(d_failed);
        return false;
    }
    if (!cuda_check(cudaDeviceSynchronize(), "euler_residual_kernel synchronize", error)) {
        cudaFree(d_failed);
        return false;
    }
    if (!read_kernel_failed_flag(d_failed, error)) {
        cudaFree(d_failed);
        return false;
    }
    cudaFree(d_failed);
    return true;
}

bool compute_euler_residual_gpu_timed(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    float gamma,
    float* elapsed_ms,
    std::string* error) {
    if (elapsed_ms) *elapsed_ms = 0.0f;
    if (mesh.cell_count() <= 0 || mesh.face_count() <= 0) {
        if (error) *error = "DeviceMesh is not ready";
        return false;
    }

    int* d_failed = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc failed", error)) goto fail;
    if (!cuda_check(cudaEventCreate(&start), "cudaEventCreate start", error)) goto fail;
    if (!cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop", error)) goto fail;
    if (!cuda_check(cudaEventRecord(start), "cudaEventRecord start", error)) goto fail;
    if (!launch_euler_residual_kernel(mesh, freestream, gamma, d_failed, error)) goto fail;
    if (!cuda_check(cudaEventRecord(stop), "cudaEventRecord stop", error)) goto fail;
    if (!cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop", error)) goto fail;
    if (elapsed_ms) {
        if (!cuda_check(cudaEventElapsedTime(elapsed_ms, start, stop), "cudaEventElapsedTime", error)) goto fail;
    }
    if (!read_kernel_failed_flag(d_failed, error)) goto fail;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_failed);
    return true;

fail:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_failed);
    return false;
}

bool compute_euler_residual_gpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    float gamma,
    std::vector<EulerFlux>& residual,
    std::string* error) {
    DeviceMesh device_mesh;
    if (!device_mesh.upload_mesh(mesh, error)) return false;
    if (!device_mesh.upload_state(q, error)) return false;
    if (!compute_euler_residual_gpu(device_mesh, freestream, gamma, error)) return false;
    return device_mesh.download_residual(residual, error);
}

std::size_t estimate_euler_residual_gpu_bytes(const CfdMesh& mesh) {
    std::size_t face_bytes = mesh.faces.size() * 7 * sizeof(float);
    std::size_t state_reads = 0;
    for (const auto& face : mesh.faces) {
        state_reads += DeviceMesh::NVAR * sizeof(float);
        if (face.boundary == BoundaryKind::Interior) {
            state_reads += DeviceMesh::NVAR * sizeof(float);
        }
    }
    std::size_t residual_writes = state_reads;
    return face_bytes + state_reads + residual_writes;
}

} // namespace Cfd
} // namespace AeroSim
