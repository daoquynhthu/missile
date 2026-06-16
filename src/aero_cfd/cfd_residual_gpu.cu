#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/cuda_utils.hpp"

#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__device__ bool d_conservative_to_primitive(ConservativeState q, float gamma, PrimitiveState& w) {
    if (q.rho <= 0.0f || !isfinite(q.rho)) return false;
    float inv_rho = 1.0f / q.rho;
    w.rho = q.rho;
    w.u = q.rho_u * inv_rho;
    w.v = q.rho_v * inv_rho;
    w.w = q.rho_w * inv_rho;
    float kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    w.p = (gamma - 1.0f) * (q.rho_E - q.rho * kinetic);
    return isfinite(w.rho) && isfinite(w.u) && isfinite(w.v) && isfinite(w.w) &&
        isfinite(w.p) && w.p > 0.0f;
}

__device__ ConservativeState d_primitive_to_conservative(PrimitiveState w, float gamma) {
    float kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    ConservativeState q;
    q.rho = w.rho;
    q.rho_u = w.rho * w.u;
    q.rho_v = w.rho * w.v;
    q.rho_w = w.rho * w.w;
    q.rho_E = w.p / (gamma - 1.0f) + w.rho * kinetic;
    return q;
}

__device__ float d_speed_of_sound(PrimitiveState w, float gamma) {
    return sqrtf(gamma * w.p / w.rho);
}

__device__ EulerFlux d_physical_flux(PrimitiveState w, float gamma, float nx, float ny, float nz) {
    float vn = w.u*nx + w.v*ny + w.w*nz;
    float kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    float rho_E = w.p / (gamma - 1.0f) + w.rho * kinetic;

    EulerFlux f;
    f.mass = w.rho * vn;
    f.mom_x = w.rho * w.u * vn + w.p * nx;
    f.mom_y = w.rho * w.v * vn + w.p * ny;
    f.mom_z = w.rho * w.w * vn + w.p * nz;
    f.energy = (rho_E + w.p) * vn;
    return f;
}

__device__ EulerFlux d_slip_wall_flux(PrimitiveState w, float nx, float ny, float nz) {
    EulerFlux f;
    f.mass = 0.0f;
    f.mom_x = w.p * nx;
    f.mom_y = w.p * ny;
    f.mom_z = w.p * nz;
    f.energy = 0.0f;
    return f;
}

__device__ PrimitiveState d_farfield_ghost_state(PrimitiveState left, PrimitiveState freestream, float gamma,
    float nx, float ny, float nz) {
    float vn_inf = freestream.u*nx + freestream.v*ny + freestream.w*nz;
    float a_inf = d_speed_of_sound(freestream, gamma);
    if (vn_inf >= a_inf) return left;
    return freestream;
}

__device__ EulerFlux d_hllc_flux(PrimitiveState left, PrimitiveState right, float gamma, float nx, float ny, float nz) {
    float vn_l = left.u*nx + left.v*ny + left.w*nz;
    float vn_r = right.u*nx + right.v*ny + right.w*nz;
    float a_l = d_speed_of_sound(left, gamma);
    float a_r = d_speed_of_sound(right, gamma);
    float s_l = fminf(vn_l - a_l, vn_r - a_r);
    float s_r = fmaxf(vn_l + a_l, vn_r + a_r);

    EulerFlux f_l = d_physical_flux(left, gamma, nx, ny, nz);
    EulerFlux f_r = d_physical_flux(right, gamma, nx, ny, nz);
    ConservativeState q_l = d_primitive_to_conservative(left, gamma);
    ConservativeState q_r = d_primitive_to_conservative(right, gamma);

    if (s_l >= 0.0f) return f_l;
    if (s_r <= 0.0f) return f_r;

    float denom = left.rho * (s_l - vn_l) - right.rho * (s_r - vn_r);
    float s_m = (right.p - left.p + left.rho*vn_l*(s_l - vn_l) - right.rho*vn_r*(s_r - vn_r)) / denom;

    if (s_m >= 0.0f) {
        float rho_star = left.rho * (s_l - vn_l) / (s_l - s_m);
        float e_l = q_l.rho_E / left.rho;
        float e_star = e_l + (s_m - vn_l) * (s_m + left.p / (left.rho * (s_l - vn_l)));
        ConservativeState q_star;
        q_star.rho = rho_star;
        q_star.rho_u = rho_star * (left.u + (s_m - vn_l) * nx);
        q_star.rho_v = rho_star * (left.v + (s_m - vn_l) * ny);
        q_star.rho_w = rho_star * (left.w + (s_m - vn_l) * nz);
        q_star.rho_E = rho_star * e_star;

        EulerFlux f = f_l;
        f.mass += s_l * (q_star.rho - q_l.rho);
        f.mom_x += s_l * (q_star.rho_u - q_l.rho_u);
        f.mom_y += s_l * (q_star.rho_v - q_l.rho_v);
        f.mom_z += s_l * (q_star.rho_w - q_l.rho_w);
        f.energy += s_l * (q_star.rho_E - q_l.rho_E);
        return f;
    }

    float rho_star = right.rho * (s_r - vn_r) / (s_r - s_m);
    float e_r = q_r.rho_E / right.rho;
    float e_star = e_r + (s_m - vn_r) * (s_m + right.p / (right.rho * (s_r - vn_r)));
    ConservativeState q_star;
    q_star.rho = rho_star;
    q_star.rho_u = rho_star * (right.u + (s_m - vn_r) * nx);
    q_star.rho_v = rho_star * (right.v + (s_m - vn_r) * ny);
    q_star.rho_w = rho_star * (right.w + (s_m - vn_r) * nz);
    q_star.rho_E = rho_star * e_star;

    EulerFlux f = f_r;
    f.mass += s_r * (q_star.rho - q_r.rho);
    f.mom_x += s_r * (q_star.rho_u - q_r.rho_u);
    f.mom_y += s_r * (q_star.rho_v - q_r.rho_v);
    f.mom_z += s_r * (q_star.rho_w - q_r.rho_w);
    f.energy += s_r * (q_star.rho_E - q_r.rho_E);
    return f;
}

__device__ void d_add_scaled(EulerFlux* residual, int cell, EulerFlux flux, float scale) {
    atomicAdd(&residual[cell].mass, flux.mass * scale);
    atomicAdd(&residual[cell].mom_x, flux.mom_x * scale);
    atomicAdd(&residual[cell].mom_y, flux.mom_y * scale);
    atomicAdd(&residual[cell].mom_z, flux.mom_z * scale);
    atomicAdd(&residual[cell].energy, flux.energy * scale);
}

__global__ void euler_residual_kernel(
    const CfdFace* faces,
    int face_count,
    const ConservativeState* q,
    PrimitiveState freestream,
    float gamma,
    EulerFlux* residual,
    int* failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= face_count) return;

    CfdFace face = faces[idx];
    PrimitiveState wl;
    if (!d_conservative_to_primitive(q[face.left_cell], gamma, wl)) {
        atomicExch(failed, 1);
        return;
    }

    EulerFlux flux;
    if (face.boundary == BoundaryKind::Interior) {
        PrimitiveState wr;
        if (!d_conservative_to_primitive(q[face.right_cell], gamma, wr)) {
            atomicExch(failed, 1);
            return;
        }
        flux = d_hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
    } else if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall) {
        flux = d_slip_wall_flux(wl, face.nx, face.ny, face.nz);
    } else {
        PrimitiveState wr = d_farfield_ghost_state(wl, freestream, gamma, face.nx, face.ny, face.nz);
        flux = d_hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
    }

    d_add_scaled(residual, face.left_cell, flux, -face.area);
    if (face.boundary == BoundaryKind::Interior) {
        d_add_scaled(residual, face.right_cell, flux, face.area);
    }
}

bool launch_euler_residual_kernel(
    GpuCfdBuffers& buffers,
    const PrimitiveState& freestream,
    float gamma,
    int* d_failed,
    std::string* error) {
    if (!buffers.clear_residual(error)) return false;
    if (!cuda_check(cudaMemset(d_failed, 0, sizeof(int)), "cudaMemset failed", error)) return false;

    int block = 128;
    int grid = (buffers.face_count() + block - 1) / block;
    euler_residual_kernel<<<grid, block>>>(
        buffers.faces_device(),
        buffers.face_count(),
        buffers.state_device(),
        freestream,
        gamma,
        buffers.residual_device(),
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
    GpuCfdBuffers& buffers,
    const PrimitiveState& freestream,
    float gamma,
    std::string* error) {
    if (buffers.cell_count() <= 0 || buffers.face_count() <= 0 ||
        !buffers.faces_device() || !buffers.state_device() || !buffers.residual_device()) {
        if (error) *error = "GPU buffers are not ready";
        return false;
    }

    int* d_failed = nullptr;

    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc failed", error)) goto fail;
    if (!launch_euler_residual_kernel(buffers, freestream, gamma, d_failed, error)) goto fail;
    if (!cuda_check(cudaDeviceSynchronize(), "euler_residual_kernel synchronize", error)) goto fail;
    if (!read_kernel_failed_flag(d_failed, error)) goto fail;

    cudaFree(d_failed);
    return true;

fail:
    cudaFree(d_failed);
    return false;
}

bool compute_euler_residual_gpu_timed(
    GpuCfdBuffers& buffers,
    const PrimitiveState& freestream,
    float gamma,
    float* elapsed_ms,
    std::string* error) {
    if (elapsed_ms) *elapsed_ms = 0.0f;
    if (buffers.cell_count() <= 0 || buffers.face_count() <= 0 ||
        !buffers.faces_device() || !buffers.state_device() || !buffers.residual_device()) {
        if (error) *error = "GPU buffers are not ready";
        return false;
    }

    int* d_failed = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc failed", error)) goto fail;
    if (!cuda_check(cudaEventCreate(&start), "cudaEventCreate start", error)) goto fail;
    if (!cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop", error)) goto fail;
    if (!cuda_check(cudaEventRecord(start), "cudaEventRecord start", error)) goto fail;
    if (!launch_euler_residual_kernel(buffers, freestream, gamma, d_failed, error)) goto fail;
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
    GpuCfdBuffers buffers;
    if (!buffers.upload_mesh(mesh, error)) return false;
    if (!buffers.upload_state(q, error)) return false;
    if (!compute_euler_residual_gpu(buffers, freestream, gamma, error)) return false;
    return buffers.download_residual(residual, error);
}

std::size_t estimate_euler_residual_gpu_bytes(const CfdMesh& mesh) {
    std::size_t face_bytes = mesh.faces.size() * sizeof(CfdFace);
    std::size_t state_reads = 0;
    std::size_t residual_writes = 0;
    for (const auto& face : mesh.faces) {
        state_reads += sizeof(ConservativeState);
        residual_writes += sizeof(EulerFlux);
        if (face.boundary == BoundaryKind::Interior) {
            state_reads += sizeof(ConservativeState);
            residual_writes += sizeof(EulerFlux);
        }
    }
    return face_bytes + state_reads + residual_writes;
}

} // namespace Cfd
} // namespace AeroSim
