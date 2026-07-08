#include "aero_cfd/reconstruction.hpp"
#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__global__ void init_float_one_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 1.0f;
}

__device__ bool d_conservative_to_primitive(const float* q, int cell, int nvar, float gamma,
    float& rho, float& u, float& v, float& w, float& p) {
    rho = q[cell * nvar + 0];
    if (rho <= 0.0f || !__finitef(rho)) return false;
    float inv_rho = 1.0f / rho;
    u = q[cell * nvar + 1] * inv_rho;
    v = q[cell * nvar + 2] * inv_rho;
    w = q[cell * nvar + 3] * inv_rho;
    float kinetic = 0.5f * (u*u + v*v + w*w);
    p = (gamma - 1.0f) * (q[cell * nvar + 4] - rho * kinetic);
    return __finitef(u) && __finitef(v) && __finitef(w) && __finitef(p) && p > 0.0f;
}

__device__ PrimitiveGradient d_apply_limiter(PrimitiveGradient gradient, PrimitiveLimiter limiter) {
    gradient.drho_dx *= limiter.rho;
    gradient.drho_dy *= limiter.rho;
    gradient.drho_dz *= limiter.rho;
    gradient.du_dx *= limiter.u;
    gradient.du_dy *= limiter.u;
    gradient.du_dz *= limiter.u;
    gradient.dv_dx *= limiter.v;
    gradient.dv_dy *= limiter.v;
    gradient.dv_dz *= limiter.v;
    gradient.dw_dx *= limiter.w;
    gradient.dw_dy *= limiter.w;
    gradient.dw_dz *= limiter.w;
    gradient.dp_dx *= limiter.p;
    gradient.dp_dy *= limiter.p;
    gradient.dp_dz *= limiter.p;
    return gradient;
}

__global__ void apply_limiter_kernel(PrimitiveGradient* gradients, const PrimitiveLimiter* limiters, int cell_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cell_count) return;
    gradients[idx] = d_apply_limiter(gradients[idx], limiters[idx]);
}

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

__global__ void gg_gradient_kernel_atomic(
    const float* d_q,
    int nvar, int n_cells, int n_faces,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const float* d_nx, const float* d_ny, const float* d_nz, const float* d_area,
    const float* d_volume,
    float gamma,
    float* d_gradients,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    float rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    float nx = d_nx[idx], ny = d_ny[idx], nz = d_nz[idx];
    float area = d_area[idx];

    float rhoF, uF, vF, wF, pF;
    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        float rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        rhoF = 0.5f * (rhoL + rhoR);
        uF = 0.5f * (uL + uR);
        vF = 0.5f * (vL + vR);
        wF = 0.5f * (wL + wR);
        pF = 0.5f * (pL + pR);

        if (d_volume[right] <= 0.0f) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        float right_scale = -area / d_volume[right];
        float* gR = d_gradients + right * DeviceMesh::NGRAD;
        float drho_r = rhoF - rhoR;
        float du_r = uF - uR;
        float dv_r = vF - vR;
        float dw_r = wF - wR;
        float dp_r = pF - pR;
        atomicAdd(&gR[0], drho_r * nx * right_scale);
        atomicAdd(&gR[1], drho_r * ny * right_scale);
        atomicAdd(&gR[2], drho_r * nz * right_scale);
        atomicAdd(&gR[3], du_r * nx * right_scale);
        atomicAdd(&gR[4], du_r * ny * right_scale);
        atomicAdd(&gR[5], du_r * nz * right_scale);
        atomicAdd(&gR[6], dv_r * nx * right_scale);
        atomicAdd(&gR[7], dv_r * ny * right_scale);
        atomicAdd(&gR[8], dv_r * nz * right_scale);
        atomicAdd(&gR[9], dw_r * nx * right_scale);
        atomicAdd(&gR[10], dw_r * ny * right_scale);
        atomicAdd(&gR[11], dw_r * nz * right_scale);
        atomicAdd(&gR[12], dp_r * nx * right_scale);
        atomicAdd(&gR[13], dp_r * ny * right_scale);
        atomicAdd(&gR[14], dp_r * nz * right_scale);
    } else {
        rhoF = rhoL; uF = uL; vF = vL; wF = wL; pF = pL;
    }

    if (d_volume[left] <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    float left_scale = area / d_volume[left];
    float drho = rhoF - rhoL;
    float du = uF - uL;
    float dv = vF - vL;
    float dw = wF - wL;
    float dp = pF - pL;

    float* gL = d_gradients + left * DeviceMesh::NGRAD;
    atomicAdd(&gL[0], drho * nx * left_scale);
    atomicAdd(&gL[1], drho * ny * left_scale);
    atomicAdd(&gL[2], drho * nz * left_scale);
    atomicAdd(&gL[3], du * nx * left_scale);
    atomicAdd(&gL[4], du * ny * left_scale);
    atomicAdd(&gL[5], du * nz * left_scale);
    atomicAdd(&gL[6], dv * nx * left_scale);
    atomicAdd(&gL[7], dv * ny * left_scale);
    atomicAdd(&gL[8], dv * nz * left_scale);
    atomicAdd(&gL[9], dw * nx * left_scale);
    atomicAdd(&gL[10], dw * ny * left_scale);
    atomicAdd(&gL[11], dw * nz * left_scale);
    atomicAdd(&gL[12], dp * nx * left_scale);
    atomicAdd(&gL[13], dp * ny * left_scale);
    atomicAdd(&gL[14], dp * nz * left_scale);
}

__global__ void gg_gradient_kernel_colored(
    const float* d_q,
    int nvar, int n_cells,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const float* d_nx, const float* d_ny, const float* d_nz, const float* d_area,
    const float* d_volume,
    float gamma,
    int face_start, int face_end,
    float* d_gradients,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + face_start;
    if (idx >= face_end) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    float rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    float nx = d_nx[idx], ny = d_ny[idx], nz = d_nz[idx];
    float area = d_area[idx];

    float rhoF, uF, vF, wF, pF;
    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        float rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        rhoF = 0.5f * (rhoL + rhoR);
        uF = 0.5f * (uL + uR);
        vF = 0.5f * (vL + vR);
        wF = 0.5f * (wL + wR);
        pF = 0.5f * (pL + pR);

        if (d_volume[right] <= 0.0f) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        float right_scale = -area / d_volume[right];
        float* gR = d_gradients + right * DeviceMesh::NGRAD;
        float drho_r = rhoF - rhoR;
        float du_r = uF - uR;
        float dv_r = vF - vR;
        float dw_r = wF - wR;
        float dp_r = pF - pR;
        gR[0] += drho_r * nx * right_scale;
        gR[1] += drho_r * ny * right_scale;
        gR[2] += drho_r * nz * right_scale;
        gR[3] += du_r * nx * right_scale;
        gR[4] += du_r * ny * right_scale;
        gR[5] += du_r * nz * right_scale;
        gR[6] += dv_r * nx * right_scale;
        gR[7] += dv_r * ny * right_scale;
        gR[8] += dv_r * nz * right_scale;
        gR[9] += dw_r * nx * right_scale;
        gR[10] += dw_r * ny * right_scale;
        gR[11] += dw_r * nz * right_scale;
        gR[12] += dp_r * nx * right_scale;
        gR[13] += dp_r * ny * right_scale;
        gR[14] += dp_r * nz * right_scale;
    } else {
        rhoF = rhoL; uF = uL; vF = vL; wF = wL; pF = pL;
    }

    if (d_volume[left] <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    float left_scale = area / d_volume[left];
    float drho = rhoF - rhoL;
    float du = uF - uL;
    float dv = vF - vL;
    float dw = wF - wL;
    float dp = pF - pL;

    float* gL = d_gradients + left * DeviceMesh::NGRAD;
    gL[0] += drho * nx * left_scale;
    gL[1] += drho * ny * left_scale;
    gL[2] += drho * nz * left_scale;
    gL[3] += du * nx * left_scale;
    gL[4] += du * ny * left_scale;
    gL[5] += du * nz * left_scale;
    gL[6] += dv * nx * left_scale;
    gL[7] += dv * ny * left_scale;
    gL[8] += dv * nz * left_scale;
    gL[9] += dw * nx * left_scale;
    gL[10] += dw * ny * left_scale;
    gL[11] += dw * nz * left_scale;
    gL[12] += dp * nx * left_scale;
    gL[13] += dp * ny * left_scale;
    gL[14] += dp * nz * left_scale;
}

constexpr int kMINMAX_STRIDE = 10;

__global__ void init_minmax_kernel(
    const float* d_q, int nvar, int n_cells, float gamma,
    float* d_minmax,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    float rho, u, v, w, p;
    if (!d_conservative_to_primitive(d_q, idx, nvar, gamma, rho, u, v, w, p)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        float* m = d_minmax + idx * kMINMAX_STRIDE;
        m[0] = 1e10f;  m[1] = -1e10f;
        m[2] = 1e10f;  m[3] = -1e10f;
        m[4] = 1e10f;  m[5] = -1e10f;
        m[6] = 1e10f;  m[7] = -1e10f;
        m[8] = 1e10f;  m[9] = -1e10f;
        return;
    }
    float* m = d_minmax + idx * kMINMAX_STRIDE;
    m[0] = rho; m[1] = rho;
    m[2] = u;   m[3] = u;
    m[4] = v;   m[5] = v;
    m[6] = w;   m[7] = w;
    m[8] = p;   m[9] = p;
}

__global__ void update_minmax_kernel(
    const float* d_q, int nvar, int n_cells, int n_faces,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    float gamma,
    float* d_minmax,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;
    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::Interior)) return;
    int left = d_left_cell[idx];
    int right = d_right_cell[idx];
    if (left < 0 || left >= n_cells || right < 0 || right >= n_cells) return;

    float rhoL, uL, vL, wL, pL;
    float rhoR, uR, vR, wR, pR;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    auto update = [](float* min_addr, float* max_addr, float val) {
        atomic_min_float(min_addr, val);
        atomic_max_float(max_addr, val);
    };

    float* mL = d_minmax + left * kMINMAX_STRIDE;
    update(&mL[0], &mL[1], rhoR);
    update(&mL[2], &mL[3], uR);
    update(&mL[4], &mL[5], vR);
    update(&mL[6], &mL[7], wR);
    update(&mL[8], &mL[9], pR);

    float* mR = d_minmax + right * kMINMAX_STRIDE;
    update(&mR[0], &mR[1], rhoL);
    update(&mR[2], &mR[3], uL);
    update(&mR[4], &mR[5], vL);
    update(&mR[6], &mR[7], wL);
    update(&mR[8], &mR[9], pL);
}

__device__ float limiter_theta_device(float center, float reconstructed, float min_val, float max_val) {
    if (reconstructed > max_val) {
        float denom = reconstructed - center;
        if (denom <= 0.0f) return 0.0f;
        return fmaxf(0.0f, fminf(1.0f, (max_val - center) / denom));
    }
    if (reconstructed < min_val) {
        float denom = reconstructed - center;
        if (denom >= 0.0f) return 0.0f;
        return fmaxf(0.0f, fminf(1.0f, (min_val - center) / denom));
    }
    return 1.0f;
}

__global__ void bj_limiter_kernel(
    const float* d_q, int nvar, int n_cells, int n_faces,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const float* d_face_cx, const float* d_face_cy, const float* d_face_cz,
    const float* d_cx, const float* d_cy, const float* d_cz,
    float gamma,
    const float* d_gradients,
    const float* d_minmax,
    float* d_limiters,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    float rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    float dxL = d_face_cx[idx] - d_cx[left];
    float dyL = d_face_cy[idx] - d_cy[left];
    float dzL = d_face_cz[idx] - d_cz[left];

    const float* gL = d_gradients + left * DeviceMesh::NGRAD;
    float rec_rho = rhoL + gL[0]*dxL + gL[1]*dyL + gL[2]*dzL;
    float rec_u = uL + gL[3]*dxL + gL[4]*dyL + gL[5]*dzL;
    float rec_v = vL + gL[6]*dxL + gL[7]*dyL + gL[8]*dzL;
    float rec_w = wL + gL[9]*dxL + gL[10]*dyL + gL[11]*dzL;
    float rec_p = pL + gL[12]*dxL + gL[13]*dyL + gL[14]*dzL;

    const float* mL = d_minmax + left * kMINMAX_STRIDE;
    float t_rho = limiter_theta_device(rhoL, rec_rho, mL[0], mL[1]);
    float t_u = limiter_theta_device(uL, rec_u, mL[2], mL[3]);
    float t_v = limiter_theta_device(vL, rec_v, mL[4], mL[5]);
    float t_w = limiter_theta_device(wL, rec_w, mL[6], mL[7]);
    float t_p = limiter_theta_device(pL, rec_p, mL[8], mL[9]);

    float* limL = d_limiters + left * 5;
    atomic_min_float(&limL[0], t_rho);
    atomic_min_float(&limL[1], t_u);
    atomic_min_float(&limL[2], t_v);
    atomic_min_float(&limL[3], t_w);
    atomic_min_float(&limL[4], t_p);

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        float rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }

        float dxR = d_face_cx[idx] - d_cx[right];
        float dyR = d_face_cy[idx] - d_cy[right];
        float dzR = d_face_cz[idx] - d_cz[right];

        const float* gR = d_gradients + right * DeviceMesh::NGRAD;
        rec_rho = rhoR + gR[0]*dxR + gR[1]*dyR + gR[2]*dzR;
        rec_u = uR + gR[3]*dxR + gR[4]*dyR + gR[5]*dzR;
        rec_v = vR + gR[6]*dxR + gR[7]*dyR + gR[8]*dzR;
        rec_w = wR + gR[9]*dxR + gR[10]*dyR + gR[11]*dzR;
        rec_p = pR + gR[12]*dxR + gR[13]*dyR + gR[14]*dzR;

        const float* mR = d_minmax + right * kMINMAX_STRIDE;
        t_rho = limiter_theta_device(rhoR, rec_rho, mR[0], mR[1]);
        t_u = limiter_theta_device(uR, rec_u, mR[2], mR[3]);
        t_v = limiter_theta_device(vR, rec_v, mR[4], mR[5]);
        t_w = limiter_theta_device(wR, rec_w, mR[6], mR[7]);
        t_p = limiter_theta_device(pR, rec_p, mR[8], mR[9]);

        float* limR = d_limiters + right * 5;
        atomic_min_float(&limR[0], t_rho);
        atomic_min_float(&limR[1], t_u);
        atomic_min_float(&limR[2], t_v);
        atomic_min_float(&limR[3], t_w);
        atomic_min_float(&limR[4], t_p);
    }
}

} // namespace

bool compute_gradients_gpu(DeviceMesh& mesh, float gamma, std::string* error, int* d_failed) {
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) return true;
    if (!mesh.gradients_device()) {
        if (error) *error = "gradients buffer not allocated";
        return false;
    }

    std::size_t grad_bytes = DeviceMesh::NGRAD * mesh.cell_count() * sizeof(float);
    if (!cuda_check(cudaMemset(mesh.gradients_device(), 0, grad_bytes), "cudaMemset gradients", error)) return false;

    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int nc = static_cast<int>(mesh.cell_count());
    int n_colors = mesh.color_count();
    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();

    if (n_colors > 0) {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c + 1];
            int nf_c  = end - start;
            int grid_c = (nf_c + block - 1) / block;
            gg_gradient_kernel_colored<<<grid_c, block>>>(
                mesh.state_device(),
                DeviceMesh::NVAR, nc,
                fd.left_cell, fd.right_cell, fd.boundary,
                fd.nx, fd.ny, fd.nz, fd.area,
                cd.volume, gamma,
                start, end,
                mesh.gradients_device(),
                d_failed);
            if (!cuda_check(cudaGetLastError(), "gg_gradient_kernel_colored", error)) return false;
        }
    } else {
        int grid = (nf + block - 1) / block;
        gg_gradient_kernel_atomic<<<grid, block>>>(
            mesh.state_device(),
            DeviceMesh::NVAR, nc, nf,
            fd.left_cell, fd.right_cell, fd.boundary,
            fd.nx, fd.ny, fd.nz, fd.area,
            cd.volume, gamma,
            mesh.gradients_device(),
            d_failed);
        if (!cuda_check(cudaGetLastError(), "gg_gradient_kernel_atomic", error)) return false;
    }
    if (!cuda_check(cudaDeviceSynchronize(), "gg_gradient_kernel synchronize", error)) return false;

    if (d_failed) {
        int host_failed = 0;
        if (!cuda_check(cudaMemcpy(&host_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost),
                "gg_gradient read d_failed", error)) return false;
        if (host_failed) {
            if (error) *error = "invalid cell state detected in gradient computation";
            return false;
        }
    }
    return true;
}

bool compute_limiters_gpu(DeviceMesh& mesh, float gamma, std::string* error, int* d_failed) {
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) return true;
    if (!mesh.gradients_device() || !mesh.limiters_device()) {
        if (error) *error = "gradient/limiter buffers not allocated";
        return false;
    }

    int nc = static_cast<int>(mesh.cell_count());
    int nf = static_cast<int>(mesh.face_count());
    std::size_t minmax_bytes = kMINMAX_STRIDE * static_cast<std::size_t>(nc) * sizeof(float);

    float* d_minmax = nullptr;
    if (!cuda_check(cudaMalloc(&d_minmax, minmax_bytes), "cudaMalloc minmax", error)) return false;

    int block = 128;
    int cell_grid = (nc + block - 1) / block;
    int face_grid = (nf + block - 1) / block;

    init_minmax_kernel<<<cell_grid, block>>>(
        mesh.state_device(), DeviceMesh::NVAR, nc, gamma, d_minmax, d_failed);
    if (!cuda_check(cudaGetLastError(), "init_minmax_kernel", error)) { cudaFree(d_minmax); return false; }

    update_minmax_kernel<<<face_grid, block>>>(
        mesh.state_device(), DeviceMesh::NVAR, nc, nf,
        mesh.face_data().left_cell, mesh.face_data().right_cell, mesh.face_data().boundary,
        gamma, d_minmax, d_failed);
    if (!cuda_check(cudaGetLastError(), "update_minmax_kernel", error)) { cudaFree(d_minmax); return false; }

    int limiter_grid = (nc * 5 + block - 1) / block;
    init_float_one_kernel<<<limiter_grid, block>>>(mesh.limiters_device(), nc * 5);
    if (!cuda_check(cudaGetLastError(), "init_float_one limiters", error)) { cudaFree(d_minmax); return false; }

    bj_limiter_kernel<<<face_grid, block>>>(
        mesh.state_device(), DeviceMesh::NVAR, nc, nf,
        mesh.face_data().left_cell, mesh.face_data().right_cell, mesh.face_data().boundary,
        mesh.face_data().cx, mesh.face_data().cy, mesh.face_data().cz,
        mesh.cell_data().cx, mesh.cell_data().cy, mesh.cell_data().cz,
        gamma,
        mesh.gradients_device(), d_minmax,
        mesh.limiters_device(), d_failed);
    if (!cuda_check(cudaGetLastError(), "bj_limiter_kernel", error)) { cudaFree(d_minmax); return false; }
    cudaFree(d_minmax);

    if (d_failed) {
        int host_failed = 0;
        if (!cuda_check(cudaMemcpy(&host_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost),
                "limiter read d_failed", error)) return false;
        if (host_failed) {
            if (error) *error = "invalid cell state detected in limiter computation";
            return false;
        }
    }
    return true;
}

bool apply_limiter_gpu(DeviceMesh& mesh, std::string* error) {
    return apply_limiter_gpu(mesh, true, error);
}

bool apply_limiter_gpu(DeviceMesh& mesh, bool sync, std::string* error) {
    if (mesh.cell_count() == 0 || !mesh.gradients_device() || !mesh.limiters_device()) {
        if (error) *error = "GPU gradient/limiter buffers are not ready";
        return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    apply_limiter_kernel<<<grid, block>>>(
        reinterpret_cast<PrimitiveGradient*>(mesh.gradients_device()),
        reinterpret_cast<const PrimitiveLimiter*>(mesh.limiters_device()),
        nc);
    if (!cuda_check(cudaGetLastError(), "apply_limiter_kernel launch", error)) return false;
    if (sync && !cuda_check(cudaDeviceSynchronize(), "apply_limiter_kernel synchronize", error)) return false;
    return true;
}

} // namespace Cfd
} // namespace AeroSim
