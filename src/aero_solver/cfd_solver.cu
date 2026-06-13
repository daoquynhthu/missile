#include "aero_solver/cfd_solver.hpp"
#include "aero_solver/aero_skin_friction.hpp"
#include <cstdio>
#include <cmath>
#include <stdexcept>

// CUDA error check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

#define CUDA_KERNEL_CHECK() do { \
    cudaError_t err = cudaPeekAtLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

namespace AeroSim {
namespace Solver {

// ─── Device math helpers ──────────────────────────────────────────────────

__device__ inline float sqf(float x) { return x * x; }

// Conservative to primitive: Q = [ρ, ρu, ρv, ρw, ρE] → [ρ, u, v, w, p]
__device__ inline void cons_to_prim(const float* Q, float gamma, float* prim) {
    float rho = Q[0];
    float inv_rho = 1.0f / fmaxf(rho, 1e-20f);
    float u = Q[1] * inv_rho;
    float v = Q[2] * inv_rho;
    float w = Q[3] * inv_rho;
    float E = Q[4] * inv_rho;
    float ke = 0.5f * (u*u + v*v + w*w);
    float p = (gamma - 1.0f) * rho * (E - ke);
    p = fmaxf(p, 1e-20f);
    prim[0] = rho; prim[1] = u; prim[2] = v; prim[3] = w; prim[4] = p;
}

// Speed of sound
__device__ inline float speed_of_sound(float gamma, float p, float rho) {
    return sqrtf(gamma * p / fmaxf(rho, 1e-20f));
}

// ─── HLLC Riemann flux ────────────────────────────────────────────────────

__device__ inline void hllc_flux(
    const float* QL, const float* QR,
    float nx, float ny, float nz,
    float gamma,
    float* flux)
{
    float rhoL = QL[0], rhoR = QR[0];
    float uL = QL[1]/rhoL, vL = QL[2]/rhoL, wL = QL[3]/rhoL;
    float uR = QR[1]/rhoR, vR = QR[2]/rhoR, wR = QR[3]/rhoR;
    float EL = QL[4]/rhoL, ER = QR[4]/rhoR;
    float vnL = uL*nx + vL*ny + wL*nz;
    float vnR = uR*nx + vR*ny + wR*nz;
    float keL = 0.5f*(uL*uL + vL*vL + wL*wL);
    float keR = 0.5f*(uR*uR + vR*vR + wR*wR);
    float pL = (gamma-1.0f)*rhoL*(EL - keL);
    float pR = (gamma-1.0f)*rhoR*(ER - keR);
    pL = fmaxf(pL, 1e-20f); pR = fmaxf(pR, 1e-20f);
    float aL = sqrtf(gamma*pL/rhoL);
    float aR = sqrtf(gamma*pR/rhoR);

    // HLL wave speeds (Davis estimate)
    float SL = fminf(vnL - aL, vnR - aR);
    float SR = fmaxf(vnL + aL, vnR + aR);
    float SM = (pR - pL + rhoL*vnL*(SL - vnL) - rhoR*vnR*(SR - vnR))
             / (rhoL*(SL - vnL) - rhoR*(SR - vnR) + 1e-20f);

    float rho_starL = rhoL * (SL - vnL) / (SL - SM + 1e-20f);
    float rho_starR = rhoR * (SR - vnR) / (SR - SM + 1e-20f);
    float p_star = pL + rhoL*(SL - vnL)*(SM - vnL);

    // Compute flux based on wave speed region
    if (SL >= 0.0f) {
        // F_L
        float fn = vnL;
        flux[0] = rhoL * fn;
        flux[1] = (rhoL*uL*fn + pL*nx);
        flux[2] = (rhoL*vL*fn + pL*ny);
        flux[3] = (rhoL*wL*fn + pL*nz);
        flux[4] = (rhoL*EL + pL) * fn;
    } else if (SL < 0.0f && SM >= 0.0f) {
        // F*_L
        float fn = SM;
        float vel_factor = rho_starL * SM;
        flux[0] = rho_starL * fn;
        flux[1] = vel_factor * uL + p_star * nx;
        flux[2] = vel_factor * vL + p_star * ny;
        flux[3] = vel_factor * wL + p_star * nz;
        flux[4] = (rho_starL * (EL + (SM - vnL)*(SM + pL/(rhoL*(SL - vnL) + 1e-20f))) + p_star) * fn;
    } else if (SM < 0.0f && SR >= 0.0f) {
        // F*_R
        float fn = SM;
        float vel_factor = rho_starR * SM;
        flux[0] = rho_starR * fn;
        flux[1] = vel_factor * uR + p_star * nx;
        flux[2] = vel_factor * vR + p_star * ny;
        flux[3] = vel_factor * wR + p_star * nz;
        flux[4] = (rho_starR * (ER + (SM - vnR)*(SM + pR/(rhoR*(SR - vnR) + 1e-20f))) + p_star) * fn;
    } else {
        // F_R
        float fn = vnR;
        flux[0] = rhoR * fn;
        flux[1] = (rhoR*uR*fn + pR*nx);
        flux[2] = (rhoR*vR*fn + pR*ny);
        flux[3] = (rhoR*wR*fn + pR*nz);
        flux[4] = (rhoR*ER + pR) * fn;
    }
}

// Minmod slope limiter
__device__ inline float minmod(float a, float b, float c) {
    float s = (a > 0) ? 1.0f : ((a < 0) ? -1.0f : 0.0f);
    if (s * b <= 0 || s * c <= 0) return 0.0f;
    float m = fminf(fabsf(a), fminf(fabsf(b), fabsf(c)));
    return s * m;
}

// ─── Device-side Q initialization ─────────────────────────────────────────

__global__ void init_Q_kernel(float* Q, int n, const float* Q_inf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < 5; ++k) Q[i*5 + k] = Q_inf[k];
}

// ─── Green-Gauss gradient computation ─────────────────────────────────────

__global__ void compute_gradients_kernel(
    const float* nodes,
    const int4* tets,
    const int4* tet_neighbors,
    const float* tet_volumes,
    const float* tet_centers,
    const int* boundary_type,
    int num_tets,
    const float* Q,
    float gamma,
    const float* Q_inf,
    float* grad)  // [num_tets][5][3]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tets) return;

    float prim[5];
    cons_to_prim(&Q[i*5], gamma, prim);
    int4 tet = tets[i];
    int4 nb = tet_neighbors[i];
    float vol = tet_volumes[i];
    float3 cc = ((float3*)tet_centers)[i];

    float g[5][3] = {{0}};

    for (int f = 0; f < 4; ++f) {
        int nb_idx, btype;
        switch (f) {
            case 0: nb_idx = nb.x; btype = boundary_type[i*4+0]; break;
            case 1: nb_idx = nb.y; btype = boundary_type[i*4+1]; break;
            case 2: nb_idx = nb.z; btype = boundary_type[i*4+2]; break;
            default: nb_idx = nb.w; btype = boundary_type[i*4+3]; break;
        }
        int a, b, c;
        switch (f) {
            case 0: a = tet.y; b = tet.z; c = tet.w; break;
            case 1: a = tet.x; b = tet.z; c = tet.w; break;
            case 2: a = tet.x; b = tet.y; c = tet.w; break;
            case 3: a = tet.x; b = tet.y; c = tet.z; break;
        }
        float3 v0 = ((float3*)nodes)[a];
        float3 v1 = ((float3*)nodes)[b];
        float3 v2 = ((float3*)nodes)[c];
        float3 e1 = v1 - v0, e2 = v2 - v0;
        float3 nf = cross(e1, e2);  // = 2 * area * n (unoriented)
        float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);

        // Ensure outward-pointing normal
        float3 dr = fc - cc;
        if (dr.x*nf.x + dr.y*nf.y + dr.z*nf.z < 0) {
            nf = make_float3(-nf.x, -nf.y, -nf.z);
        }

        float prim_face[5];
        if (btype == 0) {
            float prim_nb[5];
            cons_to_prim(&Q[nb_idx*5], gamma, prim_nb);
            for (int k = 0; k < 5; ++k) prim_face[k] = 0.5f * (prim[k] + prim_nb[k]);
        } else if (btype == 1) {
            for (int k = 0; k < 5; ++k) prim_face[k] = prim[k];
        } else {
            cons_to_prim(Q_inf, gamma, prim_face);
        }

        for (int k = 0; k < 5; ++k) {
            g[k][0] += prim_face[k] * nf.x;
            g[k][1] += prim_face[k] * nf.y;
            g[k][2] += prim_face[k] * nf.z;
        }
    }

    float inv_vol = 1.0f / fmaxf(vol, 1e-20f);
    for (int k = 0; k < 5; ++k) {
        grad[i*15 + k*3 + 0] = g[k][0] * inv_vol * 0.5f;
        grad[i*15 + k*3 + 1] = g[k][1] * inv_vol * 0.5f;
        grad[i*15 + k*3 + 2] = g[k][2] * inv_vol * 0.5f;
    }
}

// ─── FVM solver kernel (single condition, global memory Q) ────────────────

__global__ void fvm_solver_kernel(
    const float* nodes,
    const int4* tets,
    const int4* tet_neighbors,
    const float* tet_volumes,
    const float* tet_centers,
    const int* boundary_type,
    int num_tets,

    // Q arrays in global memory
    float* Q,
    float* Q0,

    // Gradient storage [num_tets][5][3]
    float* grad,

    // Freestream conditions
    float mach, float alpha_deg, float beta_deg,
    float gamma,
    float CFL,
    bool use_muscl,

    // Output: body forces
    float* out_CX, float* out_CY, float* out_CZ,
    float* out_Cl, float* out_Cm, float* out_Cn,

    // Solver stats
    int* out_iterations, float* out_residual)
{
    extern __shared__ float smem[];
    float* shared_dt = smem;

    float alpha = alpha_deg * 3.14159265f / 180.0f;
    float beta  = beta_deg  * 3.14159265f / 180.0f;
    float ca = cosf(alpha), sa = sinf(alpha);
    float cb = cosf(beta),  sb = sinf(beta);

    float u_dir = -ca * cb;
    float v_dir = -sb;
    float w_dir = -sa * cb;
    float u_inf = u_dir * mach;
    float v_inf = v_dir * mach;
    float w_inf = w_dir * mach;

    float Q_inf[5];
    Q_inf[0] = 1.0f;
    Q_inf[1] = u_inf;
    Q_inf[2] = v_inf;
    Q_inf[3] = w_inf;
    float p_inf = 1.0f / gamma;
    float ke_inf = 0.5f * (u_inf*u_inf + v_inf*v_inf + w_inf*w_inf);
    Q_inf[4] = p_inf / (gamma - 1.0f) + ke_inf;

    float prim_inf[5];
    cons_to_prim(Q_inf, gamma, prim_inf);

    int tid = threadIdx.x;
    int n = num_tets;
    int nthreads = blockDim.x;

    float res = 1.0f;
    int iter;
    float RK4_a[4] = {0.25f, 1.0f/3.0f, 0.5f, 1.0f};

    float min_rho = 1e10f, max_rho = 0.0f;
    float min_p = 1e10f, max_p = 0.0f;

    for (iter = 0; iter < 20000; ++iter) {
        // ── 1. Green-Gauss gradients ──
        for (int i = tid; i < n; i += nthreads) {
            float prim[5];
            cons_to_prim(&Q[i*5], gamma, prim);
            int4 tet = tets[i];
            int4 nb = tet_neighbors[i];
            float vol = tet_volumes[i];
            float3 cc = ((float3*)tet_centers)[i];
            float g[5][3] = {{0}};

            for (int f = 0; f < 4; ++f) {
                int nb_idx, btype;
                switch (f) {
                    case 0: nb_idx = nb.x; btype = boundary_type[i*4+0]; break;
                    case 1: nb_idx = nb.y; btype = boundary_type[i*4+1]; break;
                    case 2: nb_idx = nb.z; btype = boundary_type[i*4+2]; break;
                    default: nb_idx = nb.w; btype = boundary_type[i*4+3]; break;
                }
                int a, b, c;
                switch (f) {
                    case 0: a = tet.y; b = tet.z; c = tet.w; break;
                    case 1: a = tet.x; b = tet.z; c = tet.w; break;
                    case 2: a = tet.x; b = tet.y; c = tet.w; break;
                    case 3: a = tet.x; b = tet.y; c = tet.z; break;
                }
                float3 v0 = ((float3*)nodes)[a];
                float3 v1 = ((float3*)nodes)[b];
                float3 v2 = ((float3*)nodes)[c];
                float3 e1 = v1 - v0, e2 = v2 - v0;
                float3 nf = cross(e1, e2);
                float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);
                float3 dr = fc - cc;
                if (dr.x*nf.x + dr.y*nf.y + dr.z*nf.z < 0) {
                    nf = make_float3(-nf.x, -nf.y, -nf.z);
                }

                float prim_face[5];
                if (btype == 0) {
                    float prim_nb[5];
                    cons_to_prim(&Q[nb_idx*5], gamma, prim_nb);
                    for (int k = 0; k < 5; ++k) prim_face[k] = 0.5f * (prim[k] + prim_nb[k]);
                } else if (btype == 1) {
                    for (int k = 0; k < 5; ++k) prim_face[k] = prim[k];
                } else {
                    for (int k = 0; k < 5; ++k) prim_face[k] = prim_inf[k];
                }
                for (int k = 0; k < 5; ++k) {
                    g[k][0] += prim_face[k] * nf.x;
                    g[k][1] += prim_face[k] * nf.y;
                    g[k][2] += prim_face[k] * nf.z;
                }
            }

            float inv_vol = 1.0f / fmaxf(vol, 1e-20f);
            for (int k = 0; k < 5; ++k) {
                grad[i*15 + k*3 + 0] = g[k][0] * inv_vol * 0.5f;
                grad[i*15 + k*3 + 1] = g[k][1] * inv_vol * 0.5f;
                grad[i*15 + k*3 + 2] = g[k][2] * inv_vol * 0.5f;
            }
        }
        __syncthreads();

        // ── 2. Global time step ──
        __shared__ float dt_shared;
        float local_max_dt = 1e10f;
        for (int i = tid; i < n; i += nthreads) {
            float prim[5];
            cons_to_prim(&Q[i*5], gamma, prim);
            float a = speed_of_sound(gamma, prim[4], prim[0]);
            float vmag = sqrtf(prim[1]*prim[1] + prim[2]*prim[2] + prim[3]*prim[3]);
            float h = powf(tet_volumes[i], 1.0f/3.0f);
            float dt_cell = CFL * h / fmaxf(vmag + a, 1e-20f);
            if (dt_cell < local_max_dt) local_max_dt = dt_cell;
        }
        shared_dt[tid] = local_max_dt;
        __syncthreads();
        for (int s = nthreads/2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_dt[tid+s] < shared_dt[tid]) shared_dt[tid] = shared_dt[tid+s];
            }
            __syncthreads();
        }
        if (tid == 0) dt_shared = shared_dt[0];
        __syncthreads();
        float dt = dt_shared;

        // ── 3. Save Q0 ──
        for (int i = tid; i < n; i += nthreads) {
            for (int k = 0; k < 5; ++k) Q0[i*5 + k] = Q[i*5 + k];
        }
        __syncthreads();

        // ── 4. RK4 with MUSCL reconstruction ──
        for (int stage = 0; stage < 4; ++stage) {
            for (int i = tid; i < n; i += nthreads) {
                float R[5] = {0,0,0,0,0};
                int4 tet = tets[i];
                int4 nb = tet_neighbors[i];
                float vol = tet_volumes[i];
                float3 cc = ((float3*)tet_centers)[i];
                float prim_i[5];
                cons_to_prim(&Q[i*5], gamma, prim_i);

                for (int f = 0; f < 4; ++f) {
                    int nb_idx, btype;
                    switch (f) {
                        case 0: nb_idx = nb.x; btype = boundary_type[i*4+0]; break;
                        case 1: nb_idx = nb.y; btype = boundary_type[i*4+1]; break;
                        case 2: nb_idx = nb.z; btype = boundary_type[i*4+2]; break;
                        default: nb_idx = nb.w; btype = boundary_type[i*4+3]; break;
                    }
                    int a, b, c;
                    switch (f) {
                        case 0: a = tet.y; b = tet.z; c = tet.w; break;
                        case 1: a = tet.x; b = tet.z; c = tet.w; break;
                        case 2: a = tet.x; b = tet.y; c = tet.w; break;
                        case 3: a = tet.x; b = tet.y; c = tet.z; break;
                    }
                    float3 v0 = ((float3*)nodes)[a];
                    float3 v1 = ((float3*)nodes)[b];
                    float3 v2 = ((float3*)nodes)[c];
                    float3 e1 = v1 - v0, e2 = v2 - v0;
                    float3 nf_raw = cross(e1, e2);
                    float area = 0.5f * sqrtf(nf_raw.x*nf_raw.x + nf_raw.y*nf_raw.y + nf_raw.z*nf_raw.z);
                    float inv2a = 1.0f / (2.0f*area + 1e-20f);
                    float fnx = nf_raw.x * inv2a;
                    float fny = nf_raw.y * inv2a;
                    float fnz = nf_raw.z * inv2a;
                    float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);

                    // Ensure normal points outward
                    float3 df = fc - cc;
                    if (df.x*fnx + df.y*fny + df.z*fnz < 0) {
                        fnx = -fnx; fny = -fny; fnz = -fnz;
                    }

                    // MUSCL: prim at face center = prim_i + grad·(fc - cc)
                    float prim_L[5], prim_R[5];

                    if (!use_muscl) {
                        for (int k = 0; k < 5; ++k) {
                            prim_L[k] = prim_i[k];
                            prim_R[k] = (btype == 2) ? prim_inf[k] : prim_i[k];
                        }
                    } else {
                        for (int k = 0; k < 5; ++k) {
                            float* g = &grad[i*15 + k*3];
                            float rec = prim_i[k] + (g[0]*df.x + g[1]*df.y + g[2]*df.z);
                            if (k == 0 && rec < 1e-10f) rec = 1e-10f;
                            if (k == 4 && rec < 1e-10f) rec = 1e-10f;

                            if (btype == 0) {
                                float prim_nb[5];
                                cons_to_prim(&Q[nb_idx*5], gamma, prim_nb);
                                float3 cc_nb = ((float3*)tet_centers)[nb_idx];
                                float3 df_nb = fc - cc_nb;
                                float* gnb = &grad[nb_idx*15 + k*3];
                                float rec_nb = prim_nb[k] + (gnb[0]*df_nb.x + gnb[1]*df_nb.y + gnb[2]*df_nb.z);
                                if (k == 0 && rec_nb < 1e-10f) rec_nb = 1e-10f;
                                if (k == 4 && rec_nb < 1e-10f) rec_nb = 1e-10f;

                                float lo = fminf(prim_i[k], prim_nb[k]);
                                float hi = fmaxf(prim_i[k], prim_nb[k]);
                                prim_L[k] = fminf(fmaxf(rec, lo), hi);
                                prim_R[k] = fminf(fmaxf(rec_nb, lo), hi);
                            } else if (btype == 1) {
                                prim_L[k] = rec;
                                if (k == 0) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                if (k == 4) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                prim_R[k] = prim_L[k];
                            } else {
                                prim_L[k] = rec;
                                if (k == 0) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                if (k == 4) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                prim_R[k] = prim_inf[k];
                            }
                        }
                    }

                    float QL[5], QR[5];
                    QL[0] = prim_L[0];
                    QL[1] = prim_L[0] * prim_L[1];
                    QL[2] = prim_L[0] * prim_L[2];
                    QL[3] = prim_L[0] * prim_L[3];
                    QL[4] = prim_L[4]/(gamma-1.0f) + 0.5f*prim_L[0]*(prim_L[1]*prim_L[1]+prim_L[2]*prim_L[2]+prim_L[3]*prim_L[3]);

                    if (btype == 0) {
                        QR[0] = prim_R[0];
                        QR[1] = prim_R[0] * prim_R[1];
                        QR[2] = prim_R[0] * prim_R[2];
                        QR[3] = prim_R[0] * prim_R[3];
                        QR[4] = prim_R[4]/(gamma-1.0f) + 0.5f*prim_R[0]*(prim_R[1]*prim_R[1]+prim_R[2]*prim_R[2]+prim_R[3]*prim_R[3]);
                    } else if (btype == 1) {
                        float vn = prim_L[1]*fnx + prim_L[2]*fny + prim_L[3]*fnz;
                        float uR = prim_L[1] - 2.0f*vn*fnx;
                        float vR = prim_L[2] - 2.0f*vn*fny;
                        float wR = prim_L[3] - 2.0f*vn*fnz;
                        QR[0] = prim_L[0];
                        QR[1] = prim_L[0]*uR;
                        QR[2] = prim_L[0]*vR;
                        QR[3] = prim_L[0]*wR;
                        QR[4] = prim_L[4]/(gamma-1.0f) + 0.5f*prim_L[0]*(uR*uR + vR*vR + wR*wR);
                    } else {
                        for (int k = 0; k < 5; ++k) QR[k] = Q_inf[k];
                    }

                    float flux[5];
                    hllc_flux(QL, QR, fnx, fny, fnz, gamma, flux);
                    for (int k = 0; k < 5; ++k) R[k] -= flux[k] * area / vol;
                }

                for (int k = 0; k < 5; ++k) {
                    Q[i*5 + k] = Q0[i*5 + k] + RK4_a[stage] * dt * R[k];
                }
                {
                    float rho = Q[i*5 + 0];
                    if (rho < 1e-10f) {
                        for (int k = 0; k < 5; ++k) Q[i*5 + k] = Q0[i*5 + k];
                    } else {
                        float inv_rho = 1.0f / rho;
                        float u = Q[i*5 + 1] * inv_rho;
                        float v = Q[i*5 + 2] * inv_rho;
                        float w = Q[i*5 + 3] * inv_rho;
                        float E = Q[i*5 + 4] * inv_rho;
                        float ke = 0.5f * (u*u + v*v + w*w);
                        float p = (gamma - 1.0f) * rho * (E - ke);
                        if (p < 1e-10f) {
                            Q[i*5 + 0] = rho;
                            Q[i*5 + 4] = 1e-10f / (gamma - 1.0f) + 0.5f * rho * (u*u + v*v + w*w);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // ── 5. Residual ──
        float local_sum = 0.0f;
        for (int i = tid; i < n; i += nthreads) {
            for (int k = 0; k < 5; ++k) {
                float dq = Q[i*5 + k] - Q0[i*5 + k];
                local_sum += dq * dq;
            }
        }
        shared_dt[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads/2; s > 0; s >>= 1) {
            if (tid < s) shared_dt[tid] += shared_dt[tid+s];
            __syncthreads();
        }
        if (tid == 0) {
            res = sqrtf(shared_dt[0] / (5.0f * n));
            if (iter % 500 == 0) {
                printf("[fvm iter=%d] res=%.6e\n", iter, res);
            }
        }
        __syncthreads();

        if (res < 1e-8f || isnan(res)) { if (tid == 0 && isnan(res)) printf("[fvm] NaN residual at iter %d\n", iter); break; }
    }

    // Body forces on wall faces (F = -∮ p·n·dA)
    if (tid == 0) {
        float FX = 0, FY = 0, FZ = 0;
        float MX = 0, MY = 0, MZ = 0;
        for (int i = 0; i < n; ++i) {
            int4 nb = tet_neighbors[i];
            float3 cc = ((float3*)tet_centers)[i];
            for (int f = 0; f < 4; ++f) {
                int btype;
                switch (f) {
                    case 0: btype = boundary_type[i*4+0]; break;
                    case 1: btype = boundary_type[i*4+1]; break;
                    case 2: btype = boundary_type[i*4+2]; break;
                    default: btype = boundary_type[i*4+3]; break;
                }
                if (btype != 1) continue;
                int4 tet = tets[i];
                int a, b, c;
                switch (f) {
                    case 0: a = tet.y; b = tet.z; c = tet.w; break;
                    case 1: a = tet.x; b = tet.z; c = tet.w; break;
                    case 2: a = tet.x; b = tet.y; c = tet.w; break;
                    case 3: a = tet.x; b = tet.y; c = tet.z; break;
                }
                float3 v0 = ((float3*)nodes)[a];
                float3 v1 = ((float3*)nodes)[b];
                float3 v2 = ((float3*)nodes)[c];
                float3 e1 = v1 - v0, e2 = v2 - v0;
                float3 nf = cross(e1, e2);
                float area = 0.5f * sqrtf(nf.x*nf.x + nf.y*nf.y + nf.z*nf.z);
                float nx = nf.x / (2.0f*area + 1e-20f);
                float ny = nf.y / (2.0f*area + 1e-20f);
                float nz = nf.z / (2.0f*area + 1e-20f);
                float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);
                float3 dr = fc - cc;
                if (dr.x*nx + dr.y*ny + dr.z*nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
                float prim[5];
                cons_to_prim(&Q[i*5], gamma, prim);
                float fmag = prim[4] * area;
                FX -= fmag * nx;
                FY -= fmag * ny;
                FZ -= fmag * nz;
                MX -= fmag * (ny*fc.z - nz*fc.y);
                MY -= fmag * (nz*fc.x - nx*fc.z);
                MZ -= fmag * (nx*fc.y - ny*fc.x);
            }
        }
        out_CX[0] = FX;
        out_CY[0] = FY;
        out_CZ[0] = FZ;
        out_Cl[0] = MX;
        out_Cm[0] = MY;
        out_Cn[0] = MZ;
        out_iterations[0] = iter;
        out_residual[0] = res;
    }
}

// ─── Batch FVM solver kernel ──────────────────────────────────────────────

__global__ void fvm_batch_kernel(
    const float* nodes,
    const int4* tets,
    const int4* tet_neighbors,
    const float* tet_volumes,
    const float* tet_centers,
    const int* boundary_type,
    int num_tets,

    float* Q_all,
    float* Q0_all,
    float* grad_all,

    const float* machs, const float* alphas, const float* betas,
    int num_conditions,
    float gamma, float CFL, bool use_muscl,

    float* forces_out,
    int* iters_out, float* res_out)
{
    int cond = blockIdx.x;
    if (cond >= num_conditions) return;

    float* Q  = Q_all  + cond * num_tets * 5;
    float* Q0 = Q0_all + cond * num_tets * 5;
    float* grad = grad_all + cond * num_tets * 15;

    extern __shared__ float smem[];
    float* shared_dt = smem;

    float mach = machs[cond];
    float alpha_deg = alphas[cond];
    float beta_deg = betas[cond];
    float alpha = alpha_deg * 3.14159265f / 180.0f;
    float beta  = beta_deg  * 3.14159265f / 180.0f;
    float ca = cosf(alpha), sa = sinf(alpha);
    float cb = cosf(beta),  sb = sinf(beta);

    float u_dir = -ca * cb;
    float v_dir = -sb;
    float w_dir = -sa * cb;
    float u_inf = u_dir * mach;
    float v_inf = v_dir * mach;
    float w_inf = w_dir * mach;

    float Q_inf[5];
    Q_inf[0] = 1.0f;
    Q_inf[1] = u_inf;
    Q_inf[2] = v_inf;
    Q_inf[3] = w_inf;
    float p_inf = 1.0f / gamma;
    float ke_inf = 0.5f * (u_inf*u_inf + v_inf*v_inf + w_inf*w_inf);
    Q_inf[4] = p_inf / (gamma - 1.0f) + ke_inf;

    float prim_inf[5];
    cons_to_prim(Q_inf, gamma, prim_inf);

    int tid = threadIdx.x;
    int n = num_tets;
    int nthreads = blockDim.x;

    // Init Q = Q_inf
    for (int i = tid; i < n; i += nthreads) {
        for (int k = 0; k < 5; ++k) Q[i*5 + k] = Q_inf[k];
        for (int k = 0; k < 5; ++k) Q0[i*5 + k] = Q_inf[k];
    }
    __syncthreads();

    float res = 1.0f;
    int iter;
    float RK4_a[4] = {0.25f, 1.0f/3.0f, 0.5f, 1.0f};

    for (iter = 0; iter < 20000; ++iter) {
        // ── 1. Green-Gauss gradients ──
        for (int i = tid; i < n; i += nthreads) {
            float prim[5];
            cons_to_prim(&Q[i*5], gamma, prim);
            int4 tet = tets[i];
            int4 nb = tet_neighbors[i];
            float vol = tet_volumes[i];
            float3 cc = ((float3*)tet_centers)[i];
            float g[5][3] = {{0}};

            for (int f = 0; f < 4; ++f) {
                int nb_idx, btype;
                switch (f) {
                    case 0: nb_idx = nb.x; btype = boundary_type[i*4+0]; break;
                    case 1: nb_idx = nb.y; btype = boundary_type[i*4+1]; break;
                    case 2: nb_idx = nb.z; btype = boundary_type[i*4+2]; break;
                    default: nb_idx = nb.w; btype = boundary_type[i*4+3]; break;
                }
                int a, b, c;
                switch (f) {
                    case 0: a = tet.y; b = tet.z; c = tet.w; break;
                    case 1: a = tet.x; b = tet.z; c = tet.w; break;
                    case 2: a = tet.x; b = tet.y; c = tet.w; break;
                    case 3: a = tet.x; b = tet.y; c = tet.z; break;
                }
                float3 v0 = ((float3*)nodes)[a];
                float3 v1 = ((float3*)nodes)[b];
                float3 v2 = ((float3*)nodes)[c];
                float3 e1 = v1 - v0, e2 = v2 - v0;
                float3 nf = cross(e1, e2);
                float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);
                float3 dr = fc - cc;
                if (dr.x*nf.x + dr.y*nf.y + dr.z*nf.z < 0) {
                    nf = make_float3(-nf.x, -nf.y, -nf.z);
                }
                float prim_face[5];
                if (btype == 0) {
                    float prim_nb[5];
                    cons_to_prim(&Q[nb_idx*5], gamma, prim_nb);
                    for (int k = 0; k < 5; ++k) prim_face[k] = 0.5f * (prim[k] + prim_nb[k]);
                } else if (btype == 1) {
                    for (int k = 0; k < 5; ++k) prim_face[k] = prim[k];
                } else {
                    for (int k = 0; k < 5; ++k) prim_face[k] = prim_inf[k];
                }
                for (int k = 0; k < 5; ++k) {
                    g[k][0] += prim_face[k] * nf.x;
                    g[k][1] += prim_face[k] * nf.y;
                    g[k][2] += prim_face[k] * nf.z;
                }
            }
            float inv_vol = 1.0f / fmaxf(vol, 1e-20f);
            for (int k = 0; k < 5; ++k) {
                grad[i*15 + k*3 + 0] = g[k][0] * inv_vol * 0.5f;
                grad[i*15 + k*3 + 1] = g[k][1] * inv_vol * 0.5f;
                grad[i*15 + k*3 + 2] = g[k][2] * inv_vol * 0.5f;
            }
        }
        __syncthreads();

        // ── 2. Global time step ──
        __shared__ float dt_shared;
        float local_max_dt = 1e10f;
        for (int i = tid; i < n; i += nthreads) {
            float prim[5];
            cons_to_prim(&Q[i*5], gamma, prim);
            float a = speed_of_sound(gamma, prim[4], prim[0]);
            float vmag = sqrtf(prim[1]*prim[1] + prim[2]*prim[2] + prim[3]*prim[3]);
            float h = powf(tet_volumes[i], 1.0f/3.0f);
            float dt_cell = CFL * h / fmaxf(vmag + a, 1e-20f);
            if (dt_cell < local_max_dt) local_max_dt = dt_cell;
        }
        shared_dt[tid] = local_max_dt;
        __syncthreads();
        for (int s = nthreads/2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_dt[tid+s] < shared_dt[tid]) shared_dt[tid] = shared_dt[tid+s];
            }
            __syncthreads();
        }
        if (tid == 0) dt_shared = shared_dt[0];
        __syncthreads();
        float dt = dt_shared;

        // ── 3. Save Q0 ──
        for (int i = tid; i < n; i += nthreads) {
            for (int k = 0; k < 5; ++k) Q0[i*5 + k] = Q[i*5 + k];
        }
        __syncthreads();

        // ── 4. RK4 with MUSCL ──
        for (int stage = 0; stage < 4; ++stage) {
            for (int i = tid; i < n; i += nthreads) {
                float R[5] = {0,0,0,0,0};
                int4 tet = tets[i];
                int4 nb = tet_neighbors[i];
                float vol = tet_volumes[i];
                float3 cc = ((float3*)tet_centers)[i];
                float prim_i[5];
                cons_to_prim(&Q[i*5], gamma, prim_i);

                for (int f = 0; f < 4; ++f) {
                    int nb_idx, btype;
                    switch (f) {
                        case 0: nb_idx = nb.x; btype = boundary_type[i*4+0]; break;
                        case 1: nb_idx = nb.y; btype = boundary_type[i*4+1]; break;
                        case 2: nb_idx = nb.z; btype = boundary_type[i*4+2]; break;
                        default: nb_idx = nb.w; btype = boundary_type[i*4+3]; break;
                    }
                    int a, b, c;
                    switch (f) {
                        case 0: a = tet.y; b = tet.z; c = tet.w; break;
                        case 1: a = tet.x; b = tet.z; c = tet.w; break;
                        case 2: a = tet.x; b = tet.y; c = tet.w; break;
                        case 3: a = tet.x; b = tet.y; c = tet.z; break;
                    }
                    float3 v0 = ((float3*)nodes)[a];
                    float3 v1 = ((float3*)nodes)[b];
                    float3 v2 = ((float3*)nodes)[c];
                    float3 e1 = v1 - v0, e2 = v2 - v0;
                    float3 nf_raw = cross(e1, e2);
                    float area = 0.5f * sqrtf(nf_raw.x*nf_raw.x + nf_raw.y*nf_raw.y + nf_raw.z*nf_raw.z);
                    float inv2a = 1.0f / (2.0f*area + 1e-20f);
                    float fnx = nf_raw.x * inv2a;
                    float fny = nf_raw.y * inv2a;
                    float fnz = nf_raw.z * inv2a;
                    float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);

                    // Ensure normal points outward
                    float3 df = fc - cc;
                    if (df.x*fnx + df.y*fny + df.z*fnz < 0) {
                        fnx = -fnx; fny = -fny; fnz = -fnz;
                    }

                    // MUSCL: prim at face center = prim_i + grad·(fc - cc)
                    float prim_L[5], prim_R[5];

                    if (!use_muscl) {
                        for (int k = 0; k < 5; ++k) {
                            prim_L[k] = prim_i[k];
                            prim_R[k] = (btype == 2) ? prim_inf[k] : prim_i[k];
                        }
                    } else {
                        for (int k = 0; k < 5; ++k) {
                            float* g = &grad[i*15 + k*3];
                            float rec = prim_i[k] + (g[0]*df.x + g[1]*df.y + g[2]*df.z);
                            if (k == 0 && rec < 1e-10f) rec = 1e-10f;
                            if (k == 4 && rec < 1e-10f) rec = 1e-10f;

                            if (btype == 0) {
                                float prim_nb[5];
                                cons_to_prim(&Q[nb_idx*5], gamma, prim_nb);
                                float3 cc_nb = ((float3*)tet_centers)[nb_idx];
                                float3 df_nb = fc - cc_nb;
                                float* gnb = &grad[nb_idx*15 + k*3];
                                float rec_nb = prim_nb[k] + (gnb[0]*df_nb.x + gnb[1]*df_nb.y + gnb[2]*df_nb.z);
                                if (k == 0 && rec_nb < 1e-10f) rec_nb = 1e-10f;
                                if (k == 4 && rec_nb < 1e-10f) rec_nb = 1e-10f;

                                float lo = fminf(prim_i[k], prim_nb[k]);
                                float hi = fmaxf(prim_i[k], prim_nb[k]);
                                prim_L[k] = fminf(fmaxf(rec, lo), hi);
                                prim_R[k] = fminf(fmaxf(rec_nb, lo), hi);
                            } else if (btype == 1) {
                                prim_L[k] = rec;
                                if (k == 0) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                if (k == 4) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                prim_R[k] = prim_L[k];
                            } else {
                                prim_L[k] = rec;
                                if (k == 0) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                if (k == 4) prim_L[k] = fmaxf(prim_L[k], 1e-10f);
                                prim_R[k] = prim_inf[k];
                            }
                        }
                    }

                    float QL[5], QR[5];
                    QL[0] = prim_L[0];
                    QL[1] = prim_L[0] * prim_L[1];
                    QL[2] = prim_L[0] * prim_L[2];
                    QL[3] = prim_L[0] * prim_L[3];
                    QL[4] = prim_L[4]/(gamma-1.0f) + 0.5f*prim_L[0]*(prim_L[1]*prim_L[1]+prim_L[2]*prim_L[2]+prim_L[3]*prim_L[3]);

                    if (btype == 0) {
                        QR[0] = prim_R[0];
                        QR[1] = prim_R[0] * prim_R[1];
                        QR[2] = prim_R[0] * prim_R[2];
                        QR[3] = prim_R[0] * prim_R[3];
                        QR[4] = prim_R[4]/(gamma-1.0f) + 0.5f*prim_R[0]*(prim_R[1]*prim_R[1]+prim_R[2]*prim_R[2]+prim_R[3]*prim_R[3]);
                    } else if (btype == 1) {
                        float vn = prim_L[1]*fnx + prim_L[2]*fny + prim_L[3]*fnz;
                        float uR = prim_L[1] - 2.0f*vn*fnx;
                        float vR = prim_L[2] - 2.0f*vn*fny;
                        float wR = prim_L[3] - 2.0f*vn*fnz;
                        QR[0] = prim_L[0];
                        QR[1] = prim_L[0]*uR;
                        QR[2] = prim_L[0]*vR;
                        QR[3] = prim_L[0]*wR;
                        QR[4] = prim_L[4]/(gamma-1.0f) + 0.5f*prim_L[0]*(uR*uR + vR*vR + wR*wR);
                    } else {
                        for (int k = 0; k < 5; ++k) QR[k] = Q_inf[k];
                    }

                    float flux[5];
                    hllc_flux(QL, QR, fnx, fny, fnz, gamma, flux);
                    for (int k = 0; k < 5; ++k) R[k] -= flux[k] * area / vol;
                }

                for (int k = 0; k < 5; ++k) {
                    Q[i*5 + k] = Q0[i*5 + k] + RK4_a[stage] * dt * R[k];
                }
                {
                    float rho = Q[i*5 + 0];
                    if (rho < 1e-10f) {
                        for (int k = 0; k < 5; ++k) Q[i*5 + k] = Q0[i*5 + k];
                    } else {
                        float inv_rho = 1.0f / rho;
                        float u = Q[i*5 + 1] * inv_rho;
                        float v = Q[i*5 + 2] * inv_rho;
                        float w = Q[i*5 + 3] * inv_rho;
                        float E = Q[i*5 + 4] * inv_rho;
                        float ke = 0.5f * (u*u + v*v + w*w);
                        float p = (gamma - 1.0f) * rho * (E - ke);
                        if (p < 1e-10f) {
                            Q[i*5 + 0] = rho;
                            Q[i*5 + 4] = 1e-10f / (gamma - 1.0f) + 0.5f * rho * (u*u + v*v + w*w);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // ── 5. Residual ──
        float local_sum = 0.0f;
        for (int i = tid; i < n; i += nthreads) {
            for (int k = 0; k < 5; ++k) {
                float dq = Q[i*5 + k] - Q0[i*5 + k];
                local_sum += dq * dq;
            }
        }
        shared_dt[tid] = local_sum;
        __syncthreads();
        for (int s = nthreads/2; s > 0; s >>= 1) {
            if (tid < s) shared_dt[tid] += shared_dt[tid+s];
            __syncthreads();
        }
        if (tid == 0) {
            res = sqrtf(shared_dt[0] / (5.0f * n));
        }
        __syncthreads();
        if (res < 1e-8f || isnan(res)) break;
    }

    // Body forces on wall faces (F = -∮ p·n·dA)
    if (tid == 0) {
        float FX = 0, FY = 0, FZ = 0;
        float MX = 0, MY = 0, MZ = 0;
        for (int i = 0; i < n; ++i) {
            int4 nb = tet_neighbors[i];
            float3 cc = ((float3*)tet_centers)[i];
            for (int f = 0; f < 4; ++f) {
                int btype;
                switch (f) {
                    case 0: btype = boundary_type[i*4+0]; break;
                    case 1: btype = boundary_type[i*4+1]; break;
                    case 2: btype = boundary_type[i*4+2]; break;
                    default: btype = boundary_type[i*4+3]; break;
                }
                if (btype != 1) continue;
                int4 tet = tets[i];
                int a, b, c;
                switch (f) {
                    case 0: a = tet.y; b = tet.z; c = tet.w; break;
                    case 1: a = tet.x; b = tet.z; c = tet.w; break;
                    case 2: a = tet.x; b = tet.y; c = tet.w; break;
                    case 3: a = tet.x; b = tet.y; c = tet.z; break;
                }
                float3 v0 = ((float3*)nodes)[a];
                float3 v1 = ((float3*)nodes)[b];
                float3 v2 = ((float3*)nodes)[c];
                float3 e1 = v1 - v0, e2 = v2 - v0;
                float3 nf = cross(e1, e2);
                float area = 0.5f * sqrtf(nf.x*nf.x + nf.y*nf.y + nf.z*nf.z);
                float nx = nf.x / (2.0f*area + 1e-20f);
                float ny = nf.y / (2.0f*area + 1e-20f);
                float nz = nf.z / (2.0f*area + 1e-20f);
                float3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);
                float3 dr = fc - cc;
                if (dr.x*nx + dr.y*ny + dr.z*nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
                float prim[5];
                cons_to_prim(&Q[i*5], gamma, prim);
                float fmag = prim[4] * area;
                FX -= fmag * nx;
                FY -= fmag * ny;
                FZ -= fmag * nz;
                MX -= fmag * (ny*fc.z - nz*fc.y);
                MY -= fmag * (nz*fc.x - nx*fc.z);
                MZ -= fmag * (nx*fc.y - ny*fc.x);
            }
        }
        forces_out[cond*6 + 0] = FX;
        forces_out[cond*6 + 1] = FY;
        forces_out[cond*6 + 2] = FZ;
        forces_out[cond*6 + 3] = MX;
        forces_out[cond*6 + 4] = MY;
        forces_out[cond*6 + 5] = MZ;
        iters_out[cond] = iter;
        res_out[cond] = res;
    }
}

// ─── CfdSolver implementation ──────────────────────────────────────────────

CfdSolver::CfdSolver() {}
CfdSolver::~CfdSolver() {
    if (d_nodes) { cudaFree(d_nodes); d_nodes = nullptr; }
    if (d_tets) { cudaFree(d_tets); d_tets = nullptr; }
    if (d_tet_neighbors) { cudaFree(d_tet_neighbors); d_tet_neighbors = nullptr; }
    if (d_tet_volumes) { cudaFree(d_tet_volumes); d_tet_volumes = nullptr; }
    if (d_tet_centers) { cudaFree(d_tet_centers); d_tet_centers = nullptr; }
    if (d_boundary_type) { cudaFree(d_boundary_type); d_boundary_type = nullptr; }
    free_scratch();
}

bool CfdSolver::alloc_scratch() {
    if (num_tets == 0) return false;
    free_scratch();
    CUDA_CHECK(cudaMalloc(&d_persist_Q, num_tets * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_Q0, num_tets * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_grad, num_tets * 15 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_CX, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_CY, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_CZ, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_Cl, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_Cm, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_Cn, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_persist_iter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_persist_res, sizeof(float)));
    return true;
}

void CfdSolver::free_scratch() {
    if (d_persist_Q) { cudaFree(d_persist_Q); d_persist_Q = nullptr; }
    if (d_persist_Q0) { cudaFree(d_persist_Q0); d_persist_Q0 = nullptr; }
    if (d_persist_grad) { cudaFree(d_persist_grad); d_persist_grad = nullptr; }
    if (d_persist_CX) { cudaFree(d_persist_CX); d_persist_CX = nullptr; }
    if (d_persist_CY) { cudaFree(d_persist_CY); d_persist_CY = nullptr; }
    if (d_persist_CZ) { cudaFree(d_persist_CZ); d_persist_CZ = nullptr; }
    if (d_persist_Cl) { cudaFree(d_persist_Cl); d_persist_Cl = nullptr; }
    if (d_persist_Cm) { cudaFree(d_persist_Cm); d_persist_Cm = nullptr; }
    if (d_persist_Cn) { cudaFree(d_persist_Cn); d_persist_Cn = nullptr; }
    if (d_persist_iter) { cudaFree(d_persist_iter); d_persist_iter = nullptr; }
    if (d_persist_res) { cudaFree(d_persist_res); d_persist_res = nullptr; }
}

bool CfdSolver::load_mesh(const TetMesh& mesh) {
    if (mesh.nodes.empty() || mesh.tets.empty()) return false;

    // Copy nodes
    num_nodes = static_cast<int>(mesh.nodes.size());
    num_tets = static_cast<int>(mesh.tets.size());

    CUDA_CHECK(cudaMalloc(&d_nodes, num_nodes * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_nodes, mesh.nodes.data(), num_nodes * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tets, num_tets * sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_tets, mesh.tets.data(), num_tets * sizeof(int4), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tet_neighbors, num_tets * sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_tet_neighbors, mesh.tet_neighbors.data(), num_tets * sizeof(int4), cudaMemcpyHostToDevice));

    // Unpack boundary_type: for each tet, 4 faces
    std::vector<int> h_btype(num_tets * 4, 0);
    for (int i = 0; i < num_tets; ++i) {
        int4 nb = mesh.tet_neighbors[i];
        h_btype[i*4 + 0] = (nb.x >= 0) ? 0 : ((nb.x == -1) ? 1 : 2);
        h_btype[i*4 + 1] = (nb.y >= 0) ? 0 : ((nb.y == -1) ? 1 : 2);
        h_btype[i*4 + 2] = (nb.z >= 0) ? 0 : ((nb.z == -1) ? 1 : 2);
        h_btype[i*4 + 3] = (nb.w >= 0) ? 0 : ((nb.w == -1) ? 1 : 2);
    }
    CUDA_CHECK(cudaMalloc(&d_boundary_type, num_tets * 4 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_boundary_type, h_btype.data(), num_tets * 4 * sizeof(int), cudaMemcpyHostToDevice));

    // Compute tet volumes and centers
    std::vector<float> h_vol(num_tets, 0);
    std::vector<float3> h_centers(num_tets);
    for (int i = 0; i < num_tets; ++i) {
        int4 t = mesh.tets[i];
        float3 v0 = mesh.nodes[t.x];
        float3 v1 = mesh.nodes[t.y];
        float3 v2 = mesh.nodes[t.z];
        float3 v3 = mesh.nodes[t.w];
        float3 e1 = v1 - v0, e2 = v2 - v0, e3 = v3 - v0;
        float det = e1.x * (e2.y * e3.z - e2.z * e3.y)
                  - e1.y * (e2.x * e3.z - e2.z * e3.x)
                  + e1.z * (e2.x * e3.y - e2.y * e3.x);
        h_vol[i] = fabsf(det) / 6.0f;
        h_centers[i] = (v0 + v1 + v2 + v3) * 0.25f;
    }

    CUDA_CHECK(cudaMalloc(&d_tet_volumes, num_tets * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tet_volumes, h_vol.data(), num_tets * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tet_centers, num_tets * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_tet_centers, h_centers.data(), num_tets * sizeof(float3), cudaMemcpyHostToDevice));

    return true;
}

CfdResult CfdSolver::solve(
    float mach, float alpha_deg, float beta_deg,
    float ref_area, float ref_length, float ref_span,
    float com_x, float com_y, float com_z,
    const CfdConfig& cfg,
    float* d_Q_init,
    float** d_Q_out)
{
    (void)com_x; (void)com_y; (void)com_z;
    CfdResult result = {};

    bool use_persist = (d_persist_Q != nullptr);
    float *d_Q, *d_Q0, *d_grad;
    float *d_CX, *d_CY, *d_CZ, *d_Cl, *d_Cm, *d_Cn;
    int* d_iter;
    float* d_res;

    if (use_persist) {
        d_Q = d_persist_Q; d_Q0 = d_persist_Q0; d_grad = d_persist_grad;
        d_CX = d_persist_CX; d_CY = d_persist_CY; d_CZ = d_persist_CZ;
        d_Cl = d_persist_Cl; d_Cm = d_persist_Cm; d_Cn = d_persist_Cn;
        d_iter = d_persist_iter; d_res = d_persist_res;
    } else {
        CUDA_CHECK(cudaMalloc(&d_Q,  num_tets * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q0, num_tets * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, num_tets * 15 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_CX, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_CY, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_CZ, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Cl, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Cm, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Cn, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_iter, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));
    }

    // Freestream state
    float alpha_rad = alpha_deg * 3.14159265f / 180.0f;
    float beta_rad  = beta_deg  * 3.14159265f / 180.0f;
    float u_dir = -cosf(alpha_rad)*cosf(beta_rad);
    float v_dir = -sinf(beta_rad);
    float w_dir = -sinf(alpha_rad)*cosf(beta_rad);
    float u_inf = u_dir * mach;
    float v_inf = v_dir * mach;
    float w_inf = w_dir * mach;
    float Q_inf_h[5] = {1.0f, u_inf, v_inf, w_inf,
        1.0f/(cfg.gamma*(cfg.gamma-1.0f)) + 0.5f*(u_inf*u_inf+v_inf*v_inf+w_inf*w_inf)};

    // Initialize Q: warm-start or freestream
    if (d_Q_init != nullptr) {
        if (d_Q_init != d_Q) {
            CUDA_CHECK(cudaMemcpy(d_Q, d_Q_init, num_tets * 5 * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    } else {
        float* d_Q_inf;
        CUDA_CHECK(cudaMalloc(&d_Q_inf, 5 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q_inf, Q_inf_h, 5 * sizeof(float), cudaMemcpyHostToDevice));
        int init_threads = 256;
        int init_blocks = (num_tets + init_threads - 1) / init_threads;
        init_Q_kernel<<<init_blocks, init_threads>>>(d_Q, num_tets, d_Q_inf);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaFree(d_Q_inf));
    }

    int threads = std::min(num_tets, 256);
    int smem_size = threads * sizeof(float);  // shared_dt
    fvm_solver_kernel<<<1, threads, smem_size>>>(
        d_nodes, d_tets, d_tet_neighbors, d_tet_volumes,
        d_tet_centers, d_boundary_type, num_tets,
        d_Q, d_Q0, d_grad,
        mach, alpha_deg, beta_deg, cfg.gamma, cfg.cfl, cfg.muscl,
        d_CX, d_CY, d_CZ, d_Cl, d_Cm, d_Cn,
        d_iter, d_res);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    float CX, CY, CZ, Cl, Cm, Cn;
    int iter;
    float res;
    CUDA_CHECK(cudaMemcpy(&CX, d_CX, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&CY, d_CY, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&CZ, d_CZ, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&Cl, d_Cl, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&Cm, d_Cm, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&Cn, d_Cn, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost));

    if (!use_persist) {
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_Q0));
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_CX));
        CUDA_CHECK(cudaFree(d_CY));
        CUDA_CHECK(cudaFree(d_CZ));
        CUDA_CHECK(cudaFree(d_Cl));
        CUDA_CHECK(cudaFree(d_Cm));
        CUDA_CHECK(cudaFree(d_Cn));
        CUDA_CHECK(cudaFree(d_iter));
        CUDA_CHECK(cudaFree(d_res));
    }

    // Normalize forces by ref_area and q_∞
    // In our dimensionless formulation, q_∞ = γ * p_∞ * M² / 2 = γ * (1/γ) * M²/2 = M²/2
    float q_inf = 0.5f * mach * mach;

    result.CX = CX / (q_inf * ref_area);
    result.CY = CY / (q_inf * ref_area);
    result.CZ = CZ / (q_inf * ref_area);
    result.Cl = Cl / (q_inf * ref_area * ref_span);
    result.Cm = Cm / (q_inf * ref_area * ref_length);
    result.Cn = Cn / (q_inf * ref_area * ref_span);
    result.iterations = iter;
    result.residual = res;

    // Convert to wind frame (CL, CD)
    float ca = cosf(alpha_rad), sa = sinf(alpha_rad);
    float cb = cosf(beta_rad),  sb = sinf(beta_rad);

    float Fsx = result.CX * ca * cb + result.CY * sb + result.CZ * sa * cb;
    float Fsz = -result.CX * sa + result.CZ * ca;
    result.CD = -Fsx;
    result.CL = -Fsz;

    if (d_Q_out) *d_Q_out = d_Q;

    return result;
}

std::vector<CfdResult> CfdSolver::solve_batch(
    const std::vector<float>& machs,
    const std::vector<float>& alphas,
    float beta_deg,
    float ref_area, float ref_length, float ref_span,
    float com_x, float com_y, float com_z,
    const CfdConfig& cfg)
{
    (void)com_x; (void)com_y; (void)com_z;
    int num_cond = (int)(machs.size() * alphas.size());

    // Build flat condition array: all machs × all alphas
    std::vector<float> h_machs(num_cond), h_alphas(num_cond), h_betas(num_cond, beta_deg);
    int idx = 0;
    for (float m : machs)
        for (float a : alphas) {
            h_machs[idx] = m;
            h_alphas[idx] = a;
            idx++;
        }

    float *d_machs, *d_alphas, *d_betas, *d_forces;
    int *d_iters; float* d_res;
    float *d_Q_all, *d_Q0_all, *d_grad_all;

    CUDA_CHECK(cudaMalloc(&d_machs, num_cond * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alphas, num_cond * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_betas, num_cond * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_forces, num_cond * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_iters, num_cond * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_res, num_cond * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_all, num_cond * num_tets * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q0_all, num_cond * num_tets * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_all, num_cond * num_tets * 15 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_machs, h_machs.data(), num_cond * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas.data(), num_cond * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_betas, h_betas.data(), num_cond * sizeof(float), cudaMemcpyHostToDevice));

    int threads = std::min(num_tets, 256);
    int smem_size = threads * sizeof(float);
    fvm_batch_kernel<<<num_cond, threads, smem_size>>>(
        d_nodes, d_tets, d_tet_neighbors, d_tet_volumes,
        d_tet_centers, d_boundary_type, num_tets,
        d_Q_all, d_Q0_all, d_grad_all,
        d_machs, d_alphas, d_betas, num_cond,
        cfg.gamma, cfg.cfl, cfg.muscl,
        d_forces, d_iters, d_res);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_forces(num_cond * 6);
    std::vector<int> h_iters(num_cond);
    std::vector<float> h_res(num_cond);
    CUDA_CHECK(cudaMemcpy(h_forces.data(), d_forces, num_cond * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_iters.data(), d_iters, num_cond * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, num_cond * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_machs));
    CUDA_CHECK(cudaFree(d_alphas));
    CUDA_CHECK(cudaFree(d_betas));
    CUDA_CHECK(cudaFree(d_forces));
    CUDA_CHECK(cudaFree(d_iters));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_Q_all));
    CUDA_CHECK(cudaFree(d_Q0_all));
    CUDA_CHECK(cudaFree(d_grad_all));

    // Post-process per condition
    std::vector<CfdResult> results(num_cond);
    for (int i = 0; i < num_cond; ++i) {
        float mach = h_machs[i];
        float alpha_deg = h_alphas[i];
        float q_inf = 0.5f * mach * mach;
        float FX = h_forces[i*6+0], FY = h_forces[i*6+1], FZ = h_forces[i*6+2];
        float MX = h_forces[i*6+3], MY = h_forces[i*6+4], MZ = h_forces[i*6+5];

        results[i].CX = FX / (q_inf * ref_area);
        results[i].CY = FY / (q_inf * ref_area);
        results[i].CZ = FZ / (q_inf * ref_area);
        results[i].Cl = MX / (q_inf * ref_area * ref_span);
        results[i].Cm = MY / (q_inf * ref_area * ref_length);
        results[i].Cn = MZ / (q_inf * ref_area * ref_span);
        results[i].iterations = h_iters[i];
        results[i].residual = h_res[i];

        float alpha_rad = alpha_deg * 3.14159265f / 180.0f;
        float beta_rad = beta_deg * 3.14159265f / 180.0f;
        float ca = cosf(alpha_rad), sa = sinf(alpha_rad);
        float cb = cosf(beta_rad), sb = sinf(beta_rad);
        float Fsx = results[i].CX * ca * cb + results[i].CY * sb + results[i].CZ * sa * cb;
        float Fsz = -results[i].CX * sa + results[i].CZ * ca;
        results[i].CD = -Fsx;
        results[i].CL = -Fsz;
    }
    return results;
}

std::vector<CfdResult> CfdSolver::solve_batch_warm(
    const std::vector<float>& machs,
    const std::vector<float>& alphas,
    const std::vector<float>& betas,
    float ref_area, float ref_length, float ref_span,
    float com_x, float com_y, float com_z,
    const CfdConfig& cfg)
{
    (void)com_x; (void)com_y; (void)com_z;
    int num_m = static_cast<int>(machs.size());
    int num_a = static_cast<int>(alphas.size());
    int num_b = static_cast<int>(betas.size());
    std::vector<CfdResult> results;
    results.reserve(static_cast<size_t>(num_m) * num_a * num_b);

    // Allocate persistent scratch if not already done
    bool own_scratch = (d_persist_Q == nullptr);
    if (own_scratch) alloc_scratch();

    float* warm_Q = nullptr;
    float prev_a = -999.0f, prev_b = -999.0f, prev_m = -999.0f;

    for (int im = 0; im < num_m; ++im) {
        for (int ia = 0; ia < num_a; ++ia) {
            for (int ib = 0; ib < num_b; ++ib) {
                float m = machs[im];
                float a = alphas[ia];
                float b = betas[ib];

                // Cold-start when α or β changes (symmetry locking)
                if (warm_Q && (fabsf(a - prev_a) > 0.1f || fabsf(b - prev_b) > 0.1f))
                    warm_Q = nullptr;

                // Cold-start when Mach jump exceeds 5
                if (im > 0 && ia == 0 && ib == 0 &&
                    fabsf(m - prev_m) > 5.0f)
                    warm_Q = nullptr;

                CfdResult res = solve(
                    m, a, b,
                    ref_area, ref_length, ref_span,
                    com_x, com_y, com_z, cfg,
                    warm_Q, &warm_Q);
                results.push_back(res);

                prev_a = a; prev_b = b; prev_m = m;
            }
        }
    }

    if (own_scratch) free_scratch();

    return results;
}

} // namespace Solver
} // namespace AeroSim
