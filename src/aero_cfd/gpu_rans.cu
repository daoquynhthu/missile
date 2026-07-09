#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/real.hpp"
#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
namespace AeroSim {
namespace Cfd {

namespace {

__device__ Real d_sa_vorticity(const PrimitiveGradient& g) {
    Real vx = g.dw_dy - g.dv_dz;
    Real vy = g.du_dz - g.dw_dx;
    Real vz = g.dv_dx - g.du_dy;
    return real_sqrt(vx*vx + vy*vy + vz*vz);
}

__global__ void rans_source_kernel(
    Real* d_q,
    Real* d_residual,
    const Real* d_gradients,
    const Real* d_volume,
    const Real* d_wall_distance,
    int n_cells, int nvar,
    Real gamma, Real Re,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[idx * nvar + 4] - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real wall_distance = d_wall_distance[idx];
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    const PrimitiveGradient* g = reinterpret_cast<const PrimitiveGradient*>(d_gradients) + idx;
    Real nu_tilde = d_q[idx * nvar + 5] * inv_rho;

    if (!real_isfinite(nu_tilde)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    constexpr Real karman = 0.41f;
    constexpr Real cb1 = 0.1355f;
    constexpr Real cb2 = 0.622f;
    constexpr Real sigma = 2.0f / 3.0f;
    constexpr Real cw2 = 0.3f;
    constexpr Real cw3 = 2.0f;
    constexpr Real cv1 = 7.1f;
    constexpr Real ct3 = 1.2f;
    constexpr Real ct4 = 0.5f;

    Real grad_nu2 = g->dnu_tilde_dx * g->dnu_tilde_dx
                 + g->dnu_tilde_dy * g->dnu_tilde_dy
                 + g->dnu_tilde_dz * g->dnu_tilde_dz
                 + 1e-30f;
    Real diffusion = (cb2 / sigma) * grad_nu2;

    Real chi = Re * rho * nu_tilde + 1e-30f;

    Real source;
    if (chi >= 0.0f) {
        Real chi3 = chi*chi*chi;
        Real cv13 = cv1*cv1*cv1;
        Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);

        Real vort = d_sa_vorticity(*g);
        Real fv2 = 1.0f - chi / (1.0f + chi * fv1 + 1e-30f);
        Real chi_fv2_nu = nu_tilde * fv2;
        Real inv_kd2 = 1.0f / (karman * karman * wall_distance * wall_distance + 1e-30f);
        Real omega_tilde = vort + chi_fv2_nu * inv_kd2;

        Real production = cb1 * omega_tilde * nu_tilde;

        Real r = nu_tilde / (omega_tilde * karman * karman * wall_distance * wall_distance + 1e-30f);
        Real r6 = r*r*r*r*r*r;
        Real cw1 = cb1 / (karman*karman) + (1.0f + cb2) / sigma;
        Real fw_g = r + cw2 * (r6 - r);
        Real fw_num = 1.0f + cw3*cw3*cw3*cw3*cw3*cw3;
        Real fw_den = fw_g*fw_g*fw_g*fw_g*fw_g*fw_g + cw3*cw3*cw3*cw3*cw3*cw3 + 1e-30f;
        Real fw = fw_g * powf(fw_num / fw_den, 1.0f / 6.0f);
        Real destruction = cw1 * fw * (nu_tilde / wall_distance) * (nu_tilde / wall_distance);

        source = production - destruction + diffusion;
    } else {
        Real ft2 = ct3 * expf(-ct4 * chi * chi);
        Real vort = d_sa_vorticity(*g);
        Real cw1 = cb1 / (karman*karman) + (1.0f + cb2) / sigma;
        source = cb1 * (1.0f - ft2) * vort * nu_tilde
               - cw1 * (nu_tilde / wall_distance) * (nu_tilde / wall_distance)
               + diffusion;
    }

    Real vol_source = rho * source;

    if (!real_isfinite(vol_source)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    d_residual[idx * nvar + 5] += vol_source;
}

} // namespace

bool compute_rans_source_gpu(DeviceMesh& mesh, Real gamma, Real Re, int* d_failed, std::string* error) {
    if (mesh.cell_count() == 0) return true;
    if (!mesh.gradients_device()) {
        if (error) *error = "gradients not allocated for RANS source";
        return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    rans_source_kernel<<<grid, block>>>(
        mesh.state_device(),
        mesh.residual_device(),
        mesh.gradients_device(),
        cd.volume, cd.wall_distance,
        nc, DeviceMesh::NVAR, gamma, Re,
        d_failed);
    if (!cuda_check(cudaGetLastError(), "rans_source_kernel launch")) return false;
    return true;
}

} // namespace Cfd
} // namespace AeroSim