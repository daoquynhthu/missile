#include "aero_cfd/rans.hpp"
#include "aero_cfd/viscous.hpp"
#include "aero_cfd/cfd_mesh.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace AeroSim {
namespace Cfd {

Real sa_vorticity(const PrimitiveGradient& grad) {
    Real vort_x = grad.dw_dy - grad.dv_dz;
    Real vort_y = grad.du_dz - grad.dw_dx;
    Real vort_z = grad.dv_dx - grad.du_dy;
    return std::sqrt(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z);
}

Real sa_omega_tilde(Real vorticity, Real nu_tilde, Real wall_distance, Real karman) {
    Real chi = nu_tilde / (1.0f / 1e6f);
    Real fv1 = chi*chi*chi / (chi*chi*chi + karman*karman*karman);
    Real inv_kd2 = 1.0f / (karman * karman * wall_distance * wall_distance);
    return vorticity + nu_tilde * fv1 * inv_kd2;
}

RansSource compute_rans_source(
    const PrimitiveState& w,
    const PrimitiveGradient& grad,
    Real wall_distance,
    Real mu,
    Real rho,
    Real Re) {
    RansSource s;

    if (wall_distance <= 0.0f || !std::isfinite(wall_distance)) {
        s.total_source = 0.0f;
        return s;
    }

    Real nu = mu / rho;

    Real chi = w.nu_tilde / nu;
    Real chi3 = chi*chi*chi;
    constexpr Real cv1 = 7.1f;
    Real cv13 = cv1*cv1*cv1;
    Real fv1 = chi3 / (chi3 + cv13);

    Real vort = sa_vorticity(grad);
    Real omega_tilde = sa_omega_tilde(vort, w.nu_tilde, wall_distance, 0.41f);

    constexpr Real cb1 = 0.1355f;
    constexpr Real cb2 = 0.622f;
    constexpr Real sigma = 2.0f / 3.0f;
    constexpr Real cw2 = 0.3f;
    constexpr Real cw3 = 2.0f;
    constexpr Real karman = 0.41f;

    Real production = cb1 * omega_tilde * w.nu_tilde;

    Real r = w.nu_tilde / (omega_tilde * karman * karman * wall_distance * wall_distance + 1e-30f);
    Real r6 = r*r*r*r*r*r;
    Real cw1 = cb1 / (karman*karman) + (1.0f + cb2) / sigma;
    Real g = r + cw2 * (r6 - r);
    Real fw = g * std::pow((1.0f + cw3*cw3*cw3*cw3*cw3*cw3) / (g*g*g*g*g*g + cw3*cw3*cw3*cw3*cw3*cw3), 1.0f / 6.0f);
    Real destruction = cw1 * fw * (w.nu_tilde / wall_distance) * (w.nu_tilde / wall_distance);

    Real grad_nu2 = grad.dnu_tilde_dx * grad.dnu_tilde_dx
                 + grad.dnu_tilde_dy * grad.dnu_tilde_dy
                 + grad.dnu_tilde_dz * grad.dnu_tilde_dz;
    Real diffusion = (cb2 / sigma) * grad_nu2;

    Real source = production - destruction + diffusion;
    Real vol_source = rho * source;

    s.production = production;
    s.destruction = destruction;
    s.diffusion = diffusion;
    s.total_source = vol_source;

    return s;
}

std::vector<RansSource> compute_rans_sources(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    Real Re) {
    std::vector<RansSource> sources(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        PrimitiveState w;
        if (!conservative_to_primitive(q[i], gamma, w)) continue;
        Real mu = 1.0f;
        sources[i] = compute_rans_source(
            w, gradients[i], mesh.cells[i].h_min, mu, q[i].rho, Re);
    }
    return sources;
}

} // namespace Cfd
} // namespace AeroSim