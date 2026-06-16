#include "aero_cfd/viscous.hpp"

#include <algorithm>
#include <cmath>

namespace AeroSim {
namespace Cfd {

float primitive_temperature(const PrimitiveState& w) {
    return w.p / std::max(w.rho, 1e-30f);
}

float sutherland_viscosity(
    float temperature,
    float reference_temperature,
    float sutherland_temperature) {
    if (temperature <= 0.0f || reference_temperature <= 0.0f || sutherland_temperature < 0.0f) {
        return 0.0f;
    }
    float t_ratio = temperature / reference_temperature;
    return std::pow(t_ratio, 1.5f) * (reference_temperature + sutherland_temperature) /
        (temperature + sutherland_temperature);
}

PrimitiveState no_slip_isothermal_wall_state(const PrimitiveState& interior, float wall_temperature) {
    PrimitiveState wall = interior;
    wall.u = 0.0f;
    wall.v = 0.0f;
    wall.w = 0.0f;
    wall.rho = interior.p / std::max(wall_temperature, 1e-30f);
    wall.p = interior.p;
    return wall;
}

PrimitiveState no_slip_adiabatic_wall_state(const PrimitiveState& interior) {
    return no_slip_isothermal_wall_state(interior, primitive_temperature(interior));
}

ViscousGradient viscous_gradient_from_primitive_gradient(
    const PrimitiveState& w,
    const PrimitiveGradient& gradient) {
    ViscousGradient out;
    out.du_dx = gradient.du_dx;
    out.du_dy = gradient.du_dy;
    out.du_dz = gradient.du_dz;
    out.dv_dx = gradient.dv_dx;
    out.dv_dy = gradient.dv_dy;
    out.dv_dz = gradient.dv_dz;
    out.dw_dx = gradient.dw_dx;
    out.dw_dy = gradient.dw_dy;
    out.dw_dz = gradient.dw_dz;

    float inv_rho2 = 1.0f / std::max(w.rho * w.rho, 1e-30f);
    out.dT_dx = (w.rho * gradient.dp_dx - w.p * gradient.drho_dx) * inv_rho2;
    out.dT_dy = (w.rho * gradient.dp_dy - w.p * gradient.drho_dy) * inv_rho2;
    out.dT_dz = (w.rho * gradient.dp_dz - w.p * gradient.drho_dz) * inv_rho2;
    return out;
}

std::vector<ViscousGradient> compute_viscous_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    float gamma,
    bool use_least_squares) {
    std::vector<PrimitiveGradient> primitive_gradients = use_least_squares ?
        compute_least_squares_gradients(mesh, q, gamma) :
        compute_green_gauss_gradients(mesh, q, gamma);
    if (primitive_gradients.size() != q.size()) return {};

    std::vector<ViscousGradient> gradients(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        PrimitiveState w;
        if (!conservative_to_primitive(q[i], gamma, w)) return {};
        gradients[i] = viscous_gradient_from_primitive_gradient(w, primitive_gradients[i]);
    }
    return gradients;
}

} // namespace Cfd
} // namespace AeroSim
