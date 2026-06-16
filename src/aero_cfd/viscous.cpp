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

ViscousGradient orthogonal_face_gradient_correction(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const ViscousGradient& averaged_gradient,
    float dx,
    float dy,
    float dz) {
    ViscousGradient corrected = averaged_gradient;
    float d2 = dx*dx + dy*dy + dz*dz;
    if (d2 <= 1e-30f) return corrected;

    float inv_d2 = 1.0f / d2;
    float projected_du = averaged_gradient.du_dx*dx + averaged_gradient.du_dy*dy + averaged_gradient.du_dz*dz;
    float projected_dv = averaged_gradient.dv_dx*dx + averaged_gradient.dv_dy*dy + averaged_gradient.dv_dz*dz;
    float projected_dw = averaged_gradient.dw_dx*dx + averaged_gradient.dw_dy*dy + averaged_gradient.dw_dz*dz;
    float projected_dT = averaged_gradient.dT_dx*dx + averaged_gradient.dT_dy*dy + averaged_gradient.dT_dz*dz;

    float left_T = primitive_temperature(left);
    float right_T = primitive_temperature(right);
    float du_corr = ((right.u - left.u) - projected_du) * inv_d2;
    float dv_corr = ((right.v - left.v) - projected_dv) * inv_d2;
    float dw_corr = ((right.w - left.w) - projected_dw) * inv_d2;
    float dT_corr = ((right_T - left_T) - projected_dT) * inv_d2;

    corrected.du_dx += du_corr * dx;
    corrected.du_dy += du_corr * dy;
    corrected.du_dz += du_corr * dz;
    corrected.dv_dx += dv_corr * dx;
    corrected.dv_dy += dv_corr * dy;
    corrected.dv_dz += dv_corr * dz;
    corrected.dw_dx += dw_corr * dx;
    corrected.dw_dy += dw_corr * dy;
    corrected.dw_dz += dw_corr * dz;
    corrected.dT_dx += dT_corr * dx;
    corrected.dT_dy += dT_corr * dy;
    corrected.dT_dz += dT_corr * dz;
    return corrected;
}

float inviscid_timestep(const PrimitiveState& w, float h, float gamma, float cfl) {
    float vmag = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
    return cfl * h / std::max(vmag + speed_of_sound(w, gamma), 1e-30f);
}

float viscous_timestep(float rho, float h, float reynolds, float mu, float cfl) {
    if (rho <= 0.0f || h <= 0.0f || reynolds <= 0.0f || mu <= 0.0f) return 0.0f;
    return cfl * rho * h * h * reynolds / mu;
}

WallFlux compute_wall_flux(
    const PrimitiveState& interior,
    const ViscousGradient& gradient,
    float nx,
    float ny,
    float nz,
    float mu,
    float conductivity,
    float q_ref,
    float heat_ref) {
    WallFlux out;
    float div_u = gradient.du_dx + gradient.dv_dy + gradient.dw_dz;
    float tauxx = 2.0f * mu * (gradient.du_dx - div_u / 3.0f);
    float tauyy = 2.0f * mu * (gradient.dv_dy - div_u / 3.0f);
    float tauzz = 2.0f * mu * (gradient.dw_dz - div_u / 3.0f);
    float tauxy = mu * (gradient.du_dy + gradient.dv_dx);
    float tauxz = mu * (gradient.du_dz + gradient.dw_dx);
    float tauyz = mu * (gradient.dv_dz + gradient.dw_dy);

    float tx = tauxx*nx + tauxy*ny + tauxz*nz;
    float ty = tauxy*nx + tauyy*ny + tauyz*nz;
    float tz = tauxz*nx + tauyz*ny + tauzz*nz;
    float normal_tau = tx*nx + ty*ny + tz*nz;
    out.tau_x = tx - normal_tau*nx;
    out.tau_y = ty - normal_tau*ny;
    out.tau_z = tz - normal_tau*nz;

    out.q_wall = -conductivity * (gradient.dT_dx*nx + gradient.dT_dy*ny + gradient.dT_dz*nz);
    float tau_mag = std::sqrt(out.tau_x*out.tau_x + out.tau_y*out.tau_y + out.tau_z*out.tau_z);
    out.cf = tau_mag / std::max(q_ref, 1e-30f);
    out.st = out.q_wall / std::max(heat_ref, 1e-30f);
    (void)interior;
    return out;
}

} // namespace Cfd
} // namespace AeroSim
