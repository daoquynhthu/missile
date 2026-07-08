#include "aero_cfd/real.hpp"
#pragma once

#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/reconstruction.hpp"

#include <vector>

namespace AeroSim {
namespace Cfd {

struct ViscousGradient {
    Real du_dx = 0.0f;
    Real du_dy = 0.0f;
    Real du_dz = 0.0f;
    Real dv_dx = 0.0f;
    Real dv_dy = 0.0f;
    Real dv_dz = 0.0f;
    Real dw_dx = 0.0f;
    Real dw_dy = 0.0f;
    Real dw_dz = 0.0f;
    Real dT_dx = 0.0f;
    Real dT_dy = 0.0f;
    Real dT_dz = 0.0f;
};

struct WallFlux {
    Real tau_x = 0.0f;
    Real tau_y = 0.0f;
    Real tau_z = 0.0f;
    Real q_wall = 0.0f;
    Real cf = 0.0f;
    Real st = 0.0f;
};

Real primitive_temperature(const PrimitiveState& w);

Real sutherland_viscosity(
    Real temperature,
    Real reference_temperature = 1.0f,
    Real sutherland_temperature = 0.36867f);

PrimitiveState no_slip_isothermal_wall_state(const PrimitiveState& interior, Real wall_temperature);

PrimitiveState no_slip_adiabatic_wall_state(const PrimitiveState& interior);

ViscousGradient viscous_gradient_from_primitive_gradient(
    const PrimitiveState& w,
    const PrimitiveGradient& gradient);

std::vector<ViscousGradient> compute_viscous_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    bool use_least_squares = false);

ViscousGradient orthogonal_face_gradient_correction(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const ViscousGradient& averaged_gradient,
    Real dx,
    Real dy,
    Real dz);

Real inviscid_timestep(const PrimitiveState& w, Real h, Real gamma, Real cfl);

Real viscous_timestep(Real rho, Real h, Real reynolds, Real mu, Real cfl);

WallFlux compute_wall_flux(
    const PrimitiveState& interior,
    const ViscousGradient& gradient,
    Real nx,
    Real ny,
    Real nz,
    Real mu,
    Real conductivity,
    Real q_ref,
    Real heat_ref);

} // namespace Cfd
} // namespace AeroSim


