#pragma once

#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/reconstruction.hpp"

#include <vector>

namespace AeroSim {
namespace Cfd {

struct ViscousGradient {
    float du_dx = 0.0f;
    float du_dy = 0.0f;
    float du_dz = 0.0f;
    float dv_dx = 0.0f;
    float dv_dy = 0.0f;
    float dv_dz = 0.0f;
    float dw_dx = 0.0f;
    float dw_dy = 0.0f;
    float dw_dz = 0.0f;
    float dT_dx = 0.0f;
    float dT_dy = 0.0f;
    float dT_dz = 0.0f;
};

float primitive_temperature(const PrimitiveState& w);

float sutherland_viscosity(
    float temperature,
    float reference_temperature = 1.0f,
    float sutherland_temperature = 0.36867f);

PrimitiveState no_slip_isothermal_wall_state(const PrimitiveState& interior, float wall_temperature);

PrimitiveState no_slip_adiabatic_wall_state(const PrimitiveState& interior);

ViscousGradient viscous_gradient_from_primitive_gradient(
    const PrimitiveState& w,
    const PrimitiveGradient& gradient);

std::vector<ViscousGradient> compute_viscous_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    float gamma,
    bool use_least_squares = false);

} // namespace Cfd
} // namespace AeroSim
