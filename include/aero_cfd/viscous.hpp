#pragma once

#include "aero_cfd/cfd_state.hpp"

namespace AeroSim {
namespace Cfd {

float primitive_temperature(const PrimitiveState& w);

float sutherland_viscosity(
    float temperature,
    float reference_temperature = 1.0f,
    float sutherland_temperature = 0.36867f);

PrimitiveState no_slip_isothermal_wall_state(const PrimitiveState& interior, float wall_temperature);

PrimitiveState no_slip_adiabatic_wall_state(const PrimitiveState& interior);

} // namespace Cfd
} // namespace AeroSim
