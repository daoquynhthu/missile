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

} // namespace Cfd
} // namespace AeroSim
