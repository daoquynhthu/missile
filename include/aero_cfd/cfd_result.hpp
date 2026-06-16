#pragma once

namespace AeroSim {
namespace Cfd {

struct CfdForceResult {
    float CX = 0.0f;
    float CY = 0.0f;
    float CZ = 0.0f;
    float CD = 0.0f;
    float CL = 0.0f;
    float Cl = 0.0f;
    float Cm = 0.0f;
    float Cn = 0.0f;
    float Q_wall = 0.0f;
    float q_stag = 0.0f;
    int iterations = 0;
    float residual = 0.0f;
};

} // namespace Cfd
} // namespace AeroSim

