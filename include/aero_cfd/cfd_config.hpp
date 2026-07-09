#pragma once

#include "aero_cfd/real.hpp"
#include "aero_cfd/diagnostics.hpp"

namespace AeroSim {
namespace Cfd {

constexpr int CFD_NVAR = 6;

struct CfdConfig {
    Real cfl = 0.5f;
    int max_iter = 1000;
    Real convergence_tol = 1e-8f;
    Real gamma = 1.4f;
    Real ref_area = 1.0f;
    Real ref_length = 1.0f;
    Real ref_span = 1.0f;
    int reconstruction_order = 1;
    bool use_gpu = false;
    bool cpu_oracle = false;
    DiagnosticLevel diagnostic_level = DiagnosticLevel::Off;

    // Viscous NS parameters
    bool viscous = false;
    Real Re = 1e6f;
    Real prandtl = 0.72f;
    Real mu_ref = 1.0f;
    Real T_ref = 288.15f;
    Real sutherland_T = 110.4f;
    Real wall_temperature = 300.0f;

    // RANS SA turbulence
    bool turbulence = false;
};

struct FreestreamCondition {
    Real mach = 2.0f;
    Real alpha_deg = 0.0f;
    Real beta_deg = 0.0f;
    Real nu_tilde = 0.0f;
};

} // namespace Cfd
} // namespace AeroSim
