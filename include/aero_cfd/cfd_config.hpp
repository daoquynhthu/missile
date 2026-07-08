#pragma once

#include "aero_cfd/real.hpp"
#include "aero_cfd/diagnostics.hpp"

namespace AeroSim {
namespace Cfd {

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
};

struct FreestreamCondition {
    Real mach = 2.0f;
    Real alpha_deg = 0.0f;
    Real beta_deg = 0.0f;
};

} // namespace Cfd
} // namespace AeroSim
