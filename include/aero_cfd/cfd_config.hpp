#pragma once

#include "aero_cfd/diagnostics.hpp"

namespace AeroSim {
namespace Cfd {

struct CfdConfig {
    float cfl = 0.5f;
    int max_iter = 1000;
    float convergence_tol = 1e-8f;
    float gamma = 1.4f;
    float ref_area = 1.0f;
    float ref_length = 1.0f;
    float ref_span = 1.0f;
    int reconstruction_order = 1;
    bool use_gpu = false;
    bool cpu_oracle = false;
    DiagnosticLevel diagnostic_level = DiagnosticLevel::Off;
};

struct FreestreamCondition {
    float mach = 2.0f;
    float alpha_deg = 0.0f;
    float beta_deg = 0.0f;
};

} // namespace Cfd
} // namespace AeroSim
