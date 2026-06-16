#pragma once

namespace AeroSim {
namespace Cfd {

struct CfdConfig {
    float cfl = 0.5f;
    int max_iter = 1000;
    float convergence_tol = 1e-8f;
    float gamma = 1.4f;
};

struct FreestreamCondition {
    float mach = 2.0f;
    float alpha_deg = 0.0f;
    float beta_deg = 0.0f;
};

} // namespace Cfd
} // namespace AeroSim

