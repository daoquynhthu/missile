#pragma once

#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <vector>

namespace AeroSim {
namespace Cfd {

struct RansSource {
    Real production = 0.0f;
    Real destruction = 0.0f;
    Real diffusion = 0.0f;
    Real total_source = 0.0f;
};

Real sa_vorticity(const PrimitiveGradient& grad);

Real sa_omega_tilde(Real vorticity, Real nu_tilde, Real wall_distance, Real karman, Real rho, Real Re);

RansSource compute_rans_source(
    const PrimitiveState& w,
    const PrimitiveGradient& grad,
    Real wall_distance,
    Real mu,
    Real rho,
    Real Re);

std::vector<RansSource> compute_rans_sources(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    Real Re);

} // namespace Cfd
} // namespace AeroSim