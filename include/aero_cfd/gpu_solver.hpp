#pragma once

#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_result.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

struct CfdSolveSummary;

// GPU solver loop — Phase 2 target.
// Currently a placeholder; the full implementation will replace CfdSolver::solve()
// with a device-side iteration loop in Phase 2.

} // namespace Cfd
} // namespace AeroSim
