#pragma once

#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_result.hpp"
#include "aero_cfd/cfd_state.hpp"

#include <vector>

namespace AeroSim {
namespace Cfd {

struct CfdSolveSummary {
    CfdForceResult forces;
    CfdDiagnostics diagnostics;
    std::vector<float> residual_history;
    bool converged = false;
    bool failed = false;
};

class CfdSolver {
public:
    bool load_mesh(const CfdMesh& mesh);

    CfdSolveSummary solve(const FreestreamCondition& condition, const CfdConfig& config);

    CfdSolveSummary solve_from_state(
        const FreestreamCondition& condition,
        const CfdConfig& config,
        const std::vector<ConservativeState>& initial_state);

    const CfdMesh& mesh() const { return mesh_; }

private:
    CfdMesh mesh_;
};

} // namespace Cfd
} // namespace AeroSim
