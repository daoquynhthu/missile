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
    std::vector<float> residual_history;
    bool converged = false;
    bool failed = false;
};

class CfdSolver {
public:
    bool load_mesh(const CfdMesh& mesh);

    CfdSolveSummary solve(const FreestreamCondition& condition, const CfdConfig& config);

    const CfdMesh& mesh() const { return mesh_; }

private:
    CfdMesh mesh_;
};

} // namespace Cfd
} // namespace AeroSim

