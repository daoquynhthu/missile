#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_state.hpp"

#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

enum class DiagnosticLevel : int {
    Off = 0,
    Basic = 1,
    Detailed = 2,
    Verbose = 3
};

struct StateBounds {
    float min_rho = 0.0f;
    float max_rho = 0.0f;
    float min_p = 0.0f;
    float max_p = 0.0f;
    float min_mach = 0.0f;
    float max_mach = 0.0f;
    int bad_cell = -1;
    bool valid = false;
};

struct DtLimiterSnapshot {
    int iteration = -1;
    int cell = -1;
    float dt = 0.0f;
    float h_min = 0.0f;
    float signal_speed = 0.0f;
};

struct FailureSnapshot {
    bool valid = false;
    int iteration = -1;
    int cell = -1;
    std::string reason;
    ConservativeState state;
    PrimitiveState primitive;
};

struct CfdDiagnostics {
    std::vector<StateBounds> state_bounds_history;
    std::vector<DtLimiterSnapshot> dt_limiter_history;
    FailureSnapshot failure;
};

StateBounds compute_state_bounds(const std::vector<ConservativeState>& q, float gamma);

FailureSnapshot make_failure_snapshot(
    int iteration,
    int cell,
    const char* reason,
    const ConservativeState& q,
    float gamma);

bool write_vtk_cells(
    const char* path,
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    float gamma,
    std::string* error = nullptr);

} // namespace Cfd
} // namespace AeroSim
