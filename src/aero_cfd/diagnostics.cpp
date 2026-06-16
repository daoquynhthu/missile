#include "aero_cfd/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace AeroSim {
namespace Cfd {

StateBounds compute_state_bounds(const std::vector<ConservativeState>& q, float gamma) {
    StateBounds bounds;
    if (q.empty()) return bounds;

    bounds.min_rho = std::numeric_limits<float>::max();
    bounds.min_p = std::numeric_limits<float>::max();
    bounds.min_mach = std::numeric_limits<float>::max();
    bounds.max_rho = -std::numeric_limits<float>::max();
    bounds.max_p = -std::numeric_limits<float>::max();
    bounds.max_mach = -std::numeric_limits<float>::max();
    bounds.valid = true;

    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        PrimitiveState w;
        if (!conservative_to_primitive(q[i], gamma, w)) {
            bounds.valid = false;
            bounds.bad_cell = i;
            return bounds;
        }

        float a = speed_of_sound(w, gamma);
        float vmag = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        float mach = vmag / std::max(a, 1e-30f);
        bounds.min_rho = std::min(bounds.min_rho, w.rho);
        bounds.max_rho = std::max(bounds.max_rho, w.rho);
        bounds.min_p = std::min(bounds.min_p, w.p);
        bounds.max_p = std::max(bounds.max_p, w.p);
        bounds.min_mach = std::min(bounds.min_mach, mach);
        bounds.max_mach = std::max(bounds.max_mach, mach);
    }

    return bounds;
}

FailureSnapshot make_failure_snapshot(
    int iteration,
    int cell,
    const char* reason,
    const ConservativeState& q,
    float gamma) {
    FailureSnapshot snapshot;
    snapshot.valid = true;
    snapshot.iteration = iteration;
    snapshot.cell = cell;
    snapshot.reason = reason ? reason : "";
    snapshot.state = q;
    conservative_to_primitive(q, gamma, snapshot.primitive);
    return snapshot;
}

} // namespace Cfd
} // namespace AeroSim
