#include "aero_cfd/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace AeroSim {
namespace Cfd {

StateBounds compute_state_bounds(const std::vector<ConservativeState>& q, Real gamma) {
    StateBounds bounds;
    if (q.empty()) return bounds;

    bounds.min_rho = std::numeric_limits<Real>::max();
    bounds.min_p = std::numeric_limits<Real>::max();
    bounds.min_mach = std::numeric_limits<Real>::max();
    bounds.max_rho = -std::numeric_limits<Real>::max();
    bounds.max_p = -std::numeric_limits<Real>::max();
    bounds.max_mach = -std::numeric_limits<Real>::max();
    bounds.valid = true;

    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        PrimitiveState w;
        if (!conservative_to_primitive(q[i], gamma, w)) {
            bounds.valid = false;
            bounds.bad_cell = i;
            return bounds;
        }

        Real a = speed_of_sound(w, gamma);
        Real vmag = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        Real mach = vmag / std::max(a, 1e-30f);
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
    Real gamma) {
    FailureSnapshot snapshot;
    snapshot.valid = true;
    snapshot.iteration = iteration;
    snapshot.cell = cell;
    snapshot.reason = reason ? reason : "";
    snapshot.state = q;
    conservative_to_primitive(q, gamma, snapshot.primitive);
    return snapshot;
}

bool write_vtk_cells(
    const char* path,
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    std::string* error) {
    if (!path || !path[0]) {
        if (error) *error = "empty path";
        return false;
    }
    if (q.size() != mesh.cells.size()) {
        if (error) *error = "state size does not match mesh cells";
        return false;
    }

    std::vector<PrimitiveState> primitive(q.size());
    std::vector<Real> mach(q.size(), 0.0f);
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (!conservative_to_primitive(q[i], gamma, primitive[i])) {
            if (error) *error = "invalid state";
            return false;
        }
        Real a = speed_of_sound(primitive[i], gamma);
        Real vmag = std::sqrt(
            primitive[i].u*primitive[i].u +
            primitive[i].v*primitive[i].v +
            primitive[i].w*primitive[i].w);
        mach[i] = vmag / std::max(a, 1e-30f);
    }

    std::ofstream out(path);
    if (!out) {
        if (error) *error = "failed to open path";
        return false;
    }

    out << "# vtk DataFile Version 3.0\n";
    out << "AeroSim CFD cell diagnostics\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << mesh.nodes.size() << " float\n";
    for (const auto& node : mesh.nodes) {
        out << node.x << " " << node.y << " " << node.z << "\n";
    }

    out << "CELLS " << mesh.cells.size() << " " << mesh.cells.size() * 5 << "\n";
    for (const auto& cell : mesh.cells) {
        out << "4 " << cell.node[0] << " " << cell.node[1] << " " << cell.node[2] << " " << cell.node[3] << "\n";
    }

    out << "CELL_TYPES " << mesh.cells.size() << "\n";
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        out << "10\n";
    }

    out << "CELL_DATA " << mesh.cells.size() << "\n";
    out << "SCALARS rho float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto& w : primitive) out << w.rho << "\n";
    out << "SCALARS pressure float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto& w : primitive) out << w.p << "\n";
    out << "SCALARS mach float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (Real value : mach) out << value << "\n";

    if (!out) {
        if (error) *error = "failed while writing";
        return false;
    }
    return true;
}

} // namespace Cfd
} // namespace AeroSim

