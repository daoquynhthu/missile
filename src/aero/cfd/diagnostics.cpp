#include "aero/cfd/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

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

    for (std::size_t i = 0; i < q.size(); ++i) {
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
    out << "aerosp CFD cell diagnostics\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << mesh.nodes.size() << " float\n";
    for (const auto& node : mesh.nodes) {
        out << node.x << " " << node.y << " " << node.z << "\n";
    }

    // Count total VTK cell data size: each entry = (n_nodes + 1) + node_ids
    std::size_t total_entries = 0;
    for (const auto& cell : mesh.cells) {
        total_entries += 1 + ELEMENT_NODES[static_cast<int>(cell.type)];
    }
    out << "CELLS " << mesh.cells.size() << " " << total_entries << "\n";
    for (const auto& cell : mesh.cells) {
        int nn = ELEMENT_NODES[static_cast<int>(cell.type)];
        out << nn;
        for (int i = 0; i < nn; ++i) {
            out << " " << cell.node[i];
        }
        out << "\n";
    }

    // VTK cell types: 10=tetra, 12=hex, 13=prism, 14=pyramid
    constexpr int VTK_TETRA = 10;
    constexpr int VTK_HEX = 12;
    constexpr int VTK_WEDGE = 13;
    constexpr int VTK_PYRAMID = 14;
    out << "CELL_TYPES " << mesh.cells.size() << "\n";
    for (const auto& cell : mesh.cells) {
        switch (cell.type) {
            case ElementType::TET4:     out << VTK_TETRA << "\n"; break;
            case ElementType::HEX8:     out << VTK_HEX << "\n"; break;
            case ElementType::PENTA6:   out << VTK_WEDGE << "\n"; break;
            case ElementType::PYRAMID5: out << VTK_PYRAMID << "\n"; break;
        }
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

} // namespace cfd
} // namespace aero
} // namespace aerosp

