#include "aero_cfd/cfd_residual.hpp"

namespace AeroSim {
namespace Cfd {

bool compute_euler_residual_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual) {
    if (q.size() != mesh.cells.size()) return false;
    residual.assign(q.size(), EulerFlux{});

    for (const auto& face : mesh.faces) {
        PrimitiveState wl;
        if (!conservative_to_primitive(q[face.left_cell], gamma, wl)) return false;

        EulerFlux flux;
        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState wr;
            if (!conservative_to_primitive(q[face.right_cell], gamma, wr)) return false;
            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall) {
            flux = slip_wall_flux(wl, face.nx, face.ny, face.nz);
        } else {
            PrimitiveState wr = farfield_ghost_state(wl, freestream, gamma, face.nx, face.ny, face.nz);
            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        }

        Real area = face.area;
        residual[face.left_cell].mass -= flux.mass * area;
        residual[face.left_cell].mom_x -= flux.mom_x * area;
        residual[face.left_cell].mom_y -= flux.mom_y * area;
        residual[face.left_cell].mom_z -= flux.mom_z * area;
        residual[face.left_cell].energy -= flux.energy * area;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mass += flux.mass * area;
            residual[face.right_cell].mom_x += flux.mom_x * area;
            residual[face.right_cell].mom_y += flux.mom_y * area;
            residual[face.right_cell].mom_z += flux.mom_z * area;
            residual[face.right_cell].energy += flux.energy * area;
        }
    }

    return true;
}

} // namespace Cfd
} // namespace AeroSim

