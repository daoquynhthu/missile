#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/reconstruction.hpp"

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
        residual[face.left_cell].turbulence -= flux.turbulence * area;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mass += flux.mass * area;
            residual[face.right_cell].mom_x += flux.mom_x * area;
            residual[face.right_cell].mom_y += flux.mom_y * area;
            residual[face.right_cell].mom_z += flux.mom_z * area;
            residual[face.right_cell].energy += flux.energy * area;
            residual[face.right_cell].turbulence += flux.turbulence * area;
        }
    }

    return true;
}

bool compute_euler_residual_cpu_order2(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual) {
    if (q.size() != mesh.cells.size()) return false;
    residual.assign(q.size(), EulerFlux{});

    std::vector<PrimitiveGradient> gradients = compute_green_gauss_gradients(mesh, q, gamma);
    if (gradients.size() != mesh.cells.size()) return false;

    std::vector<PrimitiveLimiter> limiters = compute_barth_jespersen_limiters(mesh, q, gradients, gamma);
    if (limiters.size() != mesh.cells.size()) return false;

    std::vector<PrimitiveGradient> limited(gradients.size());
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        limited[i] = apply_limiter(gradients[i], limiters[i]);
    }

    for (const auto& face : mesh.faces) {
        PrimitiveState w_center;
        if (!conservative_to_primitive(q[face.left_cell], gamma, w_center)) return false;

        PrimitiveState wl = reconstruct_primitive(w_center, limited[face.left_cell],
            face.cx - mesh.cells[face.left_cell].cx,
            face.cy - mesh.cells[face.left_cell].cy,
            face.cz - mesh.cells[face.left_cell].cz);
        if (wl.rho <= 0.0f || wl.p <= 0.0f) return false;

        EulerFlux flux;
        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState w_center_r;
            if (!conservative_to_primitive(q[face.right_cell], gamma, w_center_r)) return false;

            PrimitiveState wr = reconstruct_primitive(w_center_r, limited[face.right_cell],
                face.cx - mesh.cells[face.right_cell].cx,
                face.cy - mesh.cells[face.right_cell].cy,
                face.cz - mesh.cells[face.right_cell].cz);
            if (wr.rho <= 0.0f || wr.p <= 0.0f) return false;

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
        residual[face.left_cell].turbulence -= flux.turbulence * area;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mass += flux.mass * area;
            residual[face.right_cell].mom_x += flux.mom_x * area;
            residual[face.right_cell].mom_y += flux.mom_y * area;
            residual[face.right_cell].mom_z += flux.mom_z * area;
            residual[face.right_cell].energy += flux.energy * area;
            residual[face.right_cell].turbulence += flux.turbulence * area;
        }
    }

    return true;
}

} // namespace Cfd
} // namespace AeroSim

