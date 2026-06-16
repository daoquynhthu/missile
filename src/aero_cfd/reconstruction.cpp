#include "aero_cfd/reconstruction.hpp"

#include <algorithm>
#include <cmath>

namespace AeroSim {
namespace Cfd {

namespace {

PrimitiveState average_state(const PrimitiveState& a, const PrimitiveState& b) {
    PrimitiveState out;
    out.rho = 0.5f * (a.rho + b.rho);
    out.u = 0.5f * (a.u + b.u);
    out.v = 0.5f * (a.v + b.v);
    out.w = 0.5f * (a.w + b.w);
    out.p = 0.5f * (a.p + b.p);
    return out;
}

void add_face_contribution(PrimitiveGradient& g, const PrimitiveState& w, const PrimitiveState& center,
    float nx, float ny, float nz, float scale) {
    float sx = nx * scale;
    float sy = ny * scale;
    float sz = nz * scale;
    g.drho_dx += (w.rho - center.rho) * sx;
    g.drho_dy += (w.rho - center.rho) * sy;
    g.drho_dz += (w.rho - center.rho) * sz;
    g.du_dx += (w.u - center.u) * sx;
    g.du_dy += (w.u - center.u) * sy;
    g.du_dz += (w.u - center.u) * sz;
    g.dv_dx += (w.v - center.v) * sx;
    g.dv_dy += (w.v - center.v) * sy;
    g.dv_dz += (w.v - center.v) * sz;
    g.dw_dx += (w.w - center.w) * sx;
    g.dw_dy += (w.w - center.w) * sy;
    g.dw_dz += (w.w - center.w) * sz;
    g.dp_dx += (w.p - center.p) * sx;
    g.dp_dy += (w.p - center.p) * sy;
    g.dp_dz += (w.p - center.p) * sz;
}

float positive_theta(float center, float reconstructed, float floor_value) {
    if (reconstructed >= floor_value) return 1.0f;
    if (center <= floor_value) return 0.0f;
    float denom = center - reconstructed;
    if (denom <= 0.0f || !std::isfinite(denom)) return 0.0f;
    return std::max(0.0f, std::min(1.0f, (center - floor_value) / denom));
}

PrimitiveGradient scale_gradient(const PrimitiveGradient& g, float theta) {
    PrimitiveGradient out = g;
    out.drho_dx *= theta;
    out.drho_dy *= theta;
    out.drho_dz *= theta;
    out.du_dx *= theta;
    out.du_dy *= theta;
    out.du_dz *= theta;
    out.dv_dx *= theta;
    out.dv_dy *= theta;
    out.dv_dz *= theta;
    out.dw_dx *= theta;
    out.dw_dy *= theta;
    out.dw_dz *= theta;
    out.dp_dx *= theta;
    out.dp_dy *= theta;
    out.dp_dz *= theta;
    return out;
}

} // namespace

std::vector<PrimitiveGradient> compute_green_gauss_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    float gamma) {
    if (q.size() != mesh.cells.size()) return {};

    std::vector<PrimitiveState> primitive(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (!conservative_to_primitive(q[i], gamma, primitive[i])) return {};
    }

    std::vector<PrimitiveGradient> gradients(q.size());
    for (const auto& face : mesh.faces) {
        const PrimitiveState& wl = primitive[face.left_cell];
        PrimitiveState wf = wl;
        if (face.boundary == BoundaryKind::Interior) {
            wf = average_state(wl, primitive[face.right_cell]);
        }

        float left_scale = face.area / mesh.cells[face.left_cell].volume;
        add_face_contribution(gradients[face.left_cell], wf, wl, face.nx, face.ny, face.nz, left_scale);

        if (face.boundary == BoundaryKind::Interior) {
            const PrimitiveState& wr = primitive[face.right_cell];
            float right_scale = -face.area / mesh.cells[face.right_cell].volume;
            add_face_contribution(gradients[face.right_cell], wf, wr, face.nx, face.ny, face.nz, right_scale);
        }
    }

    return gradients;
}

PrimitiveState reconstruct_primitive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    float dx,
    float dy,
    float dz) {
    PrimitiveState out;
    out.rho = center.rho + gradient.drho_dx*dx + gradient.drho_dy*dy + gradient.drho_dz*dz;
    out.u = center.u + gradient.du_dx*dx + gradient.du_dy*dy + gradient.du_dz*dz;
    out.v = center.v + gradient.dv_dx*dx + gradient.dv_dy*dy + gradient.dv_dz*dz;
    out.w = center.w + gradient.dw_dx*dx + gradient.dw_dy*dy + gradient.dw_dz*dz;
    out.p = center.p + gradient.dp_dx*dx + gradient.dp_dy*dy + gradient.dp_dz*dz;
    return out;
}

PrimitiveState reconstruct_primitive_positive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    float dx,
    float dy,
    float dz,
    float rho_floor,
    float p_floor,
    float* theta) {
    PrimitiveState raw = reconstruct_primitive(center, gradient, dx, dy, dz);
    float t = std::min(
        positive_theta(center.rho, raw.rho, rho_floor),
        positive_theta(center.p, raw.p, p_floor));
    if (theta) *theta = t;
    return reconstruct_primitive(center, scale_gradient(gradient, t), dx, dy, dz);
}

} // namespace Cfd
} // namespace AeroSim
