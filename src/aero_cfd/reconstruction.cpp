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

float limiter_theta(float center, float reconstructed, float min_value, float max_value) {
    if (reconstructed > max_value) {
        float denom = reconstructed - center;
        if (denom <= 0.0f) return 0.0f;
        return std::max(0.0f, std::min(1.0f, (max_value - center) / denom));
    }
    if (reconstructed < min_value) {
        float denom = reconstructed - center;
        if (denom >= 0.0f) return 0.0f;
        return std::max(0.0f, std::min(1.0f, (min_value - center) / denom));
    }
    return 1.0f;
}

void update_minmax(PrimitiveState& min_w, PrimitiveState& max_w, const PrimitiveState& w) {
    min_w.rho = std::min(min_w.rho, w.rho);
    min_w.u = std::min(min_w.u, w.u);
    min_w.v = std::min(min_w.v, w.v);
    min_w.w = std::min(min_w.w, w.w);
    min_w.p = std::min(min_w.p, w.p);
    max_w.rho = std::max(max_w.rho, w.rho);
    max_w.u = std::max(max_w.u, w.u);
    max_w.v = std::max(max_w.v, w.v);
    max_w.w = std::max(max_w.w, w.w);
    max_w.p = std::max(max_w.p, w.p);
}

void update_limiter(PrimitiveLimiter& limiter, const PrimitiveState& center, const PrimitiveState& reconstructed,
    const PrimitiveState& min_w, const PrimitiveState& max_w) {
    limiter.rho = std::min(limiter.rho, limiter_theta(center.rho, reconstructed.rho, min_w.rho, max_w.rho));
    limiter.u = std::min(limiter.u, limiter_theta(center.u, reconstructed.u, min_w.u, max_w.u));
    limiter.v = std::min(limiter.v, limiter_theta(center.v, reconstructed.v, min_w.v, max_w.v));
    limiter.w = std::min(limiter.w, limiter_theta(center.w, reconstructed.w, min_w.w, max_w.w));
    limiter.p = std::min(limiter.p, limiter_theta(center.p, reconstructed.p, min_w.p, max_w.p));
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

std::vector<PrimitiveLimiter> compute_barth_jespersen_limiters(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    float gamma) {
    if (q.size() != mesh.cells.size() || gradients.size() != mesh.cells.size()) return {};

    std::vector<PrimitiveState> primitive(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (!conservative_to_primitive(q[i], gamma, primitive[i])) return {};
    }

    std::vector<PrimitiveState> min_w = primitive;
    std::vector<PrimitiveState> max_w = primitive;
    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) continue;
        update_minmax(min_w[face.left_cell], max_w[face.left_cell], primitive[face.right_cell]);
        update_minmax(min_w[face.right_cell], max_w[face.right_cell], primitive[face.left_cell]);
    }

    std::vector<PrimitiveLimiter> limiters(q.size());
    for (const auto& face : mesh.faces) {
        const CfdCell& left = mesh.cells[face.left_cell];
        PrimitiveState left_recon = reconstruct_primitive(
            primitive[face.left_cell],
            gradients[face.left_cell],
            face.cx - left.cx,
            face.cy - left.cy,
            face.cz - left.cz);
        update_limiter(limiters[face.left_cell], primitive[face.left_cell], left_recon,
            min_w[face.left_cell], max_w[face.left_cell]);

        if (face.boundary == BoundaryKind::Interior) {
            const CfdCell& right = mesh.cells[face.right_cell];
            PrimitiveState right_recon = reconstruct_primitive(
                primitive[face.right_cell],
                gradients[face.right_cell],
                face.cx - right.cx,
                face.cy - right.cy,
                face.cz - right.cz);
            update_limiter(limiters[face.right_cell], primitive[face.right_cell], right_recon,
                min_w[face.right_cell], max_w[face.right_cell]);
        }
    }

    return limiters;
}

PrimitiveGradient apply_limiter(const PrimitiveGradient& gradient, const PrimitiveLimiter& limiter) {
    PrimitiveGradient out = gradient;
    out.drho_dx *= limiter.rho;
    out.drho_dy *= limiter.rho;
    out.drho_dz *= limiter.rho;
    out.du_dx *= limiter.u;
    out.du_dy *= limiter.u;
    out.du_dz *= limiter.u;
    out.dv_dx *= limiter.v;
    out.dv_dy *= limiter.v;
    out.dv_dz *= limiter.v;
    out.dw_dx *= limiter.w;
    out.dw_dy *= limiter.w;
    out.dw_dz *= limiter.w;
    out.dp_dx *= limiter.p;
    out.dp_dy *= limiter.p;
    out.dp_dz *= limiter.p;
    return out;
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
