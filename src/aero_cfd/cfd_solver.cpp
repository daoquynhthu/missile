#include "aero_cfd/cfd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace AeroSim {
namespace Cfd {

namespace {

ConservativeState add_scaled(ConservativeState q, EulerFlux f, float scale) {
    q.rho += scale * f.mass;
    q.rho_u += scale * f.mom_x;
    q.rho_v += scale * f.mom_y;
    q.rho_w += scale * f.mom_z;
    q.rho_E += scale * f.energy;
    return q;
}

float state_delta_l2(const ConservativeState& a, const ConservativeState& b) {
    float d0 = a.rho - b.rho;
    float d1 = a.rho_u - b.rho_u;
    float d2 = a.rho_v - b.rho_v;
    float d3 = a.rho_w - b.rho_w;
    float d4 = a.rho_E - b.rho_E;
    return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4;
}

void integrate_wall_forces(const CfdMesh& mesh, const std::vector<ConservativeState>& q, const FreestreamCondition& condition,
    const CfdConfig& config, CfdForceResult& result) {
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    float mx = 0.0f;
    float my = 0.0f;
    float mz = 0.0f;

    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::SlipWall && face.boundary != BoundaryKind::NoSlipWall) continue;
        PrimitiveState w;
        if (!conservative_to_primitive(q[face.left_cell], config.gamma, w)) continue;

        float px = -w.p * face.nx * face.area;
        float py = -w.p * face.ny * face.area;
        float pz = -w.p * face.nz * face.area;
        fx += px;
        fy += py;
        fz += pz;
        mx += face.cy * pz - face.cz * py;
        my += face.cz * px - face.cx * pz;
        mz += face.cx * py - face.cy * px;
    }

    float q_inf = 0.5f * condition.mach * condition.mach;
    float inv_force_ref = 1.0f / std::max(q_inf * config.ref_area, 1e-30f);
    result.CX = fx * inv_force_ref;
    result.CY = fy * inv_force_ref;
    result.CZ = fz * inv_force_ref;
    result.Cl = mx / std::max(q_inf * config.ref_area * config.ref_span, 1e-30f);
    result.Cm = my / std::max(q_inf * config.ref_area * config.ref_length, 1e-30f);
    result.Cn = mz / std::max(q_inf * config.ref_area * config.ref_span, 1e-30f);

    float alpha = condition.alpha_deg * 3.14159265358979323846f / 180.0f;
    float beta = condition.beta_deg * 3.14159265358979323846f / 180.0f;
    float ca = std::cos(alpha);
    float sa = std::sin(alpha);
    float cb = std::cos(beta);
    float sb = std::sin(beta);
    float fsx = result.CX * ca * cb + result.CY * sb + result.CZ * sa * cb;
    float fsz = -result.CX * sa + result.CZ * ca;
    result.CD = -fsx;
    result.CL = -fsz;
}

} // namespace

PrimitiveState make_freestream(float mach, float alpha_deg, float beta_deg, float gamma) {
    float alpha = alpha_deg * 3.14159265358979323846f / 180.0f;
    float beta = beta_deg * 3.14159265358979323846f / 180.0f;

    PrimitiveState w;
    w.rho = 1.0f;
    w.p = 1.0f / gamma;
    w.u = -mach * std::cos(alpha) * std::cos(beta);
    w.v = -mach * std::sin(beta);
    w.w = -mach * std::sin(alpha) * std::cos(beta);
    return w;
}

PrimitiveState farfield_ghost_state(const PrimitiveState& left, const PrimitiveState& freestream, float gamma,
    float nx, float ny, float nz) {
    float vn_inf = freestream.u*nx + freestream.v*ny + freestream.w*nz;
    float a_inf = speed_of_sound(freestream, gamma);
    if (vn_inf >= a_inf) return left;
    return freestream;
}

EulerFlux hllc_flux(const PrimitiveState& left, const PrimitiveState& right, float gamma, float nx, float ny, float nz) {
    float vn_l = left.u*nx + left.v*ny + left.w*nz;
    float vn_r = right.u*nx + right.v*ny + right.w*nz;
    float a_l = speed_of_sound(left, gamma);
    float a_r = speed_of_sound(right, gamma);
    float s_l = std::min(vn_l - a_l, vn_r - a_r);
    float s_r = std::max(vn_l + a_l, vn_r + a_r);

    EulerFlux f_l = physical_flux(left, gamma, nx, ny, nz);
    EulerFlux f_r = physical_flux(right, gamma, nx, ny, nz);
    ConservativeState q_l = primitive_to_conservative(left, gamma);
    ConservativeState q_r = primitive_to_conservative(right, gamma);

    if (s_l >= 0.0f) return f_l;
    if (s_r <= 0.0f) return f_r;

    float denom = left.rho * (s_l - vn_l) - right.rho * (s_r - vn_r);
    float s_m = (right.p - left.p + left.rho*vn_l*(s_l - vn_l) - right.rho*vn_r*(s_r - vn_r)) / denom;

    if (s_m >= 0.0f) {
        float rho_star = left.rho * (s_l - vn_l) / (s_l - s_m);
        float e_l = q_l.rho_E / left.rho;
        float p_star = left.p + left.rho * (s_l - vn_l) * (s_m - vn_l);
        float e_star = e_l + (s_m - vn_l) * (s_m + left.p / (left.rho * (s_l - vn_l)));
        ConservativeState q_star;
        q_star.rho = rho_star;
        q_star.rho_u = rho_star * (left.u + (s_m - vn_l) * nx);
        q_star.rho_v = rho_star * (left.v + (s_m - vn_l) * ny);
        q_star.rho_w = rho_star * (left.w + (s_m - vn_l) * nz);
        q_star.rho_E = rho_star * e_star;

        EulerFlux f = f_l;
        f.mass += s_l * (q_star.rho - q_l.rho);
        f.mom_x += s_l * (q_star.rho_u - q_l.rho_u);
        f.mom_y += s_l * (q_star.rho_v - q_l.rho_v);
        f.mom_z += s_l * (q_star.rho_w - q_l.rho_w);
        f.energy += s_l * (q_star.rho_E - q_l.rho_E);
        (void)p_star;
        return f;
    }

    float rho_star = right.rho * (s_r - vn_r) / (s_r - s_m);
    float e_r = q_r.rho_E / right.rho;
    float e_star = e_r + (s_m - vn_r) * (s_m + right.p / (right.rho * (s_r - vn_r)));
    ConservativeState q_star;
    q_star.rho = rho_star;
    q_star.rho_u = rho_star * (right.u + (s_m - vn_r) * nx);
    q_star.rho_v = rho_star * (right.v + (s_m - vn_r) * ny);
    q_star.rho_w = rho_star * (right.w + (s_m - vn_r) * nz);
    q_star.rho_E = rho_star * e_star;

    EulerFlux f = f_r;
    f.mass += s_r * (q_star.rho - q_r.rho);
    f.mom_x += s_r * (q_star.rho_u - q_r.rho_u);
    f.mom_y += s_r * (q_star.rho_v - q_r.rho_v);
    f.mom_z += s_r * (q_star.rho_w - q_r.rho_w);
    f.energy += s_r * (q_star.rho_E - q_r.rho_E);
    return f;
}

bool CfdSolver::load_mesh(const CfdMesh& mesh) {
    mesh_ = mesh;
    auto report = compute_mesh_metrics(mesh_);
    return report.valid;
}

CfdSolveSummary CfdSolver::solve(const FreestreamCondition& condition, const CfdConfig& config) {
    CfdSolveSummary summary;
    if (mesh_.cells.empty() || mesh_.faces.empty()) {
        summary.failed = true;
        return summary;
    }

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
    std::vector<ConservativeState> q(mesh_.cells.size(), q_inf);
    std::vector<ConservativeState> q_next(mesh_.cells.size(), q_inf);

    for (int iter = 0; iter < config.max_iter; ++iter) {
        float min_dt = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < q.size(); ++i) {
            PrimitiveState w;
            if (!conservative_to_primitive(q[i], config.gamma, w)) {
                summary.failed = true;
                return summary;
            }
            float vmag = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
            float dt = config.cfl * mesh_.cells[i].h_min / (vmag + speed_of_sound(w, config.gamma));
            min_dt = std::min(min_dt, dt);
        }

        std::vector<EulerFlux> residual(q.size());
        for (const auto& face : mesh_.faces) {
            PrimitiveState wl;
            if (!conservative_to_primitive(q[face.left_cell], config.gamma, wl)) {
                summary.failed = true;
                return summary;
            }

            EulerFlux flux;
            if (face.boundary == BoundaryKind::Interior) {
                PrimitiveState wr;
                if (!conservative_to_primitive(q[face.right_cell], config.gamma, wr)) {
                    summary.failed = true;
                    return summary;
                }
                flux = hllc_flux(wl, wr, config.gamma, face.nx, face.ny, face.nz);
            } else if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall) {
                flux = slip_wall_flux(wl, face.nx, face.ny, face.nz);
            } else {
                PrimitiveState wr = farfield_ghost_state(wl, w_inf, config.gamma, face.nx, face.ny, face.nz);
                flux = hllc_flux(wl, wr, config.gamma, face.nx, face.ny, face.nz);
            }

            float area = face.area;
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

        float l2 = 0.0f;
        for (std::size_t i = 0; i < q.size(); ++i) {
            float scale = min_dt / mesh_.cells[i].volume;
            q_next[i] = add_scaled(q[i], residual[i], scale);
            l2 += state_delta_l2(q_next[i], q[i]);
        }
        float residual_l2 = std::sqrt(l2 / (5.0f * static_cast<float>(q.size())));
        summary.residual_history.push_back(residual_l2);
        q.swap(q_next);

        if (residual_l2 < config.convergence_tol) {
            summary.converged = true;
            break;
        }
    }

    integrate_wall_forces(mesh_, q, condition, config, summary.forces);
    summary.forces.iterations = static_cast<int>(summary.residual_history.size());
    summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
    return summary;
}

} // namespace Cfd
} // namespace AeroSim
