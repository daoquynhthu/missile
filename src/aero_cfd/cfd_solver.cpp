#include "aero_cfd/cfd_solver.hpp"
#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/rans.hpp"
#include "aero_cfd/reconstruction.hpp"
#include "aero_cfd/gpu_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace AeroSim {
namespace Cfd {

namespace {

ConservativeState add_scaled(ConservativeState q, EulerFlux f, Real scale) {
    q.rho += scale * f.mass;
    q.rho_u += scale * f.mom_x;
    q.rho_v += scale * f.mom_y;
    q.rho_w += scale * f.mom_z;
    q.rho_E += scale * f.energy;
    q.rho_nu_tilde += scale * f.turbulence;
    return q;
}

Real state_delta_l2(const ConservativeState& a, const ConservativeState& b) {
    Real d0 = a.rho - b.rho;
    Real d1 = a.rho_u - b.rho_u;
    Real d2 = a.rho_v - b.rho_v;
    Real d3 = a.rho_w - b.rho_w;
    Real d4 = a.rho_E - b.rho_E;
    Real d5 = a.rho_nu_tilde - b.rho_nu_tilde;
    return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
}

void integrate_wall_forces(const CfdMesh& mesh, const std::vector<ConservativeState>& q, const FreestreamCondition& condition,
    const CfdConfig& config, CfdForceResult& result) {
    Real fx = 0.0f;
    Real fy = 0.0f;
    Real fz = 0.0f;
    Real mx = 0.0f;
    Real my = 0.0f;
    Real mz = 0.0f;

    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::SlipWall && face.boundary != BoundaryKind::NoSlipWall) continue;
        PrimitiveState w;
        if (!conservative_to_primitive(q[face.left_cell], config.gamma, w)) continue;

        Real px = -w.p * face.nx * face.area;
        Real py = -w.p * face.ny * face.area;
        Real pz = -w.p * face.nz * face.area;
        fx += px;
        fy += py;
        fz += pz;
        mx += face.cy * pz - face.cz * py;
        my += face.cz * px - face.cx * pz;
        mz += face.cx * py - face.cy * px;
    }

    Real q_inf = 0.5f * condition.mach * condition.mach;
    Real inv_force_ref = 1.0f / std::max(q_inf * config.ref_area, 1e-30f);
    result.CX = fx * inv_force_ref;
    result.CY = fy * inv_force_ref;
    result.CZ = fz * inv_force_ref;
    result.Cl = mx / std::max(q_inf * config.ref_area * config.ref_span, 1e-30f);
    result.Cm = my / std::max(q_inf * config.ref_area * config.ref_length, 1e-30f);
    result.Cn = mz / std::max(q_inf * config.ref_area * config.ref_span, 1e-30f);

    Real alpha = condition.alpha_deg * 3.14159265358979323846 / 180.0;
    Real beta = condition.beta_deg * 3.14159265358979323846 / 180.0;
    Real ca = std::cos(alpha);
    Real sa = std::sin(alpha);
    Real cb = std::cos(beta);
    Real sb = std::sin(beta);
    Real fsx = result.CX * ca * cb + result.CY * sb + result.CZ * sa * cb;
    Real fsz = -result.CX * sa + result.CZ * ca;
    result.CD = -fsx;
    result.CL = -fsz;
}

} // namespace

PrimitiveState make_freestream(Real mach, Real alpha_deg, Real beta_deg, Real gamma) {
    Real alpha = alpha_deg * 3.14159265358979323846 / 180.0;
    Real beta = beta_deg * 3.14159265358979323846 / 180.0;

    PrimitiveState w;
    w.rho = 1.0f;
    w.p = 1.0f / gamma;
    w.u = -mach * std::cos(alpha) * std::cos(beta);
    w.v = -mach * std::sin(beta);
    w.w = -mach * std::sin(alpha) * std::cos(beta);
    return w;
}

PrimitiveState farfield_ghost_state(const PrimitiveState& left, const PrimitiveState& freestream, Real gamma,
    Real nx, Real ny, Real nz) {
    Real vn_inf = freestream.u*nx + freestream.v*ny + freestream.w*nz;
    Real a_inf = speed_of_sound(freestream, gamma);
    if (vn_inf >= a_inf) return left;
    return freestream;
}

EulerFlux hllc_flux(const PrimitiveState& left, const PrimitiveState& right, Real gamma, Real nx, Real ny, Real nz) {
    Real vn_l = left.u*nx + left.v*ny + left.w*nz;
    Real vn_r = right.u*nx + right.v*ny + right.w*nz;
    Real a_l = speed_of_sound(left, gamma);
    Real a_r = speed_of_sound(right, gamma);
    Real s_l = std::min(vn_l - a_l, vn_r - a_r);
    Real s_r = std::max(vn_l + a_l, vn_r + a_r);

    EulerFlux f_l = physical_flux(left, gamma, nx, ny, nz);
    EulerFlux f_r = physical_flux(right, gamma, nx, ny, nz);
    ConservativeState q_l = primitive_to_conservative(left, gamma);
    ConservativeState q_r = primitive_to_conservative(right, gamma);

    if (s_l >= 0.0f) return f_l;
    if (s_r <= 0.0f) return f_r;

    Real denom = left.rho * (s_l - vn_l) - right.rho * (s_r - vn_r);
    Real s_m = (right.p - left.p + left.rho*vn_l*(s_l - vn_l) - right.rho*vn_r*(s_r - vn_r)) / denom;

    if (s_m >= 0.0f) {
        Real rho_star = left.rho * (s_l - vn_l) / (s_l - s_m);
        Real e_l = q_l.rho_E / left.rho;
        Real p_star = left.p + left.rho * (s_l - vn_l) * (s_m - vn_l);
        Real e_star = e_l + (s_m - vn_l) * (s_m + left.p / (left.rho * (s_l - vn_l)));
        ConservativeState q_star;
        q_star.rho = rho_star;
        q_star.rho_u = rho_star * (left.u + (s_m - vn_l) * nx);
        q_star.rho_v = rho_star * (left.v + (s_m - vn_l) * ny);
        q_star.rho_w = rho_star * (left.w + (s_m - vn_l) * nz);
        q_star.rho_E = rho_star * e_star;
        q_star.rho_nu_tilde = q_l.rho_nu_tilde * (s_l - vn_l) / (s_l - s_m);

        EulerFlux f = f_l;
        f.mass += s_l * (q_star.rho - q_l.rho);
        f.mom_x += s_l * (q_star.rho_u - q_l.rho_u);
        f.mom_y += s_l * (q_star.rho_v - q_l.rho_v);
        f.mom_z += s_l * (q_star.rho_w - q_l.rho_w);
        f.energy += s_l * (q_star.rho_E - q_l.rho_E);
        f.turbulence += s_l * (q_star.rho_nu_tilde - q_l.rho_nu_tilde);
        (void)p_star;
        return f;
    }

    Real rho_star = right.rho * (s_r - vn_r) / (s_r - s_m);
    Real e_r = q_r.rho_E / right.rho;
    Real e_star = e_r + (s_m - vn_r) * (s_m + right.p / (right.rho * (s_r - vn_r)));
    ConservativeState q_star;
    q_star.rho = rho_star;
    q_star.rho_u = rho_star * (right.u + (s_m - vn_r) * nx);
    q_star.rho_v = rho_star * (right.v + (s_m - vn_r) * ny);
    q_star.rho_w = rho_star * (right.w + (s_m - vn_r) * nz);
    q_star.rho_E = rho_star * e_star;
    q_star.rho_nu_tilde = q_r.rho_nu_tilde * (s_r - vn_r) / (s_r - s_m);

    EulerFlux f = f_r;
    f.mass += s_r * (q_star.rho - q_r.rho);
    f.mom_x += s_r * (q_star.rho_u - q_r.rho_u);
    f.mom_y += s_r * (q_star.rho_v - q_r.rho_v);
    f.mom_z += s_r * (q_star.rho_w - q_r.rho_w);
    f.energy += s_r * (q_star.rho_E - q_r.rho_E);
    f.turbulence += s_r * (q_star.rho_nu_tilde - q_r.rho_nu_tilde);
    return f;
}

bool CfdSolver::load_mesh(const CfdMesh& mesh) {
    mesh_ = mesh;
    auto report = compute_mesh_metrics(mesh_);
    return report.valid;
}

CfdSolveSummary CfdSolver::solve(const FreestreamCondition& condition, const CfdConfig& config) {
    if (config.use_gpu) {
        DeviceMesh d_mesh;
        if (!d_mesh.upload_mesh(mesh_)) {
            CfdSolveSummary s;
            s.failed = true;
            return s;
        }
    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
        ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
        std::vector<ConservativeState> q(mesh_.cells.size(), q_inf);
        if (!d_mesh.upload_state(q)) {
            CfdSolveSummary s;
            s.failed = true;
            return s;
        }
        CfdSolveSummary gpu_result = solve_gpu(d_mesh, condition, config);

        if (config.cpu_oracle && !gpu_result.failed) {
            CfdConfig cpu_cfg = config;
            cpu_cfg.use_gpu = false;
            CfdSolveSummary cpu_result = solve_from_state(condition, cpu_cfg, q);
            std::string oracle_error;
            if (!assert_oracle_equivalent(gpu_result, cpu_result, 1e-6f, 1e-6f, &oracle_error)) {
                std::fprintf(stderr, "[CPU Oracle] FAIL: %s\n", oracle_error.c_str());
                CfdSolveSummary s;
                s.failed = true;
                return s;
            }
        }
        return gpu_result;
    }

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
    std::vector<ConservativeState> q(mesh_.cells.size(), q_inf);
    return solve_from_state(condition, config, q);
}

CfdSolveSummary CfdSolver::solve_from_state(
    const FreestreamCondition& condition,
    const CfdConfig& config,
    const std::vector<ConservativeState>& initial_state) {
    CfdSolveSummary summary;
    if (mesh_.cells.empty() || mesh_.faces.empty()) {
        summary.failed = true;
        return summary;
    }
    if (initial_state.size() != mesh_.cells.size()) {
        summary.failed = true;
        return summary;
    }

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    std::vector<ConservativeState> q = initial_state;
    std::vector<ConservativeState> q_next = initial_state;
    bool diagnostics_enabled = config.diagnostic_level != DiagnosticLevel::Off;

    if (diagnostics_enabled) {
        StateBounds bounds = compute_state_bounds(q, config.gamma);
        summary.diagnostics.state_bounds_history.push_back(bounds);
        if (!bounds.valid) {
            int cell = bounds.bad_cell >= 0 ? bounds.bad_cell : 0;
            summary.failed = true;
            summary.diagnostics.failure = make_failure_snapshot(0, cell, "invalid initial state", q[cell], config.gamma);
            return summary;
        }
    }

    for (int iter = 0; iter < config.max_iter; ++iter) {
        Real min_dt = std::numeric_limits<Real>::max();
        DtLimiterSnapshot dt_limiter;
        dt_limiter.iteration = iter;
        for (std::size_t i = 0; i < q.size(); ++i) {
            PrimitiveState w;
            if (!conservative_to_primitive(q[i], config.gamma, w)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure = make_failure_snapshot(iter, static_cast<int>(i), "invalid state before timestep", q[i], config.gamma);
                }
                return summary;
            }
            Real vmag = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
            Real signal_speed = vmag + speed_of_sound(w, config.gamma);
            Real dt = config.cfl * mesh_.cells[i].h_min / signal_speed;
            if (dt < min_dt) {
                min_dt = dt;
                dt_limiter.cell = static_cast<int>(i);
                dt_limiter.dt = dt;
                dt_limiter.h_min = mesh_.cells[i].h_min;
                dt_limiter.signal_speed = signal_speed;
            }
        }
        if (diagnostics_enabled) {
            summary.diagnostics.dt_limiter_history.push_back(dt_limiter);
        }

        std::vector<EulerFlux> residual;
        if (!compute_euler_residual_cpu(mesh_, q, w_inf, config.gamma, residual)) {
            summary.failed = true;
            if (diagnostics_enabled) {
                summary.diagnostics.failure.reason = "residual assembly failed";
                summary.diagnostics.failure.valid = true;
                summary.diagnostics.failure.iteration = iter;
            }
            return summary;
        }

        if (config.turbulence) {
            std::vector<PrimitiveGradient> grads = compute_green_gauss_gradients(mesh_, q, config.gamma);
            if (grads.size() != mesh_.cells.size()) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "gradient computation failed for RANS source";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            std::vector<PrimitiveLimiter> limiters = compute_barth_jespersen_limiters(mesh_, q, grads, config.gamma);
            if (limiters.size() != mesh_.cells.size()) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "limiter computation failed for RANS source";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            std::vector<PrimitiveGradient> limited(grads.size());
            for (std::size_t i = 0; i < grads.size(); ++i)
                limited[i] = apply_limiter(grads[i], limiters[i]);

            std::vector<RansSource> sources = compute_rans_sources(mesh_, q, limited, config.gamma, config.Re);
            if (sources.size() != mesh_.cells.size()) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "RANS source computation failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            for (std::size_t i = 0; i < q.size(); ++i)
                residual[i].turbulence += sources[i].total_source * mesh_.cells[i].volume;
        }

        Real l2 = 0.0f;
        for (std::size_t i = 0; i < q.size(); ++i) {
            Real scale = min_dt / mesh_.cells[i].volume;
            q_next[i] = add_scaled(q[i], residual[i], scale);
            l2 += state_delta_l2(q_next[i], q[i]);
        }
        Real residual_l2 = std::sqrt(l2 / (static_cast<Real>(CFD_NVAR) * static_cast<Real>(q.size())));
        summary.residual_history.push_back(residual_l2);
        q.swap(q_next);

        if (diagnostics_enabled) {
            StateBounds bounds = compute_state_bounds(q, config.gamma);
            summary.diagnostics.state_bounds_history.push_back(bounds);
            if (!bounds.valid) {
                int cell = bounds.bad_cell >= 0 ? bounds.bad_cell : 0;
                summary.failed = true;
                summary.diagnostics.failure = make_failure_snapshot(iter + 1, cell, "invalid state after update", q[cell], config.gamma);
                return summary;
            }
        }

        if (residual_l2 < config.convergence_tol) {
            summary.converged = true;
            break;
        }
    }

integrate_wall_forces(mesh_, q, condition, config, summary.forces);
    summary.forces.iterations = static_cast<int>(summary.residual_history.size());
    summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
    summary.forces.turbulence_model = config.turbulence ? "rans-sa" : "laminar";
    summary.forces.fidelity = "cfd-cpu";
    return summary;
}

bool assert_oracle_equivalent(
    const CfdSolveSummary& gpu,
    const CfdSolveSummary& cpu,
    Real tol_residual,
    Real tol_forces,
    std::string* error) {
    std::size_t n = std::min(gpu.residual_history.size(), cpu.residual_history.size());
    for (std::size_t i = 0; i < n; ++i) {
        Real diff = std::fabs(gpu.residual_history[i] - cpu.residual_history[i]);
        Real base = 1.0f + std::max(std::fabs(gpu.residual_history[i]), std::fabs(cpu.residual_history[i]));
        if (diff > tol_residual * base) {
            if (error) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "residual iter=%zu GPU=%g CPU=%g diff=%g", i,
                    gpu.residual_history[i], cpu.residual_history[i], diff);
                *error = buf;
            }
            return false;
        }
    }

    struct ForcePair { const char* name; Real g; Real c; };
    ForcePair pairs[] = {
        {"CX", gpu.forces.CX, cpu.forces.CX},
        {"CY", gpu.forces.CY, cpu.forces.CY},
        {"CZ", gpu.forces.CZ, cpu.forces.CZ},
        {"Cl", gpu.forces.Cl, cpu.forces.Cl},
        {"Cm", gpu.forces.Cm, cpu.forces.Cm},
        {"Cn", gpu.forces.Cn, cpu.forces.Cn},
        {"CD", gpu.forces.CD, cpu.forces.CD},
        {"CL", gpu.forces.CL, cpu.forces.CL},
    };
    for (const auto& p : pairs) {
        Real diff = std::fabs(p.g - p.c);
        Real base = 1.0f + std::max(std::fabs(p.g), std::fabs(p.c));
        if (diff > tol_forces * base) {
            if (error) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "force %s GPU=%g CPU=%g diff=%g", p.name, p.g, p.c, diff);
                *error = buf;
            }
            return false;
        }
    }
    return true;
}

} // namespace Cfd
} // namespace AeroSim

