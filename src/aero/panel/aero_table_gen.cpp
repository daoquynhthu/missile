#include "aero/panel/aero_solver.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_mesh.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace aerosp {
namespace aero {
namespace panel {

bool generate_aero_table(
    const std::string& stl_path,
    const std::string& csv_path,
    const std::vector<double>& mach_grid,
    const std::vector<double>& alpha_grid,
    const std::vector<double>& beta_grid,
    const AeroTableConfig& cfg)
{
    if (mach_grid.empty() || alpha_grid.empty() || beta_grid.empty()) {
        std::cerr << "[aero_table_gen] Input grids must be non-empty (got mach="
                  << mach_grid.size() << " alpha=" << alpha_grid.size()
                  << " beta=" << beta_grid.size() << ")\n";
        return false;
    }

    AeroGeometry eng_geo;
    eng_geo.ref_area = cfg.ref_area;
    eng_geo.ref_length = cfg.ref_length;
    eng_geo.ref_span = cfg.ref_span;
    eng_geo.wet_area = cfg.wet_area;
    eng_geo.planform_area = cfg.planform_area;
    eng_geo.base_area = cfg.base_area;
    eng_geo.nose_fineness = cfg.nose_fineness;

    std::vector<BatchCondition> conditions;
    for (double m : mach_grid) {
        for (double a : alpha_grid) {
            for (double b : beta_grid) {
                conditions.push_back({
                    static_cast<float>(m),
                    static_cast<float>(a),
                    static_cast<float>(b),
                    cfg.com_x, cfg.com_y, cfg.com_z,
                    288.15f,
                    0.0f,
                    0.0f
                });
            }
        }
    }

    std::cout << "[aero_table_gen] Computing " << conditions.size()
              << " conditions (" << mach_grid.size() << " Mach x "
              << alpha_grid.size() << " Alpha x " << beta_grid.size()
              << " Beta)...\n";

    bool use_cfd = cfg.use_fvm;
    std::vector<aerosp::aero::cfd::CfdForceResult> cfd_results;
    std::vector<BatchResult> newtonian_results;

    if (use_cfd) {
        const double MACH_MIN  = 1.2;
        const double MACH_MAX  = 30.0;
        const double ALPHA_LIM = 30.0;
        const double BETA_LIM  = 10.0;

        for (auto& c : conditions) {
            if (c.mach < MACH_MIN || c.mach > MACH_MAX ||
                std::abs(c.alpha_deg) > ALPHA_LIM ||
                std::abs(c.beta_deg) > BETA_LIM) {
                std::cerr << "[aero_table_gen] CFD range: Mach [" << MACH_MIN << ", " << MACH_MAX
                          << "], |alpha| <= " << ALPHA_LIM << ", |beta| <= " << BETA_LIM
                          << ". Got Mach=" << c.mach << " alpha=" << c.alpha_deg
                          << " beta=" << c.beta_deg << "\n";
                return false;
            }
        }

        if (cfg.mesh_subdivisions < 0) {
            std::cerr << "[aero_table_gen] Warning: mesh_subdivisions="
                      << cfg.mesh_subdivisions << " is negative, using absolute value\n";
        }
        double effective_sub = std::max(1.0, static_cast<double>(std::abs(cfg.mesh_subdivisions)));
        int n = std::max(5, static_cast<int>(
            std::ceil(std::pow(effective_sub / 5.0, 1.0 / 3.0)) + 1.0));
        n = std::min(n, 100);

        std::cout << "[aero_table_gen] Generating cube mesh: outer_scale=" << cfg.mesh_outer_scale
                  << " n_per_dim=" << n << "\n";
        aerosp::aero::cfd::CfdMesh mesh = aerosp::aero::cfd::generate_structured_cube_mesh(
            static_cast<Real>(cfg.mesh_outer_scale), n);

        aerosp::aero::cfd::MeshQualityReport qr = aerosp::aero::cfd::compute_mesh_metrics(mesh);
        if (!qr.valid || qr.cells == 0) {
            std::cerr << "[aero_table_gen] Mesh invalid: " << qr.message
                      << " (cells=" << qr.cells << ")\n";
            return false;
        }

        aerosp::aero::cfd::CfdConfig cfd_cfg;
        cfd_cfg.use_gpu = true;
        cfd_cfg.ref_area  = static_cast<Real>(cfg.ref_area);
        cfd_cfg.ref_length = static_cast<Real>(cfg.ref_length);
        cfd_cfg.ref_span  = static_cast<Real>(cfg.ref_span);
        cfd_cfg.viscous   = cfg.viscous;
        cfd_cfg.Re        = static_cast<Real>(cfg.Re);
        cfd_cfg.prandtl   = static_cast<Real>(cfg.prandtl);
        cfd_cfg.wall_temperature = static_cast<Real>(cfg.wall_temperature);

        aerosp::aero::cfd::CfdSolver cfd_solver;
        if (!cfd_solver.load_mesh(mesh)) {
            std::cerr << "[aero_table_gen] Failed to load mesh into CFD solver\n";
            return false;
        }

        cfd_results.resize(conditions.size());
        for (size_t i = 0; i < conditions.size(); ++i) {
            auto& c = conditions[i];
            aerosp::aero::cfd::FreestreamCondition fc;
            fc.mach      = static_cast<Real>(c.mach);
            fc.alpha_deg = static_cast<Real>(c.alpha_deg);
            fc.beta_deg  = static_cast<Real>(c.beta_deg);

            std::cout << "[aero_table_gen] CFD solve [" << (i + 1) << "/" << conditions.size()
                      << "] Mach=" << fc.mach << " alpha=" << fc.alpha_deg << " beta=" << fc.beta_deg << "\n";

            auto summary = cfd_solver.solve(fc, cfd_cfg);
            if (summary.failed) {
                std::cerr << "[aero_table_gen] CFD failed Mach=" << fc.mach
                          << " alpha=" << fc.alpha_deg << " beta=" << fc.beta_deg
                          << " — skipping\n";
                continue;
            }

            summary.forces.fidelity = "cfd-gpu";
            cfd_results[i] = summary.forces;
        }
    } else {
        AeroSolver solver;
        if (!solver.load_model(stl_path, cfg.ref_area, cfg.ref_length, cfg.ref_span)) {
            std::cerr << "[aero_table_gen] Failed to load STL: " << stl_path << "\n";
            return false;
        }
        solver.set_moment_ref_point(cfg.com_x, cfg.com_y, cfg.com_z);

        newtonian_results = solver.compute_batch(conditions, eng_geo);
    }

    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "[aero_table_gen] Failed to write: " << csv_path << "\n";
        return false;
    }

    if (use_cfd) {
        csv << "Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn,Fidelity,TurbulenceModel\n";
        for (size_t i = 0; i < conditions.size(); ++i) {
            auto& c = conditions[i];
            auto& r = cfd_results[i];
            double ld = (std::abs(r.CD) > Real(1e-12)) ? static_cast<double>(r.CL / r.CD) : 0.0;
            csv << c.mach << "," << c.alpha_deg << "," << c.beta_deg << ","
                << r.CX << "," << r.CY << "," << r.CZ << ","
                << r.CL << "," << r.CD << "," << ld << ","
                << r.Cl << "," << r.Cm << "," << r.Cn << ","
                << r.fidelity << ","
                << r.turbulence_model << "\n";
        }
    } else {
        csv << "Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn\n";
        for (size_t i = 0; i < newtonian_results.size(); ++i) {
            auto& r = newtonian_results[i];
            auto& c = conditions[i];
            double ld = (std::abs(r.CD) > 1e-12f) ? static_cast<double>(r.CL / r.CD) : 0.0;
            csv << c.mach << "," << c.alpha_deg << "," << c.beta_deg << ","
                << r.CX << "," << r.CY << "," << r.CZ << ","
                << r.CL << "," << r.CD << "," << ld << ","
                << r.Cl << "," << r.Cm << "," << r.Cn << "\n";
        }
    }

    csv.close();
    std::cout << "[aero_table_gen] Done. Wrote " << conditions.size()
              << " rows to " << csv_path << "\n";
    return true;
}

} // namespace panel
} // namespace aero
} // namespace aerosp
