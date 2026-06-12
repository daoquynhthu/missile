#include "aero_solver/aero_solver.hpp"
#include "aero_solver/engineering_aero.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

namespace AeroSim {
namespace Solver {

bool generate_aero_table(
    const std::string& stl_path,
    const std::string& csv_path,
    const std::vector<double>& mach_grid,
    const std::vector<double>& alpha_grid,
    const std::vector<double>& beta_grid,
    const AeroTableConfig& cfg)
{
    // 1. Load STL
    AeroSolver solver;
    if (!solver.load_model(stl_path, cfg.ref_area, cfg.ref_length, cfg.ref_span)) {
        std::cerr << "[aero_table_gen] Failed to load STL: " << stl_path << "\n";
        return false;
    }
    solver.set_moment_ref_point(cfg.com_x, cfg.com_y, cfg.com_z);

    // 2. Build condition array
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
                    288.15f,   // T_ref (placeholder, filled by compute_batch)
                    0.0f,      // rho_ref (filled by compute_batch)
                    0.0f       // mu_ref (filled by compute_batch)
                });
            }
        }
    }

    std::cout << "[aero_table_gen] Computing " << conditions.size()
              << " conditions (" << mach_grid.size() << " Mach x "
              << alpha_grid.size() << " Alpha x " << beta_grid.size()
              << " Beta)...\n";

    // 3. Single GPU pass
    auto results = solver.compute_batch(conditions, eng_geo);

    // 4. CPU-side blend for transition region (Mach 4-6)
    AeroGeometry h_geo;
    h_geo.ref_area = static_cast<float>(cfg.ref_area);
    h_geo.ref_length = static_cast<float>(cfg.ref_length);
    h_geo.ref_span = static_cast<float>(cfg.ref_span);
    h_geo.wet_area = static_cast<float>(cfg.wet_area);
    h_geo.planform_area = static_cast<float>(cfg.planform_area);
    h_geo.base_area = static_cast<float>(cfg.base_area);
    h_geo.nose_fineness = static_cast<float>(cfg.nose_fineness);

    for (size_t i = 0; i < conditions.size(); ++i) {
        double mach = conditions[i].mach;
        double alpha_rad = conditions[i].alpha_deg * 3.141592653589793 / 180.0;
        double beta_rad  = conditions[i].beta_deg  * 3.141592653589793 / 180.0;

        if (mach >= 4.0 && mach <= 6.0) {
            auto eng = compute_engineering_coeffs(h_geo, mach, alpha_rad, beta_rad);

            if (mach >= 5.0) {
                double t = (mach - 4.0) / 2.0;
                t = std::max(0.0, std::min(1.0, t));
                auto blend = [t](double a, double b) { return (1.0 - t) * a + t * b; };
                results[i].CX = static_cast<float>(blend(eng.CX, results[i].CX));
                results[i].CY = static_cast<float>(blend(eng.CY, results[i].CY));
                results[i].CZ = static_cast<float>(blend(eng.CZ, results[i].CZ));
                results[i].Cl = static_cast<float>(blend(eng.Cl, results[i].Cl));
                results[i].Cm = static_cast<float>(blend(eng.Cm, results[i].Cm));
                results[i].Cn = static_cast<float>(blend(eng.Cn, results[i].Cn));
                results[i].CL = static_cast<float>(blend(eng.CL, results[i].CL));
                results[i].CD = static_cast<float>(blend(eng.CD, results[i].CD));
            }
        }
    }

    // 5. Write CSV
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "[aero_table_gen] Failed to write: " << csv_path << "\n";
        return false;
    }

    csv << "Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn\n";
    for (size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        auto& c = conditions[i];
        double ld = (std::abs(r.CD) > 1e-12f) ? static_cast<double>(r.CL / r.CD) : 0.0;
        csv << c.mach << "," << c.alpha_deg << "," << c.beta_deg << ","
            << r.CX << "," << r.CY << "," << r.CZ << ","
            << r.CL << "," << r.CD << "," << ld << ","
            << r.Cl << "," << r.Cm << "," << r.Cn << "\n";
    }

    csv.close();
    std::cout << "[aero_table_gen] Done. Wrote " << results.size()
              << " rows to " << csv_path << "\n";
    return true;
}

} // namespace Solver
} // namespace AeroSim
