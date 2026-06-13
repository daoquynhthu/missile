#pragma once
#include <cmath>
#include <vector>

namespace AeroSim {
namespace RefData {

// van Driest II turbulent boundary layer on a flat plate.
// Cf = 0.455 / [log10(Re_x)]^2.58 / F_c
// F_c accounts for compressibility:
//   F_c = (arcsin(A) + arcsin(B))^2 / A^2   (for adiabatic wall)
//   where A = (T_w/T_inf - 1) / sqrt(T_w/T_inf * (T_aw/T_inf - 1))
//         B = (T_aw/T_inf - T_w/T_inf) / sqrt(T_w/T_inf * (T_aw/T_inf - 1))
//
// Simplified: F_c ≈ sqrt(T_w/T_e) for engineering estimates.
// For reference, use F_c = 1.0 (incompressible) and F_c at M=5, adiabatic.

struct FlatPlateTurbulentPoint {
    double Re_x;
    double Cf_incomp;     // van Driest II, incompressible
    double Cf_comp_M5;    // compressible M=5, adiabatic wall
};

inline double van_driest_II_Cf(double Re_x) {
    // Incompressible van Driest II
    if (Re_x < 1e3) return 0.0;
    return 0.455 / std::pow(std::log10(Re_x), 2.58);
}

inline double vdII_compressibility_factor(double M, double gamma = 1.4) {
    // Approximate compressibility factor for adiabatic wall
    double T_aw = 1.0 + 0.5 * (gamma - 1.0) * 0.89 * M * M;
    double T_ratio = T_aw;
    double F_c = std::sqrt(T_ratio);
    return F_c;
}

inline std::vector<FlatPlateTurbulentPoint> flat_plate_turbulent_ref() {
    std::vector<double> Re_vals = {1e5, 1e6, 1e7, 1e8, 1e9};
    std::vector<FlatPlateTurbulentPoint> data;
    double F_c_M5 = vdII_compressibility_factor(5.0);
    for (double Re : Re_vals) {
        double Cf_i = van_driest_II_Cf(Re);
        double Cf_c = Cf_i / F_c_M5;
        data.push_back({Re, Cf_i, Cf_c});
    }
    return data;
}

} // namespace RefData
} // namespace AeroSim
