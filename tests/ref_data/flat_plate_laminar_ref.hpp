#pragma once
#include <cmath>
#include <vector>

namespace AeroSim {
namespace RefData {

// Blasius laminar boundary layer on a flat plate (incompressible).
// With van Driest compressibility correction:
//   Cf_compressible = Cf_incompressible / F_c
//   F_c = sqrt(T_w/T_e), T_e = freestream recovery temperature
//
// For adiabatic wall: T_w = T_e = T_inf * (1 + r*(gamma-1)/2 * M^2)
// For isothermal wall: T_w specified

struct FlatPlateLaminarPoint {
    double Re_x;
    double Cf_incomp;
    double Cf_comp;  // compressible Cf at M=5, T_w/T_e=1
    double St_incomp;
    double St_comp;
};

// Generate reference data for specified Reynolds numbers and Mach numbers.
// If M=0, returns incompressible values.
inline std::vector<FlatPlateLaminarPoint> flat_plate_laminar_ref(
    double Pr = 0.72, double M = 5.0, double T_ratio = 1.0)
{
    std::vector<double> Re_vals = {1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    std::vector<FlatPlateLaminarPoint> data;
    double gamma = 1.4;
    // van Driest compressibility factor for laminar
    double F_c = M > 0 ? std::sqrt(T_ratio) : 1.0;
    for (double Re : Re_vals) {
        double Cf_i = 0.664 / std::sqrt(Re);
        double Cf_c = Cf_i / F_c;
        double St_i = 0.332 / std::sqrt(Re) / std::pow(Pr, 2.0/3.0);
        double St_c = St_i / F_c;
        data.push_back({Re, Cf_i, Cf_c, St_i, St_c});
    }
    return data;
}

} // namespace RefData
} // namespace AeroSim
