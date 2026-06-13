#pragma once
#include <cmath>
#include <vector>

namespace AeroSim {
namespace RefData {

// Normal shock relations for calorically perfect gas (γ=1.4).
// Also provides equilibrium post-shock composition for Park 5-species air
// (N2, O2, N, O, NO) computed from simplified equilibrium constants.

struct NormalShockPoint {
    double M1;         // Upstream Mach
    double p_ratio;    // p2/p1
    double T_ratio;    // T2/T1
    double rho_ratio;  // rho2/rho1
    double M2;         // Downstream Mach
    // Park 5-species mass fractions (equilibrium at post-shock T, p=1atm)
    double Y_N2, Y_O2, Y_N, Y_O, Y_NO;
};

// Rankine-Hugoniot for calorically perfect gas
inline double shock_p_ratio(double M, double g = 1.4) {
    return 1.0 + 2.0 * g / (g + 1.0) * (M * M - 1.0);
}
inline double shock_rho_ratio(double M, double g = 1.4) {
    return (g + 1.0) * M * M / ((g - 1.0) * M * M + 2.0);
}
inline double shock_T_ratio(double M, double g = 1.4) {
    double pr = shock_p_ratio(M, g);
    double dr = shock_rho_ratio(M, g);
    return pr / dr;
}
inline double shock_M2(double M, double g = 1.4) {
    double num = (g - 1.0) * M * M + 2.0;
    double den = 2.0 * g * M * M - (g - 1.0);
    return std::sqrt(num / den);
}

// Simplified equilibrium mass fractions for air at given T (K) and p (atm).
// Based on curve fits of NASA CEA data for T=2000-6000K, p=1atm.
// For reference/validation only — will be replaced by actual chemistry solver in C.4.
inline void park5_equilibrium(double T, double p_atm,
    double& Y_N2, double& Y_O2, double& Y_N, double& Y_O, double& Y_NO)
{
    // Simplified equilibrium: O2 dissociates first (Ea ~ 495 kJ/mol)
    // N2 dissociation starts above ~5000K (Ea ~ 945 kJ/mol)
    double T_kK = T / 1000.0;

    // O2 fraction (sigmoid from 0.23 at low T to ~0 at high T)
    double f_O2 = 0.23 / (1.0 + std::exp(1.5 * (T_kK - 3.5)));

    // O fraction (inverse sigmoid)
    double f_O = 0.23 / (1.0 + std::exp(-1.5 * (T_kK - 4.0)));

    // N2 fraction (starts at 0.77, drops above 5000K)
    double f_N2 = 0.77 / (1.0 + std::exp(2.0 * (T_kK - 6.0)));

    // N fraction
    double f_N = 0.77 / (1.0 + std::exp(-2.0 * (T_kK - 7.0)));

    // NO fraction (peaks around 3000K)
    double f_NO = 0.08 * std::exp(-0.5 * std::pow(T_kK - 3.0, 2.0));

    // Normalize
    double sum = f_N2 + f_O2 + f_N + f_O + f_NO;
    Y_N2 = f_N2 / sum;
    Y_O2 = f_O2 / sum;
    Y_N  = f_N  / sum;
    Y_O  = f_O  / sum;
    Y_NO = f_NO / sum;
}

inline std::vector<NormalShockPoint> normal_shock_ref() {
    std::vector<double> M_vals = {5.0, 10.0, 15.0, 20.0, 25.0};
    std::vector<NormalShockPoint> data;
    for (double M : M_vals) {
        NormalShockPoint p;
        p.M1 = M;
        p.p_ratio = shock_p_ratio(M);
        p.rho_ratio = shock_rho_ratio(M);
        p.T_ratio = shock_T_ratio(M);
        p.M2 = shock_M2(M);
        // Post-shock temperature for T_ratio, assuming T1=300K
        double T2 = p.T_ratio * 300.0;
        park5_equilibrium(T2, 1.0, p.Y_N2, p.Y_O2, p.Y_N, p.Y_O, p.Y_NO);
        data.push_back(p);
    }
    return data;
}

} // namespace RefData
} // namespace AeroSim
