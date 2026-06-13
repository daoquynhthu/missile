#pragma once
#include <cmath>
#include <vector>

namespace AeroSim {
namespace RefData {

// Fay-Riddell stagnation point heat flux for a sphere in hypersonic flow.
//   q_s = 0.76 * Pr^{-0.6} * (rho_w*mu_w)^{0.1} * (rho_inf*mu_inf)^{0.4}
//         * sqrt((dp/dx)_s) * (h_s - h_w)
//
// (dp/dx)_s = sqrt(2*rho_inf / R_nose) * (1 - p_inf/p_s) * p_s
// Simplified: q_s = K * sqrt(rho_inf/R_nose) * V_inf^3 * (1 - h_w/h_s)

struct SphereHeatFluxPoint {
    double M;              // Freestream Mach
    double altitude_km;    // Altitude
    double rho_inf;        // kg/m^3
    double T_inf;          // K
    double q_stag_MW;      // Stagnation heat flux (MW/m^2)
    double T_w;            // Wall temperature (K)
};

inline double fay_riddell_q(double rho_inf, double V_inf,
    double R_nose, double T_w, double T_inf)
{
    // Simplified Fay-Riddell for engineering estimates
    // q_s = C * sqrt(rho_inf/R_nose) * V_inf^3 * (1 - h_w/h_s)
    // C ≈ 1.83e-8 for air (SI units)
    const double C = 1.83e-8;
    double h_s = 1.0e6 * V_inf * V_inf / 2.0;  // Stagnation enthalpy (J/kg)
    double Cp = 1005.0;  // J/(kg·K)
    double h_w = Cp * T_w;
    double q = C * std::sqrt(rho_inf / R_nose) * std::pow(V_inf, 3.0)
             * (1.0 - h_w / h_s);
    return q;  // W/m^2
}

inline double us76_density(double alt_km) {
    // US Standard Atmosphere 1976, simplified
    if (alt_km < 25.0) return 1.225 * std::exp(-alt_km / 8.5);
    if (alt_km < 50.0) return 1.225 * std::exp(-25.0/8.5) * std::exp(-(alt_km-25.0)/7.0);
    return 1.225 * std::exp(-25.0/8.5) * std::exp(-25.0/7.0) * std::exp(-(alt_km-50.0)/6.5);
}

inline double us76_temperature(double alt_km) {
    if (alt_km < 11.0) return 288.15 - 6.5 * alt_km;
    if (alt_km < 20.0) return 216.65;
    if (alt_km < 32.0) return 216.65 + (alt_km - 20.0);
    if (alt_km < 47.0) return 228.65 + 1.5 * (alt_km - 32.0);
    return 270.65;
}

inline double speed_of_sound(double T) {
    return std::sqrt(1.4 * 287.058 * T);
}

inline std::vector<SphereHeatFluxPoint> sphere_heatflux_ref(
    double R_nose = 0.1, double T_w = 1500.0)
{
    std::vector<SphereHeatFluxPoint> data;
    // Typical reentry trajectory points
    struct { double M; double alt_km; } points[] = {
        {5.0,  50.0},
        {10.0, 60.0},
        {15.0, 65.0},
        {20.0, 70.0},
        {25.0, 75.0},
    };
    for (auto pt : points) {
        double rho = us76_density(pt.alt_km);
        double T = us76_temperature(pt.alt_km);
        double a = speed_of_sound(T);
        double V = pt.M * a;
        double q = fay_riddell_q(rho, V, R_nose, T_w, T);
        data.push_back({pt.M, pt.alt_km, rho, T, q * 1e-6, T_w});
    }
    return data;
}

} // namespace RefData
} // namespace AeroSim
