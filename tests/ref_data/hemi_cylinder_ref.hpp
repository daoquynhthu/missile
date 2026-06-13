#pragma once
#include <cmath>
#include <vector>

namespace AeroSim {
namespace RefData {

// Modified Newtonian pressure distribution on a hemisphere-cylinder.
// Cp(theta) = Cp_max * cos^2(theta) for |theta| <= 90 deg
// Cp(theta) = 0 for |theta| > 90 deg (shadow region)
// Cp_max = (gamma+3)/(gamma+1) * (1 - 1/(gamma*M^2))

struct HemiCylinderPoint {
    double theta_deg;  // Angle from stagnation point (deg)
    double Cp;         // Pressure coefficient
    double Cp_max;     // Maximum Cp at stagnation
};

inline double cp_max_modified_newtonian(double M, double gamma = 1.4) {
    return (gamma + 3.0) / (gamma + 1.0) * (1.0 - 1.0 / (gamma * M * M));
}

inline std::vector<HemiCylinderPoint> hemi_cylinder_ref(
    double M = 10.0, double gamma = 1.4)
{
    std::vector<HemiCylinderPoint> data;
    double Cp0 = cp_max_modified_newtonian(M, gamma);
    for (int theta = 0; theta <= 180; theta += 10) {
        double t_rad = theta * 3.14159265358979323846 / 180.0;
        double Cp = (theta <= 90) ? Cp0 * std::pow(std::cos(t_rad), 2.0) : 0.0;
        data.push_back({(double)theta, Cp, Cp0});
    }
    return data;
}

} // namespace RefData
} // namespace AeroSim
