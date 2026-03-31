#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "common.hpp"

namespace AeroSim {
namespace RM {

/**
 * @brief Subsonic Aerodynamic Model for RM Dart
 * High-fidelity: includes lift, drag, and moment coefficients with angle-of-attack (AoA) 
 * and side-slip angle (SSA) dependencies.
 */
class DartAeroModel {
public:
    struct AeroCoeffs {
        double CL;  // Lift
        double CD;  // Drag
        double CY;  // Side force
        double Cl;  // Roll moment
        double Cm;  // Pitch moment
        double Cn;  // Yaw moment
    };

    /**
     * @brief Calculate aerodynamic coefficients based on Mach, AoA, and SSA
     * @param mach Mach number (expected < 0.3)
     * @param alpha Angle of Attack (rad)
     * @param beta Side-slip angle (rad)
     * @param pqr Angular rates (rad/s) for damping moments
     */
    static CUDA_HOST_DEVICE AeroCoeffs calculate_coeffs(double mach, double alpha, double beta, const Eigen::Vector3d& pqr) {
        AeroCoeffs coeffs;
        
        // Linear Aerodynamics (Subsonic, small angles)
        // Values are representative of a fin-stabilized projectile
        double CL0 = 0.0;
        double CL_alpha = 4.0; // 1/rad
        coeffs.CL = CL0 + CL_alpha * alpha;

        // Drag (Parabolic model: CD = CD0 + k * CL^2)
        double CD0 = 0.35; // Typical for a blunt-nose projectile
        double k = 0.1;
        coeffs.CD = CD0 + k * coeffs.CL * coeffs.CL;

        // Side Force
        double CY_beta = -4.0; // 1/rad
        coeffs.CY = CY_beta * beta;

        // Moments (Static Stability)
        // Cm_alpha < 0 for longitudinal stability
        double Cm_alpha = -0.5; // Reduced stiffness
        double Cm_q = -20.0;    // Increased damping
        coeffs.Cm = Cm_alpha * alpha + Cm_q * pqr.y();

        // Cn_beta > 0 for directional stability (weathercock)
        double Cn_beta = 0.5;   // Reduced stiffness
        double Cn_r = -20.0;    // Increased damping
        coeffs.Cn = Cn_beta * beta + Cn_r * pqr.z();

        // Roll Moment (Damping only for non-spinning dart)
        double Cl_p = -0.5;
        coeffs.Cl = Cl_p * pqr.x();

        return coeffs;
    }
};

} // namespace RM
} // namespace AeroSim
