#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include "utils.hpp"

namespace AeroSim {

struct AeroCoeffs {
    double Cd; // Drag Coefficient
    double Cl; // Lift Coefficient
    double Cm; // Pitching Moment Coefficient
    double Xcp; // Center of Pressure (from nose, in meters)
};

/**
 * @brief Aerodynamic Model with Mach-dependent coefficients
 * Simplified model assuming symmetric missile (axi-symmetric).
 * Coefficients depend on Mach number and Total Angle of Attack (alpha_total).
 */
class AerodynamicsModel {
public:
    struct Config {
        double ref_area;    // Reference Area (m^2)
        double ref_length;  // Reference Length (m)
        
        // Control Derivatives (per radian)
        double cm_delta_pitch = -0.8; // Pitch moment due to elevator
        double cn_delta_yaw = -0.5;   // Yaw moment due to rudder
        double cl_delta_roll = 0.5;   // Roll moment due to aileron

        // 1D Tables (Mach dependent)
        std::vector<double> mach_grid;
        std::vector<double> cd0_table;
        std::vector<double> cl_alpha_table;
        std::vector<double> xcp_table;
        
        // 2D Tables (Mach, Alpha) - Optional high fidelity
        std::vector<double> alpha_grid;
        std::vector<double> cl_table_2d; // Flattened [mach][alpha]
        std::vector<double> cd_table_2d;
        std::vector<double> cm_table_2d;
    };

    AerodynamicsModel(const Config& config) : m_config(config) {
        // Validation or default fallback could go here
    }

    /**
     * @brief Compute aerodynamic coefficients
     * @param mach Mach number
     * @param alpha Total angle of attack (rad)
     * @return AeroCoeffs struct
     */
    AeroCoeffs compute_coeffs(double mach, double alpha) const {
        // Clamp inputs
        mach = std::max(0.0, mach);
        double alpha_deg = alpha * 180.0 / 3.14159265359;

        double cd = 0.0;
        double cl = 0.0;
        double cm = 0.0;
        double xcp_ratio = 0.0;

        // Check if we have 2D tables
        if (!m_config.alpha_grid.empty() && !m_config.cl_table_2d.empty()) {
            // High Fidelity 2D Interpolation
            cl = Utils::interpolate_2d(m_config.mach_grid, m_config.alpha_grid, m_config.cl_table_2d, mach, alpha_deg);
            cd = Utils::interpolate_2d(m_config.mach_grid, m_config.alpha_grid, m_config.cd_table_2d, mach, alpha_deg);
            
            if (!m_config.cm_table_2d.empty()) {
                cm = Utils::interpolate_2d(m_config.mach_grid, m_config.alpha_grid, m_config.cm_table_2d, mach, alpha_deg);
            }
            
            // Xcp handling for 2D? 
            // Usually Xcp is derived from Cm and Cl/Cn.
            // Xcp = Xcg - (Cm / Cn) * RefLength
            // If we have Cm table, we can compute moment directly.
            // But this function returns Xcp for the `compute_forces_moments` to use.
            // compute_forces_moments calculates Moment = Force * (Xcp - Xcg).
            // So Force * (Xcp - Xcg) = Moment_aero.
            // Xcp - Xcg = Moment / Force.
            // Xcp = Xcg + Moment / Force.
            // If we output Cm, we should use it directly.
            // Let's modify `AeroCoeffs` to include Cm and let `compute_forces_moments` use it?
            // `AeroCoeffs` already has `Cm`.
            // But `compute_forces_moments` currently ignores `Cm` and calculates moment from Xcp.
            
            // Let's stick to Xcp 1D table for now if 2D Cm is missing.
             xcp_ratio = Utils::interpolate_1d(m_config.mach_grid, m_config.xcp_table, mach);
             
        } else {
            // Fallback to 1D Model
            double cd0 = Utils::interpolate_1d(m_config.mach_grid, m_config.cd0_table, mach);
            double cl_alpha = Utils::interpolate_1d(m_config.mach_grid, m_config.cl_alpha_table, mach);
            
            // Component buildup
            cl = cl_alpha * std::sin(2.0 * alpha); // Simplified stall
            cd = cd0 + 1.2 * std::sin(alpha) * std::sin(alpha); // Robust drag model
            
            xcp_ratio = Utils::interpolate_1d(m_config.mach_grid, m_config.xcp_table, mach);
        }

        // Ensure non-negative drag
        if (cd < 0.0) cd = 0.0; 

        // Xcp in meters
        // If xcp_table contains ratio (as implied by previous code), multiply by ref_length.
        // Previous code: xcp_table contained meters (3.5, 5.5 etc).
        // Let's assume the table values are in meters from nose.
        double xcp = xcp_ratio; 
        // If xcp_ratio is actually ratio (0.35, 0.55), multiply by length.
        // But the previous values were > 1.0 (3.5), so they are meters.
        // Wait, previous code:
        // m_xcp_table = {3.5, 3.8, ...}; // In meters.
        // xcp = xcp_ratio * m_config.ref_length; 
        // Wait, previous code multiplied by ref_length?
        // Let's check previous code.
        // "double xcp = xcp_ratio * m_config.ref_length;"
        // And "Assuming ref_length is body length? ... Let's assume Config::ref_length is Diameter (standard) ... Let's just use Xcp in meters in the table".
        // So the previous code was confused.
        // I will standardize: xcp_table contains Xcp in METERS from nose.
        // So xcp = xcp_ratio (renamed to xcp_val).
        
        return {cd, cl, cm, xcp};
    }

    /**
     * @brief Compute forces and moments in Body Frame
     * @param dynamic_pressure q (Pa)
     * @param mach Mach number
     * @param alpha Angle of attack (rad)
     * @param beta Sideslip angle (rad)
     * @param com Center of Mass (m from nose)
     * @return pair<Force, Moment>
     */
    std::pair<Eigen::Vector3d, Eigen::Vector3d> compute_forces_moments(
        double dynamic_pressure,
        double mach,
        double alpha,
        double beta,
        const Eigen::Vector3d& com,
        double control_pitch = 0.0,
        double control_yaw = 0.0,
        double control_roll = 0.0
    ) const {
        // Total angle of attack
        double alpha_total = std::sqrt(alpha*alpha + beta*beta);
        
        AeroCoeffs coeffs = compute_coeffs(mach, alpha_total);
        
        double Q_S = dynamic_pressure * m_config.ref_area;
        
        // Forces in Wind Frame
        double drag = coeffs.Cd * Q_S;
        double lift = coeffs.Cl * Q_S;
        
        // Convert Wind to Body
        double ca = coeffs.Cd * std::cos(alpha_total) - coeffs.Cl * std::sin(alpha_total);
        double cn = coeffs.Cd * std::sin(alpha_total) + coeffs.Cl * std::cos(alpha_total);
        
        // Decompose CN into Body Y and Z
        double cn_y = 0.0;
        double cn_z = 0.0;
        
        if (alpha_total > 1e-6) {
            cn_y = -cn * (beta / alpha_total); // Side force (approx)
            cn_z = -cn * (alpha / alpha_total); // Normal force (down is positive Z, so negative Lift)
        }
        
        Eigen::Vector3d force_body(-ca * Q_S, cn_y * Q_S, cn_z * Q_S);
        
        // Moments due to Aerodynamic Center offset
        // Xcp is distance from nose (positive).
        // Body Frame: Nose at Origin, X Forward? No, usually X Back is positive in some conventions.
        // But here we used x_cp_body = -coeffs.Xcp, implying X Forward is positive (Nose=0, Tail=-L).
        // This is consistent with Standard Aerospace Body Frame (X Forward).
        // So Body Points are X < 0? No, if Nose is 0 and Tail is -L.
        // Wait. Standard Aerospace: X Forward (Nose). Tail is at -L?
        // No, usually Nose is +X_something or Origin is CG.
        // If Origin is Nose, and X is Forward, then Tail is at -L. Correct.
        // If Xcp is "distance from nose", it is a positive scalar.
        // So its coordinate is -Xcp.
        double x_cp_body = -coeffs.Xcp; 
        
        // Lever arm from CG to CP
        Eigen::Vector3d r_arm(x_cp_body - com.x(), -com.y(), -com.z());
        
        // Moment = r x F
        Eigen::Vector3d moment_body = r_arm.cross(force_body);
        
        // Add Control Moments
        // Moment = Coeff * Q * S * L
        // Pitch Moment (My)
        double pitch_moment_ctrl = m_config.cm_delta_pitch * control_pitch * Q_S * m_config.ref_length;
        moment_body.y() += pitch_moment_ctrl;
        
        // Yaw Moment (Mz)
        double yaw_moment_ctrl = m_config.cn_delta_yaw * control_yaw * Q_S * m_config.ref_length;
        moment_body.z() += yaw_moment_ctrl;
        
        // Roll Moment (Mx)
        double roll_moment_ctrl = m_config.cl_delta_roll * control_roll * Q_S * m_config.ref_length;
        moment_body.x() += roll_moment_ctrl;
        
        return {force_body, moment_body};
    }

private:
    Config m_config;
};

} // namespace AeroSim
