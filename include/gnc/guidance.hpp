#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <algorithm>
#include "constants.hpp"
#include "coordinate_transform.hpp"

namespace AeroSim {
namespace GNC {

/**
 * @brief Guidance for Boost-Glide (Qian Xuesen) Trajectory
 * 
 * Phases:
 * 1. BOOST: Vertical launch, pitch over to gain horizontal velocity.
 * 2. COAST: Exo-atmospheric ballistic flight (Zero AoA to minimize drag, or hold attitude).
 * 3. GLIDE: Re-entry interface (e.g. 60-80km). Pull up to high AoA (10-15 deg) to skip/glide.
 * 4. TERMINAL: Dive to target (not fully implemented yet, just continues glide).
 */
class Guidance {
public:
    enum class Phase {
        BOOST,
        COAST,
        GLIDE,
        TERMINAL
    };

    struct Config {
        // Boost Phase
        double boost_end_time = 35.0;     // Time when boost phase ends (s)
        double boost_pitch_start = 2.0;   // Time to start pitch maneuver (s)
        double boost_pitch_rate = 3.0;    // Pitch rate during boost (deg/s)
        double boost_pitch_min = 0.0;     // Minimum pitch angle (deg)

        // Coast/Glide Transition
        double glide_alt_start = 50000.0; // Altitude to start glide phase (m) - Re-entry interface
        double glide_vel_min = 800.0;     // Minimum velocity to sustain glide (m/s)
        double hysteresis_margin = 5000.0; // Altitude margin for phase switching (m)

        // Glide Phase (QEGG)
        double glide_alt_end = 20000.0;    // Target altitude at end of glide (m)
        double target_range = 2200000.0;   // Target range for simple impact calculation (m)
        double glide_aoa_bias = 10.0;      // Nominal AoA (deg)
        double glide_aoa_max = 20.0;       // Max AoA (deg)
        double glide_aoa_min = 0.0;        // Min AoA (deg)
        double kp_alt = 0.08;              // Altitude error gain -> Target Vz
        double kp_vz = 0.2;                // Vertical velocity error gain -> AoA
        double max_climb_rate = 500.0;     // Max climb rate (m/s)
        double max_descent_rate = -300.0;  // Max descent rate (m/s)

        // Lateral Guidance
        double lateral_gain = 2.5;         // Gain for lateral guidance (Heading Error -> Bank Angle)
        double max_bank_angle = 60.0;      // Max bank angle (deg)

        // Terminal Phase
        double terminal_g_limit = 5.0;     // Max G-load in terminal phase
    };

    Guidance(const Config& config) 
        : m_config(config), m_phase(Phase::BOOST), m_target_ecef(0,0,0) {}

    void set_target(const Eigen::Vector3d& target_ecef) {
        m_target_ecef = target_ecef;
    }

    Phase get_phase() const { return m_phase; }

    /**
     * @brief Update Guidance Law
     * @param t Current time (s)
     * @param pos_ecef Current Position (ECEF)
     * @param vel_ecef Current Velocity (ECEF)
     * @return Target Attitude Quaternion (Body to ECEF)
     */
    Eigen::Quaterniond update(double t, const Eigen::Vector3d& pos_ecef, const Eigen::Vector3d& vel_ecef) {
        // 0. Basic State Info
        LLA current_lla = CoordinateTransform::ecef_to_lla(pos_ecef);
        double alt = current_lla.alt;
        double vel_mag = vel_ecef.norm();
        
        // Calculate Geodetic Normal (Up) - More accurate than Geocentric Radial
        // n_x = cos(lat) * cos(lon)
        // n_y = cos(lat) * sin(lon)
        // n_z = sin(lat)
        double lat = current_lla.lat;
        double lon = current_lla.lon;
        Eigen::Vector3d up(std::cos(lat)*std::cos(lon), std::cos(lat)*std::sin(lon), std::sin(lat));
        
        double vertical_vel = vel_ecef.dot(up);
        Eigen::Vector3d v_dir;
        if (vel_mag < 1e-3) {
            v_dir = up; // Default to Up if stationary to avoid NaN
        } else {
            v_dir = vel_ecef.normalized();
        }

        // 1. Determine Phase with Hysteresis
        update_phase(t, alt, vel_mag, vertical_vel);

        // 2. Compute Desired Attitude based on Phase
        Eigen::Quaterniond target_quat = Eigen::Quaterniond::Identity();

        // Common Vector Calculations
        Eigen::Vector3d east = Eigen::Vector3d::UnitY(); // Approximate if needed, but we calculate local frame below
        
        // Local North/East
        Eigen::Vector3d earth_z(0,0,1);
        Eigen::Vector3d local_north = (earth_z - (earth_z.dot(up) * up)).normalized();
        Eigen::Vector3d local_east = local_north.cross(up).normalized();

        switch (m_phase) {
            case Phase::BOOST: {
                // Open Loop Pitch Program
                Eigen::Vector3d target_dir;

                if (t < m_config.boost_pitch_start) {
                    target_dir = up; // Vertical Rise
                } else {
                    double time_since_pitch = t - m_config.boost_pitch_start;
                    double pitch_rate = m_config.boost_pitch_rate; 
                    double target_pitch_deg = 90.0 - pitch_rate * time_since_pitch;
                    
                    // Debug print for pitch logic diagnosis (Run once per second approx)
                    static int last_print_t = -1;
                    if ((int)t > last_print_t) {
                        std::cout << "[Guidance] T=" << t 
                                  << " Start=" << m_config.boost_pitch_start 
                                  << " Rate=" << pitch_rate 
                                  << " TargetPitch=" << target_pitch_deg << std::endl;
                        last_print_t = (int)t;
                    }

                    if (target_pitch_deg < m_config.boost_pitch_min) target_pitch_deg = m_config.boost_pitch_min; 

                    
                    double target_angle_from_vertical = (90.0 - target_pitch_deg) * AeroSim::Math::DEG2RAD();
                    
                    // Rotate Up vector towards East (Launch Azimuth 90)
                    // TODO: Make Launch Azimuth configurable? Assumed East for now.
                    target_dir = std::cos(target_angle_from_vertical) * up + std::sin(target_angle_from_vertical) * local_east;
                    target_dir.normalize();
                }
                
                // Construct Body Frame (Wings Level)
                Eigen::Vector3d body_x = target_dir;
                Eigen::Vector3d body_y;
                
                if (std::abs(body_x.dot(up)) > 0.999) {
                    // Vertical flight: Align Y with South
                    body_y = -local_north; 
                } else {
                    // Wings Level
                    body_y = body_x.cross(up).normalized();
                }
                
                Eigen::Vector3d body_z = body_x.cross(body_y).normalized();
                
                Eigen::Matrix3d rot_mat;
                rot_mat.col(0) = body_x;
                rot_mat.col(1) = body_y;
                rot_mat.col(2) = body_z;
                target_quat = Eigen::Quaterniond(rot_mat);
                break;
            }
            case Phase::COAST: {
                // Exo-atmospheric / Re-entry Preparation
                // Goal: Align for re-entry with positive AoA to generate lift (Skip).
                
                // Calculate Flight Path Angle (Gamma)
                double gamma_rad = std::asin(std::clamp(vertical_vel / vel_mag, -1.0, 1.0));
                
                // Desired Re-entry AoA (e.g. 20-40 degrees for high drag/lift)
                // If ascending, keep AoA low (0) to minimize drag.
                
                // Construct Target Frame based on Velocity Vector
                // Safe Normalization
                // Fix: Project UNIT vector v_dir, not velocity magnitude
                Eigen::Vector3d v_h_raw = v_dir - v_dir.dot(up) * up;
                Eigen::Vector3d v_horiz;
                if (v_h_raw.norm() < 1e-3) {
                    v_horiz = local_east;
                } else {
                    v_horiz = v_h_raw.normalized();
                }

                // Desired Re-entry AoA
                // Only pitch up if we are descending AND approaching atmosphere (< 80km)
                // This prevents discontinuity at Apogee (where vertical_vel crosses 0)
                double desired_aoa_deg = 0.0;
                if (vertical_vel < 0 && alt < 80000.0) {
                     desired_aoa_deg = 20.0; 
                }
                
                double desired_pitch_rad = gamma_rad + desired_aoa_deg * AeroSim::Math::DEG2RAD();
                
                // Target Direction (Nose)
                // Pitch relative to Horizon (Local Horizontal)
                // Target = cos(pitch)*Horiz + sin(pitch)*Up
                Eigen::Vector3d target_dir = std::cos(desired_pitch_rad) * v_horiz + std::sin(desired_pitch_rad) * up;
                target_dir.normalize();
                
                Eigen::Vector3d body_x = target_dir;
                Eigen::Vector3d body_y = body_x.cross(up).normalized();
                if (body_x.cross(up).norm() < 1e-3) body_y = local_east;
                Eigen::Vector3d body_z = body_x.cross(body_y).normalized();
                
                Eigen::Matrix3d rot_mat;
                rot_mat.col(0) = body_x;
                rot_mat.col(1) = body_y;
                rot_mat.col(2) = body_z;
                target_quat = Eigen::Quaterniond(rot_mat);
                break;
            }
            case Phase::GLIDE: {
                // Quasi-Equilibrium Glide Guidance (QEGG) with Lateral Steering (Bank-to-Turn)
                
                // A. Energy Management: Calculate Target Altitude based on Range-to-Go
                // Estimate Range-to-Go (Great Circle)
                // Project current and target to unit sphere (ignoring altitude for angle)
                // Use Haversine or Dot product on normalized ECEF
                Eigen::Vector3d p_unit = pos_ecef.normalized();
                Eigen::Vector3d t_unit = m_target_ecef.normalized();
                double central_angle = std::acos(std::clamp(p_unit.dot(t_unit), -1.0, 1.0));
                double range_to_go = central_angle * 6371000.0; // Earth Radius approx
                
                // Simple Linear Profile for Reference Altitude
                // H_ref = H_end + (R / R_total) * (H_start - H_end)
                // Note: R_total is roughly initial range. Let's use current range as R.
                // Or better: Quadratic profile to maintain energy?
                // Let's stick to linear for now as requested "Range-Altitude Profile"
                double dist_ratio = range_to_go / m_config.target_range;
                if (dist_ratio > 1.0) dist_ratio = 1.0;
                double target_alt = m_config.glide_alt_end + dist_ratio * (m_config.glide_alt_start - m_config.glide_alt_end);
                
                // B. Vertical Guidance (QEGG) -> Desired Vertical Force -> AoA
                double h_error = target_alt - alt;
                double v_z_target = m_config.kp_alt * h_error;
                v_z_target = std::clamp(v_z_target, m_config.max_descent_rate, m_config.max_climb_rate);
                
                double v_z_err = v_z_target - vertical_vel;
                double aoa_deg = m_config.glide_aoa_bias + m_config.kp_vz * v_z_err;
                aoa_deg = std::clamp(aoa_deg, m_config.glide_aoa_min, m_config.glide_aoa_max);
                double aoa = aoa_deg * AeroSim::Math::DEG2RAD();
                
                // C. Lateral Guidance -> Heading Error -> Bank Angle
                // Calculate Desired Heading (Azimuth to Target)
                // Current Heading vector (Horizontal projection of velocity)
                Eigen::Vector3d v_h_raw_glide = v_dir - v_dir.dot(up) * up;
                Eigen::Vector3d v_horiz;
                if (v_h_raw_glide.norm() < 1e-3) {
                    v_horiz = local_east;
                } else {
                    v_horiz = v_h_raw_glide.normalized();
                }

                // Target Heading vector (Horizontal projection of Vector-to-Target)
                // Vector to Target in ECEF
                Eigen::Vector3d to_target = (m_target_ecef - pos_ecef).normalized();
                Eigen::Vector3d to_target_h_raw = to_target - to_target.dot(up) * up;
                Eigen::Vector3d to_target_horiz;
                if (to_target_h_raw.norm() < 1e-3) {
                    to_target_horiz = local_east;
                } else {
                    to_target_horiz = to_target_h_raw.normalized();
                }
                
                // Heading Error (Cross product Z component in local frame)
                // Cross(v_horiz, to_target_horiz) dot Up
                double heading_err_sin = v_horiz.cross(to_target_horiz).dot(up);
                // Clamp sin just in case
                heading_err_sin = std::clamp(heading_err_sin, -1.0, 1.0);
                double heading_err = std::asin(heading_err_sin); // Radians
                
                // Bank Command: Proportional to Heading Error
                // Positive Error (Target is Left) -> Bank Left (Negative Roll)
                // Wait, standard body frame: Roll Right is positive.
                // If Target is Left, we need to turn Left. Bank Left (Negative Roll).
                // Check Cross Product: v x t. If t is left of v, v x t is Up?
                // Example: v = North, t = West. N x W = Up? No, N x W = -Down?
                // N(0,1,0) x W(-1,0,0) = (0,0,1) = Up.
                // So positive Cross means Target is Left.
                // We want Negative Roll.
                // So phi = -K * error.
                double bank_angle = -m_config.lateral_gain * heading_err * AeroSim::Math::RAD2DEG();
                bank_angle = std::clamp(bank_angle, -m_config.max_bank_angle, m_config.max_bank_angle);
                double bank_rad = bank_angle * AeroSim::Math::DEG2RAD();
                
                // D. Construct Target Attitude (Bank-to-Turn)
                // 1. Start with Velocity Frame (X=v_dir)
                // 2. Apply Bank (Rotate Y/Z around X)
                // 3. Apply Pitch (Rotate X/Z around Y)
                
                // Reference Horizontal Right (Wings Level)
                Eigen::Vector3d horizontal_right = v_dir.cross(up).normalized();
                if (v_dir.cross(up).norm() < 1e-3) horizontal_right = local_east;
                
                // Apply Bank: Rotate "Right" vector by bank angle around Velocity
                // New Right = AngleAxis(bank, v_dir) * horizontal_right
                Eigen::Vector3d banked_right = Eigen::AngleAxisd(bank_rad, v_dir) * horizontal_right;
                
                // Apply Pitch: Rotate Velocity vector "Up" (around banked_right) by AoA
                // Target X = AngleAxis(-aoa, banked_right) * v_dir ? 
                // Pitch Up is positive rotation around Right axis (Y).
                // So Target X is rotated "Up" relative to v_dir.
                // In Body Frame, v_dir is incoming, so X points into wind? No.
                // Body X is Nose. v_dir is Flight Path.
                // Nose is pitched up by AoA relative to v_dir.
                // Rotation is around Right Axis.
                Eigen::Vector3d target_nose = Eigen::AngleAxisd(aoa, banked_right) * v_dir;
                
                Eigen::Vector3d body_x = target_nose;
                Eigen::Vector3d body_y = banked_right;
                Eigen::Vector3d body_z = body_x.cross(body_y).normalized();
                
                Eigen::Matrix3d rot_mat;
                rot_mat.col(0) = body_x;
                rot_mat.col(1) = body_y;
                rot_mat.col(2) = body_z;
                target_quat = Eigen::Quaterniond(rot_mat);
                break;
            }
            case Phase::TERMINAL: {
                // Pure Pursuit with G-Limiting
                if (m_target_ecef.norm() > 1.0) {
                    Eigen::Vector3d to_target = (m_target_ecef - pos_ecef).normalized();
                    
                    // G-Limiting: Don't turn too hard
                    // Compare current velocity vector with to_target
                    // Calculate angle difference
                    double angle_diff = std::acos(std::clamp(v_dir.dot(to_target), -1.0, 1.0));
                    
                    // Max turn rate = a_max / V
                    // We can't easily limit rate here without state, but we can limit the "Target Point" deviation
                    // For now, Pure Pursuit is fine, Autopilot limits G via max fin deflection usually.
                    // But we can limit the Angle of Attack implied by the turn?
                    // Let's just point at target.
                    
                    Eigen::Vector3d body_x = to_target;
                    Eigen::Vector3d body_y;
                    if (body_x.cross(up).norm() < 1e-3) {
                        body_y = local_east;
                    } else {
                        body_y = body_x.cross(up).normalized();
                    }
                    Eigen::Vector3d body_z = body_x.cross(body_y).normalized();
                    
                    Eigen::Matrix3d rot_mat;
                    rot_mat.col(0) = body_x;
                    rot_mat.col(1) = body_y;
                    rot_mat.col(2) = body_z;
                    target_quat = Eigen::Quaterniond(rot_mat);
                } else {
                    target_quat = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), v_dir);
                }
                break;
            }
        }

        return target_quat;
    }

private:
    Config m_config;
    Phase m_phase;
    Eigen::Vector3d m_target_ecef;

    void update_phase(double t, double alt, double vel_mag, double vertical_vel) {
        if (t < m_config.boost_end_time) {
            m_phase = Phase::BOOST;
            return;
        }

        // Hysteresis Logic
        // Use margin to prevent flickering
        double margin = m_config.hysteresis_margin;

        if (m_phase == Phase::BOOST) {
            // End of boost.
            // If altitude is high OR we are ascending, go to COAST.
            // Only if we are low AND descending do we go straight to GLIDE.
            // Note: vertical_vel > -10.0 to handle near-apogee cases safely.
            if (alt > m_config.glide_alt_start || vertical_vel > -10.0) {
                m_phase = Phase::COAST;
            } else {
                m_phase = Phase::GLIDE; 
            }
        } else if (m_phase == Phase::COAST) {
            // Switch to GLIDE if we fall below interface AND are descending
            if (alt < m_config.glide_alt_start - margin && vertical_vel < 0) {
                m_phase = Phase::GLIDE;
            }
        } else if (m_phase == Phase::GLIDE) {
            // Switch to TERMINAL if slow or very low
            // Assuming terminal dive at end of glide or if energy depleted
            if (vel_mag < m_config.glide_vel_min || alt < 5000.0) { // Hardcoded 5km terminal floor
                m_phase = Phase::TERMINAL;
            }
            // Could switch back to COAST if we skip up very high?
            // Yes, Qian Xuesen trajectory skips.
            if (alt > m_config.glide_alt_start + margin && vertical_vel > 0) {
                m_phase = Phase::COAST;
            }
        }
        // TERMINAL is final
    }
};

} // namespace GNC
} // namespace AeroSim
