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
        double glide_aoa_bias = 12.0;      // Nominal AoA (deg) - Increased for better L/D
        double glide_aoa_max = 25.0;       // Max AoA (deg) - Increased for aggressive skip
        double glide_aoa_min = 0.0;        // Min AoA (deg)
        double kp_alt = 0.08;              // Altitude error gain -> Target Vz
        double kp_vz = 0.2;                // Vertical velocity error gain -> AoA
        double max_climb_rate = 500.0;     // Max climb rate (m/s)
        double max_descent_rate = -300.0;  // Max descent rate (m/s)

        // Lateral Guidance
        double lateral_gain = 3.5;         // Gain for lateral guidance (Heading Error -> Bank Angle)
        double max_bank_angle = 60.0;      // Max bank angle (deg)

        // Terminal Phase
        double terminal_range_trigger = 50000.0; // Trigger terminal phase when range < 50km
        double terminal_g_limit = 10.0;    // Max G-load in terminal phase
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
        double range_to_target = (m_target_ecef - pos_ecef).norm();
        update_phase(t, alt, vel_mag, vertical_vel, range_to_target);

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
                    
                    // Calculate Launch Azimuth to Target
                    // Vector from Pos to Target (Projected to Horizontal Plane)
                    Eigen::Vector3d to_target = (m_target_ecef - pos_ecef).normalized();
                    Eigen::Vector3d to_target_horiz = (to_target - to_target.dot(up) * up).normalized();
                    
                    // Use calculated azimuth direction instead of hardcoded East
                    target_dir = std::cos(target_angle_from_vertical) * up + std::sin(target_angle_from_vertical) * to_target_horiz;
                    target_dir.normalize();
                }
                
                // Construct Body Frame (Wings Level)
                // We want Body Z to be "Up" relative to the horizon (or consistent with launch)
                // If we pitch over, Body X follows target_dir.
                // We want wings level relative to the ground.
                // Body Y = Right Wing. Body Z = Belly.
                // If we are flying "Belly Down" (Body Z down), then Body Y is horizontal.
                // Cross(target_dir, Up) gives Right Wing (Body Y) if target_dir is forward and Up is Up?
                // Forward x Up = Right? No.
                // North x Up = East.
                // So Cross(target_dir, Up) is Right Wing.
                
                Eigen::Vector3d body_x = target_dir;
                Eigen::Vector3d body_y;
                if (body_x.cross(up).norm() < 1e-3) {
                     // Vertical. Use Launch Attitude (East is Z, South is Y)
                     // If X is Up.
                     // We want to maintain roll stability.
                     // Let's match the initial condition or just pick East.
                     // Initial state: Body Y was South.
                     body_y = local_north * -1.0; 
                } else {
                     // Wings Level: Body Y is horizontal (perp to Up and Forward)
                     // We want Body Y to be "Right".
                     // Let's test: North x Up = East. Correct.
                     body_y = body_x.cross(up).normalized();
                }
                
                // Body Z completes the triad (Down-ish)
                Eigen::Vector3d body_z = body_x.cross(body_y).normalized();
                
                Eigen::Matrix3d rot_mat;
                rot_mat.col(0) = body_x;
                rot_mat.col(1) = body_y;
                rot_mat.col(2) = body_z;
                target_quat = Eigen::Quaterniond(rot_mat);
            }
            break;
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
                Eigen::Vector3d p_unit = pos_ecef.normalized();
                Eigen::Vector3d t_unit = m_target_ecef.normalized();
                double central_angle = std::acos(std::clamp(p_unit.dot(t_unit), -1.0, 1.0));
                double range_to_go = central_angle * 6371000.0; // Earth Radius approx
                
                // Use a non-linear profile to encourage skip-glide
                // H_ref = H_end + (H_start - H_end) * (R / R_total)^n
                // n > 1 means we stay high longer and dive later?
                // n < 1 means we drop fast and glide low?
                // Qian Xuesen: Dive deep first, then pull up.
                // The current "Linear" profile might be too shallow.
                // Let's use a standard linear profile for now but with adjusted constants.
                
                double dist_ratio = range_to_go / m_config.target_range;
                if (dist_ratio > 1.0) dist_ratio = 1.0;
                double target_alt = m_config.glide_alt_end + dist_ratio * (m_config.glide_alt_start - m_config.glide_alt_end);
                
                // Altitude Error Gain Schedule
                // Increase gain as we get closer?
                // Or just use fixed gain.
                // Current: kp_alt = 0.08
                
                double h_error = target_alt - alt;
                double v_z_target = m_config.kp_alt * h_error;
                v_z_target = std::clamp(v_z_target, m_config.max_descent_rate, m_config.max_climb_rate);
                
                double v_z_err = v_z_target - vertical_vel;
                
                // Damping Term? Damping is provided by velocity loop (v_z_err).
                // AoA Command
                double aoa_deg = m_config.glide_aoa_bias + m_config.kp_vz * v_z_err;
                
                // Add "Skip" logic: If Descent Rate is high at low altitude, pull max AoA
                if (alt < m_config.glide_alt_start - 2000.0 && vertical_vel < -50.0) {
                     // Pull up hard to skip
                     aoa_deg += 5.0; 
                }
                
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
                // Proportional Navigation (PN) Implementation
                // Command Acceleration: a_cmd = N * V_closing * Omega_los
                
                if (m_target_ecef.norm() > 1.0) {
                    Eigen::Vector3d to_target = (m_target_ecef - pos_ecef);
                    double dist = to_target.norm();
                    Eigen::Vector3d los_vec = to_target.normalized();
                    
                    // Relative Velocity (assuming stationary target)
                    // v_rel = v_target - v_missile = -v_missile
                    Eigen::Vector3d v_rel = -vel_ecef;
                    double v_closing = v_rel.dot(los_vec); // Should be positive if closing
                    
                    // LOS Rate (Omega)
                    // Omega = (r x v_rel) / |r|^2 ???
                    // Standard Definition: Omega = (LOS x V_rel) / Range ???
                    // No. Omega = (r x v) / r^2 where r is relative position vector.
                    // Omega = (to_target x v_rel) / (dist * dist)
                    Eigen::Vector3d omega = to_target.cross(v_rel) / (dist * dist);
                    
                    // PN Acceleration Command (Perpendicular to LOS)
                    // a_cmd = N * V_closing * Omega
                    double N_pn = 3.0; // Navigation Constant
                    Eigen::Vector3d a_cmd = N_pn * v_closing * omega.cross(los_vec); // Direction?
                    // Omega is perpendicular to Plane defined by LOS and V_rel.
                    // Omega x LOS is in the Plane, perpendicular to LOS.
                    // This gives the direction to turn.
                    
                    // Limit Acceleration (G-Limit)
                    double a_mag = a_cmd.norm();
                    double max_accel = m_config.terminal_g_limit * 9.81;
                    if (a_mag > max_accel) {
                        a_cmd = a_cmd * (max_accel / a_mag);
                    }
                    
                    // Convert Acceleration Command to Attitude Command
                    // We need to generate aerodynamic lift in the direction of a_cmd.
                    // F_aero = a_cmd * mass.
                    // Lift Direction = a_cmd direction.
                    // We need to roll the missile so that Body Pitch Plane aligns with a_cmd.
                    // And pull AoA to generate the magnitude.
                    
                    // 1. Desired Lift Vector Direction (in ECEF)
                    // We also need to fight Gravity?
                    // a_total = a_cmd - g_vector ?
                    // Gravity acts Down. To fly straight, we need Lift = Gravity.
                    // To accelerate by a_cmd, we need Lift = a_cmd + Gravity.
                    // g_vector is roughly -Up * 9.81.
                    Eigen::Vector3d gravity = -9.81 * up;
                    Eigen::Vector3d lift_needed = a_cmd - gravity; // F/m units
                    
                    // 2. Desired Body Y (Right Wing) should be perpendicular to Lift.
                    // Body Z (Spine) is usually "Up" in body frame for standard plane?
                    // For missile, usually +AoA is "Pitch Up" in Body Y axis?
                    // Let's assume Body Z is "Up" (Direction of Lift).
                    // Or Body Y is "Right", so Lift is in -Z or +Z?
                    // Standard Aero: Lift is in Body Z-X plane, perpendicular to V.
                    // Usually Lift is "Up" relative to body.
                    // Let's assume we pull Positive AoA to generate Lift in Body -Z direction?
                    // Wait, standard Body Frame: X=Nose, Y=Right, Z=Down.
                    // Positive AoA -> Lift in -Z direction (Up).
                    // So we want Body -Z aligned with lift_needed.
                    
                    Eigen::Vector3d desired_body_z = -lift_needed.normalized();
                    
                    // 3. Desired Body X (Nose) should be roughly Velocity vector
                    // But we have AoA.
                    // Body X is Velocity rotated by AoA.
                    // For now, let's just align the Roll.
                    // Construct Rotation:
                    // X_temp = v_dir
                    // Z_temp = desired_body_z projected to be perp to X_temp?
                    
                    Eigen::Vector3d body_x_temp = v_dir;
                    Eigen::Vector3d body_y_temp = desired_body_z.cross(body_x_temp).normalized();
                    Eigen::Vector3d body_z_final = body_x_temp.cross(body_y_temp).normalized();
                    
                    // Now we have the orientation that aligns "Body Up" (-Z) with Lift direction.
                    // We also need to pitch up by AoA.
                    // Calculate required AoA based on Lift Magnitude?
                    // Lift_Mag = lift_needed.norm();
                    // Lift = 0.5 * rho * V^2 * S * Cl_alpha * alpha
                    // alpha = Lift / (Q * S * Cl_alpha)
                    // We don't have rho/S/Cl here easily.
                    // Let's use a simpler proportional mapping or max AoA.
                    // Let's just command a pitch bias?
                    // Or better: Just point the nose at the "Lead Point"?
                    // The "Attitude Command" here defines the Reference Frame for the Autopilot.
                    // The Autopilot will try to achieve this orientation.
                    // If we set Target Orientation = Aligned with Velocity but Rolled,
                    // The Autopilot will try to zero the error.
                    // But we want to maintain an AoA.
                    // If we command an orientation that is "pitched up" relative to velocity...
                    
                    // Let's apply a fixed "Max AoA" scaling?
                    // Or just let the Autopilot handle "G-Command"?
                    // Current Autopilot tracks Attitude Quaternion.
                    // So we must output the Desired Attitude (Nose direction).
                    
                    // Desired Nose = Velocity Vector rotated by 'Alpha' towards 'Lift Direction'.
                    // Alpha = k * Lift_Mag ?
                    // Let's guess alpha. Max Lift corresponds to Max AoA (e.g. 20 deg).
                    // Max G = 10g ~ 100 m/s2.
                    // Current demand = lift_needed.norm().
                    double lift_accel = lift_needed.norm();
                    double max_lift_accel = max_accel + 9.81; // Approx
                    double fraction = lift_accel / max_lift_accel;
                    if (fraction > 1.0) fraction = 1.0;
                    
                    double commanded_aoa_deg = fraction * 20.0; // Max 20 deg AoA
                    
                    // Rotate body_x_temp (Velocity) around body_y_temp (Right Wing) by -alpha (Pitch Up)
                    // Pitch Up means Nose goes "Up" (away from Z).
                    // Rotation axis is +Y. Angle is +alpha?
                    // Body Z is Down. Pitch Up is rotation about +Y that moves X towards -Z.
                    // Right Hand Rule on +Y: X rotates towards -Z. Yes.
                    // So rotate by +alpha.
                    
                    double alpha_rad = commanded_aoa_deg * AeroSim::Math::DEG2RAD();
                    Eigen::AngleAxisd pitch_rot(-alpha_rad, body_y_temp); // Wait, standard pitch up is +theta.
                    // If we rotate vectors: New_X = Rot * Old_X.
                    // We want Nose to be "Above" Velocity.
                    // Velocity is X_temp.
                    // We want Nose to be rotated around Y by +alpha.
                    Eigen::Vector3d body_x_final = Eigen::AngleAxisd(-alpha_rad, body_y_temp) * body_x_temp; 
                    // Sign?
                    // Y points Right. X points Forward. Z points Down.
                    // Rotate X around Y by +90 -> Z. (Down). This is Pitch Down.
                    // So Pitch Up is negative rotation around Y?
                    // Let's check:
                    // R_y(theta):
                    //  c  0  s
                    //  0  1  0
                    // -s  0  c
                    // R_y(90) * [1,0,0] = [0,0,-1] = -Z (Up).
                    // So Positive Rotation around Y gives Pitch Up (Body X moves to Body -Z).
                    // So we use +alpha_rad.
                    
                    body_x_final = Eigen::AngleAxisd(alpha_rad, body_y_temp) * body_x_temp;
                    
                    // Re-orthogonalize
                    Eigen::Vector3d body_z_real = body_x_final.cross(body_y_temp).normalized();
                    Eigen::Vector3d body_y_real = body_z_real.cross(body_x_final).normalized();
                    
                    Eigen::Matrix3d rot_mat;
                    rot_mat.col(0) = body_x_final;
                    rot_mat.col(1) = body_y_real;
                    rot_mat.col(2) = body_z_real;
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

    void update_phase(double t, double alt, double vel_mag, double vertical_vel, double range_to_target) {
        if (t < m_config.boost_end_time) {
            m_phase = Phase::BOOST;
            return;
        }

        // Hysteresis Logic
        // Use margin to prevent flickering
        double margin = m_config.hysteresis_margin;

        // Terminal Phase Trigger (Distance or Altitude based)
        if (m_phase == Phase::GLIDE && range_to_target < m_config.terminal_range_trigger) {
             m_phase = Phase::TERMINAL;
             return;
        }

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
