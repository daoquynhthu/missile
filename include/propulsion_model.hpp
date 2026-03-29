#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include "constants.hpp"
#include "utils.hpp"

namespace AeroSim {

struct TVCCommand {
    double pitch; // Nozzle deflection (radians)
    double yaw;   // Nozzle deflection (radians)
};

struct RCSCommand {
    double pitch; // -1.0 to 1.0
    double yaw;   // -1.0 to 1.0
    double roll;  // -1.0 to 1.0
    bool enabled;
};

/**
 * @brief Propulsion system output
 */
struct PropulsionOutput {
    Eigen::Vector3d force_body;    // Thrust force in Body frame (N)
    Eigen::Vector3d moment_body;   // Thrust moment in Body frame (N*m)
    double mass_flow_rate;         // Mass flow rate (kg/s), should be negative for mass loss
    
    // Combine outputs
    PropulsionOutput operator+(const PropulsionOutput& other) const {
        return {
            force_body + other.force_body,
            moment_body + other.moment_body,
            mass_flow_rate + other.mass_flow_rate
        };
    }
};

/**
 * @brief Reaction Control System (RCS) Model
 * Models a set of thrusters for attitude control
 */
class RCSModel {
public:
    struct Config {
        double max_thrust;      // Thrust per thruster (N)
        double isp;             // Specific Impulse (s)
        double lever_arm_x;     // Longitudinal distance from CoM (m)
        double lever_arm_r;     // Radial distance from centerline (m)
    };

    RCSModel(const Config& config) : m_config(config) {}

    PropulsionOutput compute(const RCSCommand& cmd, double current_mass, double min_mass) const {
        PropulsionOutput out = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0};
        
        if (!cmd.enabled) return out;

        // Check for fuel depletion
        if (current_mass <= min_mass) {
            return out;
        }

        // Simplified RCS Logic:
        // Assume ideal thruster pairs for pure couple generation
        
        double F = m_config.max_thrust;
        double Lx = m_config.lever_arm_x; // Arm for Pitch/Yaw (Longitudinal)
        double Lr = m_config.lever_arm_r; // Arm for Roll (Radial)
        
        // Pitch Moment (Y-axis): Thrusters firing Up/Down at Nose/Tail
        // M_pitch = F * Lx * cmd.pitch
        double m_pitch = F * Lx * std::max(-1.0, std::min(1.0, cmd.pitch));
        
        // Yaw Moment (Z-axis): Thrusters firing Left/Right at Nose/Tail
        // M_yaw = F * Lx * cmd.yaw
        double m_yaw = F * Lx * std::max(-1.0, std::min(1.0, cmd.yaw));
        
        // Roll Moment (X-axis): Thrusters firing Tangentially
        // M_roll = F * Lr * cmd.roll
        double m_roll = F * Lr * std::max(-1.0, std::min(1.0, cmd.roll));
        
        out.moment_body = Eigen::Vector3d(m_roll, m_pitch, m_yaw);
        
        // Force (Couples produce zero net force ideally, but single thrusters don't)
        // Let's assume pure couples for now (4-thruster blocks firing in opposition)
        out.force_body = Eigen::Vector3d::Zero();
        
        // Mass Flow
        // |F_total| = (|pitch| + |yaw| + |roll|) * F
        double total_thrust_active = (std::abs(cmd.pitch) + std::abs(cmd.yaw) + std::abs(cmd.roll)) * F;
        double g0 = 9.80665;
        out.mass_flow_rate = -total_thrust_active / (m_config.isp * g0);
        
        return out;
    }

private:
    Config m_config;
};

/**
 * @brief Solid Rocket Motor Model
 * Simple model with constant thrust or thrust curve
 */
class SolidMotor {
public:
    struct Config {
        // Thrust Curve (Time since ignition vs Vacuum Thrust)
        std::vector<double> time_knots;
        std::vector<double> thrust_knots; 
        
        double total_impulse;    // Total impulse (N*s) - Optional check
        double burn_time;        // Burn duration (s)
        double isp;              // Specific Impulse (s)
        Eigen::Vector3d nozzle_pos; // Nozzle position relative to vehicle reference point (m)
        double exit_area;        // Nozzle exit area (m^2) for pressure correction
        
        // Mass Properties
        double propellant_mass;  // Initial propellant mass (kg)
        double dry_mass;         // Case/Nozzle mass (kg)
        double casing_radius;    // For inertia calculation (m)
        double casing_length;    // For inertia calculation (m)
        double payload_mass;     // Payload mass (kg) for burnout check
    };

    SolidMotor(const Config& config) 
        : m_config(config), m_ignited(false), m_ignition_time(0.0) {}

    void ignite(double t) {
        if (!m_ignited) {
            m_ignited = true;
            m_ignition_time = t;
        }
    }

    // Get dry mass
    double get_dry_mass() const {
        return m_config.dry_mass;
    }
    
    // Get initial propellant mass
    double get_propellant_mass() const {
        return m_config.propellant_mass;
    }

    // Calculate inertia based on current mass
    Eigen::Matrix3d get_inertia(double current_mass) const {
        // Estimate current propellant mass
        double m_prop = current_mass - m_config.dry_mass;
        if (m_prop < 0) m_prop = 0;
        
        // Total current mass of motor system
        double m = m_config.dry_mass + m_prop;
        
        // Simplified Cylinder Inertia
        // Better: Interpolate inertia tensor from lookup table?
        // For now, scale based on mass fraction?
        // Or assume uniform depletion (burning from inside out, radius increases).
        // Let's stick to simple cylinder approximation for robustness.
        
        double r = m_config.casing_radius;
        double h = m_config.casing_length;
        
        double Ixx = 0.5 * m * r * r; // Roll
        double Iyy = (1.0/12.0) * m * (3*r*r + h*h); // Pitch/Yaw
        
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        I(0,0) = Ixx;
        I(1,1) = Iyy;
        I(2,2) = Iyy;
        return I;
    }

    /**
     * @brief Calculate thrust and mass flow
     * @param t Current simulation time
     * @param current_mass Current mass of the vehicle
     * @param ambient_pressure Ambient pressure (Pa)
     * @param tvc TVC deflection angles
     * @param com Current Center of Mass of the vehicle (m)
     */
    PropulsionOutput compute(double t, double current_mass, double ambient_pressure, const TVCCommand& tvc, const Eigen::Vector3d& com) {
        PropulsionOutput out;
        out.force_body = Eigen::Vector3d::Zero();
        out.moment_body = Eigen::Vector3d::Zero();
        out.mass_flow_rate = 0.0;

        // Debug Print (Limited)
        // static int debug_cnt = 0;
        // bool do_print = (debug_cnt++ % 1000 == 0) && (t < 70.0); // Print during boost phase

        if (!m_ignited) {
            // if (do_print) printf("[Prop] Not Ignited (t=%.2f)\n", t);
            return out;
        }
        
        // Check for burnout by mass
        if (current_mass <= m_config.dry_mass + m_config.payload_mass + 1.0) { // 1kg buffer
             // if (do_print) printf("[Prop] Mass Burnout: Mass=%.1f Limit=%.1f\n", current_mass, m_config.dry_mass + m_config.payload_mass);
             return out;
        }

        double dt = t - m_ignition_time;
        if (dt < 0 || dt > m_config.burn_time) {
            // if (do_print) printf("[Prop] Time Burnout: dt=%.2f Max=%.2f\n", dt, m_config.burn_time);
            return out;
        }

        // 1. Calculate Thrust Magnitude from Curve
        double F_vac = 0.0;
        if (m_config.time_knots.empty()) {
            // Fallback to average thrust if no curve
            if (m_config.burn_time > 0)
                F_vac = m_config.total_impulse / m_config.burn_time;
        } else {
            F_vac = Utils::interpolate_1d(m_config.time_knots, m_config.thrust_knots, dt);
        }
        
        // Pressure correction: F = F_vac - P_amb * A_exit
        double thrust_mag = F_vac - ambient_pressure * m_config.exit_area;
        if (thrust_mag < 0) thrust_mag = 0; 

        // if (do_print) printf("[Prop] Thrust=%.1f Mass=%.1f dt=%.2f\n", thrust_mag, current_mass, dt);

        // 2. Mass Flow Rate
        // m_dot = -F_vac / (Isp * g0)
        // Note: Use Vacuum Thrust for mass flow!
        double g0 = 9.80665;
        double m_dot = -F_vac / (m_config.isp * g0);
        
        out.mass_flow_rate = m_dot;

        // 3. TVC Deflection
        // Transform thrust vector from Nozzle Frame to Body Frame
        // Assuming Nozzle Frame is aligned with Body Frame when null, 
        // and rotations are small angles or standard Euler sequence.
        // Let's use standard rotation matrix for small deflections:
        // R_body_nozzle = R_y(yaw) * R_z(pitch)  <-- Assuming pitch is rotation about Z (yaw about Y)
        // Actually, let's stick to standard aerospace:
        // Pitch: rotation about Body Y axis.
        // Yaw: rotation about Body Z axis.
        // But TVC usually defines deflection angles delta_y (pitch) and delta_z (yaw).
        
        // Thrust magnitude acts along the nozzle centerline (-X in nozzle frame).
        // F_nozzle = [-F, 0, 0]
        
        // Using rotation matrices for body-to-nozzle (or vice versa):
        // F_body_x = F * cos(pitch) * cos(yaw)
        // F_body_y = F * sin(yaw)
        // F_body_z = -F * sin(pitch)  <-- Sign convention depends on axis definition
        
        // Let's verify standard signs:
        // Positive Pitch Deflection (nozzle up) -> Creates Nose Down moment (-My) -> Force needs component in +Z?
        // Let's rely on the Autopilot to command the correct sign.
        // Here we just implement the geometric transformation.
        
        double cp = cos(tvc.pitch);
        double sp = sin(tvc.pitch);
        double cy = cos(tvc.yaw);
        double sy = sin(tvc.yaw);
        
        // Assuming thrust points OUT of the nozzle (backward). 
        // Force on the body is FORWARD (+X).
        // If nozzle is deflected:
        // Pitch (around Y axis): +angle = nozzle moves up? or down?
        // Let's assume +pitch angle = nozzle deflects UP (Force vector tilts DOWN -> +Z component)
        
        // F_body = R * [F_mag, 0, 0]^T
        // For small angles:
        // Fx = F * cp * cy
        // Fy = F * sy  (Side force)
        // Fz = -F * sp (Vertical force)
        
        Eigen::Vector3d thrust_vec(thrust_mag * cp * cy, thrust_mag * sy, -thrust_mag * sp);
        
        out.force_body = thrust_vec;
        out.mass_flow_rate = m_dot;

        // 4. Moment
        // M = r x F
        // r = Vector from CoM to Nozzle Application Point
        // com is provided in state (relative to nose or arbitrary reference)
        // nozzle_pos is in same reference frame.
        // r_arm = nozzle_pos - com
        
        Eigen::Vector3d r_arm = m_config.nozzle_pos - com;
        out.moment_body = r_arm.cross(thrust_vec);
        
        return out;
    }

private:
    Config m_config;
    bool m_ignited;
    double m_ignition_time;
    double m_propellant_remaining;
};

} // namespace AeroSim
