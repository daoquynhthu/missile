#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "utils/pid.hpp"
#include "propulsion_model.hpp"

namespace AeroSim {
namespace GNC {

/**
 * @brief Autopilot for Attitude Control
 * Controls vehicle attitude using TVC (and potentially fins in future)
 */
class Autopilot {
public:
    struct Config {
        Control::PID::Config pitch_pid;
        Control::PID::Config yaw_pid;
        Control::PID::Config roll_pid; // Roll control (if roll thrusters exist)
        
        // RCS specific PIDs (often higher gain or different tuning)
        Control::PID::Config rcs_pitch_pid;
        Control::PID::Config rcs_yaw_pid;
        Control::PID::Config rcs_roll_pid;
        
        // Aero specific PIDs (for control surfaces)
        Control::PID::Config aero_pitch_pid;
        Control::PID::Config aero_yaw_pid;
        Control::PID::Config aero_roll_pid;
    };
    
    struct AutopilotOutput {
        TVCCommand tvc;
        RCSCommand rcs;
        struct AeroCommand {
            double pitch; // Pitch control surface deflection (rad)
            double yaw;   // Yaw control surface deflection (rad)
            double roll;  // Roll control surface deflection (rad)
        } aero;
    };

    Autopilot(const Config& config)
        : m_pitch_pid(config.pitch_pid),
          m_yaw_pid(config.yaw_pid),
          m_roll_pid(config.roll_pid),
          m_rcs_pitch_pid(config.rcs_pitch_pid),
          m_rcs_yaw_pid(config.rcs_yaw_pid),
          m_rcs_roll_pid(config.rcs_roll_pid),
          m_aero_pitch_pid(config.aero_pitch_pid),
          m_aero_yaw_pid(config.aero_yaw_pid),
          m_aero_roll_pid(config.aero_roll_pid) {}

    // Allow setting separate Aero PIDs
    void set_aero_pids(const Control::PID::Config& pitch, const Control::PID::Config& yaw, const Control::PID::Config& roll) {
        m_aero_pitch_pid = Control::PID(pitch);
        m_aero_yaw_pid = Control::PID(yaw);
        m_aero_roll_pid = Control::PID(roll);
    }

    /**
     * @brief Compute Control Commands
     * @param current_quat Current Attitude Quaternion (Body to Inertial/ECEF)
     * @param current_omega Current Angular Velocity in Body Frame (rad/s)
     * @param target_quat Desired Attitude Quaternion (Body to Inertial/ECEF)
     * @param dt Time step
     * @return AutopilotOutput (TVC and RCS commands)
     */
    AutopilotOutput update(const Eigen::Quaterniond& current_quat, 
                      const Eigen::Vector3d& current_omega, 
                      const Eigen::Quaterniond& target_quat, 
                      double dt) {
        
        // 1. Calculate Attitude Error
        Eigen::Quaterniond q_err = current_quat.inverse() * target_quat;
        
        if (q_err.w() < 0) {
            q_err.w() = -q_err.w();
            q_err.x() = -q_err.x();
            q_err.y() = -q_err.y();
            q_err.z() = -q_err.z();
        }

        Eigen::Vector3d error_rot = 2.0 * q_err.vec();
        double roll_err = error_rot.x();
        double pitch_err = error_rot.y();
        double yaw_err = error_rot.z();

        // 2. PID Control
        // TVC (Pitch/Yaw)
            double tvc_pitch_cmd = -m_pitch_pid.update_with_rate(pitch_err, current_omega.y(), dt);
            double tvc_yaw_cmd = -m_yaw_pid.update_with_rate(yaw_err, current_omega.z(), dt);
        
        // RCS (Pitch/Yaw/Roll)
        double rcs_pitch_cmd = m_rcs_pitch_pid.update_with_rate(pitch_err, current_omega.y(), dt);
        double rcs_yaw_cmd = m_rcs_yaw_pid.update_with_rate(yaw_err, current_omega.z(), dt);
        double rcs_roll_cmd = m_rcs_roll_pid.update_with_rate(roll_err, current_omega.x(), dt);
        
        // Aero Control (Pitch/Yaw/Roll)
        // Deflection Convention:
        // Pitch: Positive deflection (Trailing Edge Down) -> Negative Moment (Nose Down)
        // Error > 0 (Need Nose Up) -> Need Positive Moment -> Need Negative Deflection
        // So Command = -PID
        double aero_pitch_cmd = -m_aero_pitch_pid.update_with_rate(pitch_err, current_omega.y(), dt);
        
        // Yaw: Positive deflection (Trailing Edge Left) -> Positive Side Force -> Negative Yaw Moment (Nose Left)
        // Error > 0 (Need Nose Right/Positive Yaw) -> Need Positive Moment -> Need Negative Deflection?
        // Let's assume standard rudder: Positive deflection -> Negative Yaw Moment.
        // So Command = -PID.
        double aero_yaw_cmd = -m_aero_yaw_pid.update_with_rate(yaw_err, current_omega.z(), dt);
        
        // Roll: Positive deflection (Right Aileron Up, Left Down) -> Positive Roll Moment (Right Wing Down, Roll Right)
        // Error > 0 (Need Roll Right) -> Need Positive Moment -> Need Positive Deflection.
        // So Command = +PID.
        double aero_roll_cmd = m_aero_roll_pid.update_with_rate(roll_err, current_omega.x(), dt);

        return AutopilotOutput{
            {tvc_pitch_cmd, tvc_yaw_cmd},
            {rcs_pitch_cmd, rcs_yaw_cmd, rcs_roll_cmd, true},
            {aero_pitch_cmd, aero_yaw_cmd, aero_roll_cmd}
        };
    }
    
    void reset() {
        m_pitch_pid.reset();
        m_yaw_pid.reset();
        m_roll_pid.reset();
        m_rcs_pitch_pid.reset();
        m_rcs_yaw_pid.reset();
        m_rcs_roll_pid.reset();
        m_aero_pitch_pid.reset();
        m_aero_yaw_pid.reset();
        m_aero_roll_pid.reset();
    }

private:
    Control::PID m_pitch_pid;
    Control::PID m_yaw_pid;
    Control::PID m_roll_pid;
    
    Control::PID m_rcs_pitch_pid;
    Control::PID m_rcs_yaw_pid;
    Control::PID m_rcs_roll_pid;
    
    Control::PID m_aero_pitch_pid;
    Control::PID m_aero_yaw_pid;
    Control::PID m_aero_roll_pid;
};

} // namespace GNC
} // namespace AeroSim
