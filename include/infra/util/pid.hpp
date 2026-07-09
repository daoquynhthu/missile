#pragma once

#include <algorithm>

namespace AeroSim {
namespace Control {

/**
 * @brief Generic PID Controller
 */
class PID {
public:
    struct Config {
        double kp;          // Proportional gain
        double ki;          // Integral gain
        double kd;          // Derivative gain
        double output_min;  // Minimum output limit
        double output_max;  // Maximum output limit
        double integrator_min; // Anti-windup min limit
        double integrator_max; // Anti-windup max limit
    };

    PID(const Config& config) 
        : m_config(config), m_integral(0.0), m_prev_error(0.0), m_first_run(true) {}

    void reset() {
        m_integral = 0.0;
        m_prev_error = 0.0;
        m_first_run = true;
    }

    /**
     * @brief Update PID controller
     * @param error Setpoint - ProcessValue
     * @param dt Time step (seconds)
     * @return Control output
     */
    double update(double error, double dt) {
        if (dt <= 0.0) return 0.0;

        // Proportional term
        double p_term = m_config.kp * error;

        // Integral term
        m_integral += error * dt;
        // Anti-windup clamping
        m_integral = std::max(m_config.integrator_min, std::min(m_config.integrator_max, m_integral));
        double i_term = m_config.ki * m_integral;

        // Derivative term
        double d_term = 0.0;
        if (!m_first_run) {
            d_term = m_config.kd * (error - m_prev_error) / dt;
        } else {
            m_first_run = false;
        }
        m_prev_error = error;

        // Total output
        double output = p_term + i_term + d_term;

        // Output saturation
        return std::max(m_config.output_min, std::min(m_config.output_max, output));
    }

    /**
     * @brief Update PID with explicit derivative (e.g. from gyro rate)
     * Useful to avoid derivative kick on setpoint change.
     * @param error Setpoint - ProcessValue
     * @param rate_pv Rate of change of Process Value (d(PV)/dt)
     * @param dt Time step
     */
    double update_with_rate(double error, double rate_pv, double dt) {
         if (dt <= 0.0) return 0.0;

        // Proportional term
        double p_term = m_config.kp * error;

        // Integral term
        m_integral += error * dt;
        m_integral = std::max(m_config.integrator_min, std::min(m_config.integrator_max, m_integral));
        double i_term = m_config.ki * m_integral;

        // Derivative term: D = -Kd * d(PV)/dt (Derivative on Measurement)
        // This avoids spikes when setpoint changes abruptly.
        double d_term = -m_config.kd * rate_pv;

        // Total output
        double output = p_term + i_term + d_term;

        return std::max(m_config.output_min, std::min(m_config.output_max, output));
    }

private:
    Config m_config;
    double m_integral;
    double m_prev_error;
    bool m_first_run;
};

} // namespace Control
} // namespace AeroSim
