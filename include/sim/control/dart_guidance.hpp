#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "infra/common.hpp"

namespace aerosp {
namespace sim {
namespace control {

class DartGuidance {
public:
    struct Config {
        double nav_ratio = 2.0;
        double max_rudder_deg = 15.0;
        double guidance_start_time = 0.12;
        double visual_fov_deg = 120.0;
        double sensor_latency = 0.03;
        double ctrl_gain = 2.5;
        double ctrl_q_bias = 300.0;
        double max_ctrl_coeff = 0.16;
        double terminal_range = 12.0;
        double update_hz = 330.0;
        double min_tgo = 0.12;
        double max_accel = 6.0;
        double y_pos_gain = 2.2;
        double y_vel_gain = 1.0;
        double z_pos_gain = 2.2;
        double z_vel_gain = 1.1;
        double rate_damp = 0.55;
        double dp_gain = 1.75;
        double dy_gain = 1.75;
        double vz_gain = 0.35;
        double dp_step = 0.02;
        double dy_step = 0.02;
        double y_deadband = 0.005;
        double z_deadband = 0.005;
        double lookup_err_1 = 0.235835;
        double lookup_err_2 = 0.256905;
        double lookup_err_3 = 0.277977;
        double lookup_err_4 = 0.299053;
        double lookup_err_5 = 0.320132;
        double lookup_dp_1 = 0.05;
        double lookup_dp_2 = 0.08;
        double lookup_dp_3 = 0.12;
        double lookup_dp_4 = 0.15;
        double lookup_dp_5 = 0.18;
        double lookup_dp_6 = 0.20;
        double gravity_mag = 9.81;
    };

    struct GuidanceMeasurement {
        Eigen::Vector3d pos_ned = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_ned = Eigen::Vector3d::Zero();
        Eigen::Vector3d r_rel = Eigen::Vector3d::Zero();
        Eigen::Vector3d los = Eigen::Vector3d::UnitX();
        Eigen::Vector3d los_rate_ned = Eigen::Vector3d::Zero();
        double dist = 0.0;
        double x_rem = 0.0;
        double tgo = 0.0;
        double y_err = 0.0;
        double z_err = 0.0;
        bool valid = false;
    };

    struct GuidanceOutput {
        double delta_pitch = 0.0;
        double delta_yaw = 0.0;
        bool active = false;
    };

    struct GuidanceState {
        double last_update_time = -1e9;
        double held_pitch = 0.0;
        double held_yaw = 0.0;
        Eigen::Vector3d sensed_pos_ned = Eigen::Vector3d::Zero();
        Eigen::Vector3d sensed_vel_ned = Eigen::Vector3d::Zero();
        bool measurement_initialized = false;
    };

    DartGuidance(const Config& config)
        : m_config(config) {
        reset();
    }

    void reset() {
        reset_state(m_state);
    }

    GuidanceOutput update(double t,
                          const Eigen::Vector3d& pos_ned,
                          const Eigen::Vector3d& vel_ned,
                          const Eigen::Vector3d& omega_body,
                          const Eigen::Vector3d& target_ned,
                          double dt) {
        return update_closed_loop(t, pos_ned, vel_ned, omega_body, target_ned, dt, m_config, m_state);
    }

    CUDA_HOST_DEVICE static void reset_state(GuidanceState& state) {
        state.last_update_time = -1e9;
        state.held_pitch = 0.0;
        state.held_yaw = 0.0;
        state.sensed_pos_ned.setZero();
        state.sensed_vel_ned.setZero();
        state.measurement_initialized = false;
    }

    CUDA_HOST_DEVICE static GuidanceOutput update_closed_loop(double t,
                                                              const Eigen::Vector3d& pos_ned,
                                                              const Eigen::Vector3d& vel_ned,
                                                              const Eigen::Vector3d& omega_body,
                                                              const Eigen::Vector3d& target_ned,
                                                              double dt,
                                                              const Config& config,
                                                              GuidanceState& state) {
        GuidanceOutput out;
        GuidanceMeasurement meas = make_measurement(t, pos_ned, vel_ned, target_ned, dt, config, state);
        if (!meas.valid) {
            return out;
        }

        out.active = true;
        double update_period = 1.0 / clamp_min(config.update_hz, 1.0);
        if (t - state.last_update_time >= update_period - 1e-9) {
            double tgo = clamp_min(meas.tgo, 1e-3);
            double inv_tgo = 1.0 / tgo;
            double inv_tgo_sq = inv_tgo * inv_tgo;

            Eigen::Vector3d v_rel = -meas.vel_ned;
            double closing_speed = clamp_min(-(meas.r_rel.dot(v_rel)) / clamp_min(meas.dist, 1e-6), 0.0);
            double ay_pn = -config.nav_ratio * closing_speed * meas.los_rate_ned.z();
            double az_pn =  config.nav_ratio * closing_speed * meas.los_rate_ned.y();

            double ay_zem = config.y_pos_gain * meas.y_err * inv_tgo_sq;
            double az_zem = config.z_pos_gain * meas.z_err * inv_tgo_sq;
            double ay_damp = config.y_vel_gain * (-meas.vel_ned.y()) * inv_tgo;
            double az_damp = config.z_vel_gain * (-meas.vel_ned.z()) * inv_tgo;

            double ay_cmd = config.ctrl_gain * (ay_zem + ay_damp + ay_pn);
            double az_cmd = config.ctrl_gain * (az_zem + az_damp + az_pn);

            double accel_limit = clamp_min(config.max_accel, 0.1);
            ay_cmd = clamp_value(ay_cmd, -accel_limit, accel_limit);
            az_cmd = clamp_value(az_cmd, -accel_limit, accel_limit);

            if (std::abs(meas.y_err) < config.y_deadband && std::abs(meas.vel_ned.y()) < 0.05) {
                ay_cmd = 0.0;
            }
            if (std::abs(meas.z_err) < config.z_deadband && std::abs(meas.vel_ned.z()) < 0.05) {
                az_cmd = 0.0;
            }

            double v_forward = clamp_min(std::abs(meas.vel_ned.x()), 1.0);
            double r_cmd = ay_cmd / v_forward;
            double q_cmd = az_cmd / v_forward;

            double dp_target = config.dp_gain * (q_cmd - omega_body.y());
            double dy_target = config.dy_gain * (r_cmd - omega_body.z());

            dp_target -= config.rate_damp * omega_body.y();
            dy_target -= config.rate_damp * omega_body.z();

            dp_target = clamp_value(dp_target, -config.max_ctrl_coeff, config.max_ctrl_coeff);
            dy_target = clamp_value(dy_target, -config.max_ctrl_coeff, config.max_ctrl_coeff);

            state.held_pitch = apply_step_limit(state.held_pitch, dp_target, config.dp_step, config.max_ctrl_coeff);
            state.held_yaw = apply_step_limit(state.held_yaw, dy_target, config.dy_step, config.max_ctrl_coeff);
            state.last_update_time = t;
        }

        double limit_rad = config.max_rudder_deg * 3.14159265358979323846 / 180.0;
        out.delta_pitch = clamp_value(state.held_pitch, -limit_rad, limit_rad);
        out.delta_yaw = clamp_value(state.held_yaw, -limit_rad, limit_rad);
        return out;
    }

private:
    CUDA_HOST_DEVICE static double clamp_value(double value, double lower, double upper) {
        if (value < lower) return lower;
        if (value > upper) return upper;
        return value;
    }

    CUDA_HOST_DEVICE static double clamp_min(double value, double lower) {
        return value < lower ? lower : value;
    }

    CUDA_HOST_DEVICE static double clamp_abs_dot(double value) {
        return clamp_value(value, -1.0, 1.0);
    }

    CUDA_HOST_DEVICE static double apply_step_limit(double held_value,
                                                    double target_value,
                                                    double step_limit,
                                                    double abs_limit) {
        double max_step = clamp_min(step_limit, 1e-6);
        double delta = target_value - held_value;
        delta = clamp_value(delta, -max_step, max_step);
        return clamp_value(held_value + delta, -abs_limit, abs_limit);
    }

    CUDA_HOST_DEVICE static void update_measurement_state(const Eigen::Vector3d& pos_ned,
                                                          const Eigen::Vector3d& vel_ned,
                                                          double dt,
                                                          const Config& config,
                                                          GuidanceState& state) {
        if (!state.measurement_initialized) {
            state.sensed_pos_ned = pos_ned;
            state.sensed_vel_ned = vel_ned;
            state.measurement_initialized = true;
            return;
        }

        double lag = config.sensor_latency;
        if (lag <= 1e-9) {
            state.sensed_pos_ned = pos_ned;
            state.sensed_vel_ned = vel_ned;
            return;
        }

        double alpha = clamp_value(dt / clamp_min(lag, dt), 0.0, 1.0);
        state.sensed_pos_ned += (pos_ned - state.sensed_pos_ned) * alpha;
        state.sensed_vel_ned += (vel_ned - state.sensed_vel_ned) * alpha;
    }

    CUDA_HOST_DEVICE static GuidanceMeasurement make_measurement(double t,
                                                                 const Eigen::Vector3d& pos_ned,
                                                                 const Eigen::Vector3d& vel_ned,
                                                                 const Eigen::Vector3d& target_ned,
                                                                 double dt,
                                                                 const Config& config,
                                                                 GuidanceState& state) {
        GuidanceMeasurement meas;

        if (t < config.guidance_start_time) {
            return meas;
        }

        update_measurement_state(pos_ned, vel_ned, dt, config, state);

        meas.pos_ned = state.sensed_pos_ned;
        meas.vel_ned = state.sensed_vel_ned;
        meas.r_rel = target_ned - meas.pos_ned;
        meas.dist = meas.r_rel.norm();
        meas.x_rem = meas.r_rel.x();

        if (meas.dist < 0.5 || meas.x_rem <= 0.0 || meas.x_rem > config.terminal_range) {
            return meas;
        }

        double vel_norm = meas.vel_ned.norm();
        if (vel_norm < 1e-6) {
            return meas;
        }

        meas.los = meas.r_rel / clamp_min(meas.dist, 1e-6);
        double angle = std::acos(clamp_abs_dot(meas.vel_ned.dot(meas.los) / vel_norm)) * 180.0 / 3.14159265358979323846;
        if (angle > config.visual_fov_deg * 0.5) {
            return meas;
        }

        Eigen::Vector3d v_rel = -meas.vel_ned;
        meas.los_rate_ned = meas.r_rel.cross(v_rel) / clamp_min(meas.dist * meas.dist, 1e-6);
        meas.tgo = clamp_min(meas.x_rem / clamp_min(meas.vel_ned.x(), 1.0), config.min_tgo);
        meas.y_err = target_ned.y() - (meas.pos_ned.y() + meas.vel_ned.y() * meas.tgo);
        meas.z_err = target_ned.z() - (meas.pos_ned.z() + meas.vel_ned.z() * meas.tgo + 0.5 * config.gravity_mag * meas.tgo * meas.tgo);
        meas.valid = true;
        return meas;
    }

    Config m_config;
    GuidanceState m_state;
};

} // namespace control
} // namespace sim
}
