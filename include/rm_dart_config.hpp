#pragma once

#include <Eigen/Dense>
#include "constants.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace AeroSim {
namespace RM {

/**
 * @brief RoboMaster Dart 2022 Physical Parameters
 */
struct DartConfig {
    double mass = 0.22;               // Max mass (kg)
    Eigen::Vector3d com = Eigen::Vector3d(0.1, 0, 0); // Center of Mass (m from nose)
    
    // Inertia (Approx as a cylinder for now, but high-fidelity requires specific values)
    // Dart: L=0.25m, R=0.075m (approx based on 150x150 box)
    // Ix = 0.5 * m * r^2
    // Iy = Iz = 1/12 * m * (3*r^2 + L^2)
    double length = 0.25;
    double radius = 0.04; // Effective radius for inertia
    
    Eigen::Matrix3d inertia;

    // Aerodynamic Reference
    double ref_area = 0.0006;          // Reference Area (m^2) - 30x20mm capsule approx
    double ref_length = 0.25;         // Reference Length (m)

    // Launch Location (Default to user provided: 22.58, 113.96)
    double launch_lat = 22.58;
    double launch_lon = 113.96;
    double launch_alt = 1.5;

    // Environment (Computed at runtime based on location)
    double gravity_mag = 9.788;       // Default, will be updated
    double atm_density = 1.225;       // Default, will be updated
    double sound_speed = 340.0;       // Default, will be updated

    double nav_ratio = 2.0;
    double ctrl_gain = 2.5;
    double ctrl_moment_coeff_pitch = 8.0;
    double ctrl_moment_coeff_yaw = 2.0;
    double ctrl_q_bias = 300.0;
    double max_ctrl_coeff = 0.16;
    double guid_start_time = 0.12;
    double guid_terminal_range = 12.0;
    double guid_update_hz = 330.0;
    double guid_min_tgo = 0.12;
    double guid_max_accel = 6.0;
    double guid_fov_deg = 120.0;
    double guid_sensor_latency = 0.03;
    double guid_y_pos_gain = 2.2;
    double guid_y_vel_gain = 1.0;
    double guid_z_pos_gain = 2.2;
    double guid_z_vel_gain = 1.1;
    double guid_rate_damp = 0.55;
    double guid_dp_gain = 1.75;
    double guid_dy_gain = 1.75;
    double guid_vz_gain = 0.35;
    double guid_dp_step = 0.02;
    double guid_dy_step = 0.02;
    double guid_y_deadband = 0.005;
    double guid_z_deadband = 0.005;
    double guid_lookup_err_1 = 0.235835;
    double guid_lookup_err_2 = 0.256905;
    double guid_lookup_err_3 = 0.277977;
    double guid_lookup_err_4 = 0.299053;
    double guid_lookup_err_5 = 0.320132;
    double guid_lookup_dp_1 = 0.05;
    double guid_lookup_dp_2 = 0.08;
    double guid_lookup_dp_3 = 0.12;
    double guid_lookup_dp_4 = 0.15;
    double guid_lookup_dp_5 = 0.18;
    double guid_lookup_dp_6 = 0.20;
    double target_alt = 0.0;

    // Targets (Positions in local field frame, need to map to ECEF for simulation)
    // Field center is roughly (0,0,0) in our local simulation
    struct Target {
        Eigen::Vector3d pos_local;    // Position in local field frame (m)
        double azimuth_deg;           // Azimuth from launcher (deg)
        double distance;              // Distance from launcher (m)
    };

    double target_ned_z(double target_altitude = 0.0) const {
        return launch_alt - target_altitude;
    }

    Target get_outpost(double target_altitude = 0.0) const {
        return { Eigen::Vector3d(15.865 * std::cos(-6.5 * M_PI / 180.0), 
                                 15.865 * std::sin(-6.5 * M_PI / 180.0),
                                 target_ned_z(target_altitude)), 
                 -6.5, 15.865 };
    }

    Target get_base(double target_altitude = 0.0) const {
        return { Eigen::Vector3d(25.233 * std::cos(7.3 * M_PI / 180.0), 
                                 25.233 * std::sin(7.3 * M_PI / 180.0),
                                 target_ned_z(target_altitude)), 
                 7.3, 25.233 };
    }

    DartConfig() {
        double Ix = 0.5 * mass * radius * radius;
        double Iy = (1.0/12.0) * mass * (3 * radius * radius + length * length);
        inertia = Eigen::Vector3d(Ix, Iy, Iy).asDiagonal();
    }
};

} // namespace RM
} // namespace AeroSim
