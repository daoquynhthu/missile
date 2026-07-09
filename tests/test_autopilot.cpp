#include <iostream>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "sim/control/autopilot.hpp"
#include "config/missile_config.hpp"

using namespace AeroSim;

void test_autopilot_polarity() {
    std::cout << "[Test] Starting Autopilot Polarity Test..." << std::endl;

    // 1. Setup Config
    auto config = MissileDesign::load_hgv1_config();
    GNC::Autopilot autopilot(config.autopilot);

    // 2. Scenario: Pitch Up Command
    // Current Attitude: Level (Identity)
    Eigen::Quaterniond current_quat = Eigen::Quaterniond::Identity();
    
    // Target Attitude: 10 deg Pitch Up
    double target_pitch_deg = 10.0;
    double target_pitch_rad = target_pitch_deg * 3.14159 / 180.0;
    Eigen::Quaterniond target_quat;
    target_quat = Eigen::AngleAxisd(target_pitch_rad, Eigen::Vector3d::UnitY());
    
    // Current Rates: Zero
    Eigen::Vector3d current_omega = Eigen::Vector3d::Zero();
    
    double dt = 0.01;

    // 3. Run Autopilot
    auto output = autopilot.update(current_quat, current_omega, target_quat, dt);
    
    std::cout << "Target Pitch: " << target_pitch_deg << " deg" << std::endl;
    std::cout << "TVC Pitch Command: " << output.tvc.pitch << " rad" << std::endl;
    
    // 4. Analyze Polarity
    // We want to Pitch Up (Positive Y Moment).
    // In our Propulsion Model:
    //   Moment_Y = r_x * F_z - r_z * F_x = (-L) * F_z (approx)
    //   F_z (Force Body Z) = -F_thrust * sin(tvc_pitch)
    //   So Moment_Y = (-L) * (-F * sin(theta)) = +L * F * sin(theta) ?
    //   Wait, r_x is negative (Nozzle at -12, CoM at -6, r_x = -6).
    //   Moment_Y = -r_x * F_z = -(-6) * F_z = +6 * F_z.
    //   So Moment_Y has same sign as F_z.
    //   To Pitch Nose UP (Positive Moment_Y), we need Positive F_z (Force Down).
    //   F_z = -F * sin(theta).
    //   To get Positive F_z, we need Negative sin(theta).
    //   So we need NEGATIVE TVC Pitch.
    
    if (output.tvc.pitch >= 0.0) {
        std::cerr << "FAILED: Autopilot commanded Positive/Zero pitch for Positive error!" << std::endl;
        std::cerr << "        We need Negative pitch (Nozzle Up -> Force Down -> Nose Up)." << std::endl;
        exit(1);
    } else {
        std::cout << "Polarity OK: Positive Error -> Negative Command (Correct for Tail Control)." << std::endl;
    }

    // 5. Scenario: Yaw Right Command
    // Target: 10 deg Yaw (Nose Right, Positive Z Rotation)
    // We need Nose Right. Tail must move Left (Negative Y Force).
    // F_y = F * sin(yaw).
    // We need Negative F_y.
    // So we need Negative Yaw Command.
    
    Eigen::Quaterniond target_yaw = Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()));
    output = autopilot.update(current_quat, current_omega, target_yaw, dt);
    
    std::cout << "TVC Yaw Command: " << output.tvc.yaw << " rad" << std::endl;
    
    if (output.tvc.yaw >= 0.0) {
        std::cerr << "FAILED: Autopilot commanded Positive Yaw for Positive Error." << std::endl;
        std::cerr << "        We need Negative Yaw (Nozzle Right -> Force Left -> Nose Right)." << std::endl;
        exit(1);
    } else {
        std::cout << "Yaw Polarity OK: Positive Error -> Negative Command." << std::endl;
    }

    std::cout << "[Pass] Autopilot Test Passed." << std::endl;
}

int main() {
    test_autopilot_polarity();
    return 0;
}
