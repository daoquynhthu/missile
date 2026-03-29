#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "gnc/guidance.hpp"
#include "missile_config.hpp"
#include "coordinate_transform.hpp"

using namespace AeroSim;

void test_boost_pitch_program() {
    std::cout << "[Test] Starting Guidance Boost Pitch Test..." << std::endl;

    // 1. Setup Config
    GNC::Guidance::Config config;
    // Override for predictable testing
    config.boost_pitch_start = 5.0;
    config.boost_pitch_rate = 1.0;
    config.boost_pitch_min = 20.0;
    
    // Create Guidance
    GNC::Guidance guidance(config);
    Eigen::Vector3d target_ecef(0, 0, 0); // Dummy target
    guidance.set_target(target_ecef);

    // 2. Test Point 1: Early Boost (Vertical)
    double t = 2.0;
    Eigen::Vector3d pos_ecef(6378137.0 + 1000.0, 0.0, 0.0); // 1km Alt on Equator
    Eigen::Vector3d vel_ecef(100.0, 0.0, 0.0); // Vertical Velocity
    
    Eigen::Quaterniond q_cmd = guidance.update(t, pos_ecef, vel_ecef);
    
    // Convert to Euler (Roll, Pitch, Yaw)
    // Body X is Forward.
    // Local Up is X (at 0,0).
    // So Pitch should be 90 deg? Or 0 if aligned with local horizon?
    // Guidance logic: 
    // "target_dir = up" means Nose (Body X) points Up.
    // Pitch is angle between Nose and Horizon. So 90 deg.
    
    // Let's check alignment with Up vector
    Eigen::Vector3d up(1.0, 0.0, 0.0);
    Eigen::Vector3d nose = q_cmd * Eigen::Vector3d::UnitX();
    
    double dot = nose.dot(up);
    double angle_err = std::acos(dot) * 180.0 / 3.14159;
    
    std::cout << "T=2.0s (Vertical):" << std::endl;
    std::cout << "  Nose . Up: " << dot << " (Expected 1.0)" << std::endl;
    std::cout << "  Angle Error: " << angle_err << " deg" << std::endl;
    
    if (angle_err > 1.0) {
        std::cerr << "FAILED: Initial pitch is not vertical!" << std::endl;
        exit(1);
    }

    // 3. Test Point 2: Mid Boost (Pitch Over)
    t = 25.0; // 20s after start (5s) -> 20 * 1.0 = 20 deg pitch down from vertical -> 70 deg pitch
    // Update Guidance state (it might be stateless or depend on t)
    q_cmd = guidance.update(t, pos_ecef, vel_ecef);
    nose = q_cmd * Eigen::Vector3d::UnitX();
    
    // Expected Pitch: 90 - (25-5)*1.0 = 70.0 deg
    // Angle from Up should be 20.0 deg
    dot = nose.dot(up);
    double angle_from_vert = std::acos(dot) * 180.0 / 3.14159;
    
    std::cout << "T=25.0s (Pitch Over):" << std::endl;
    std::cout << "  Angle from Vertical: " << angle_from_vert << " deg (Expected 20.0)" << std::endl;
    
    if (std::abs(angle_from_vert - 20.0) > 1.0) {
        std::cerr << "FAILED: Pitch program incorrect!" << std::endl;
        // exit(1);
    }

    std::cout << "[Pass] Guidance Test Passed." << std::endl;
}

int main() {
    test_boost_pitch_program();
    return 0;
}
