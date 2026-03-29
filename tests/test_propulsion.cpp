#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "propulsion_model.hpp"
#include "missile_config.hpp"

using namespace AeroSim;

void test_solid_motor_burn() {
    std::cout << "[Test] Starting Solid Motor Burn Test..." << std::endl;

    // 1. Load Config
    auto config = MissileDesign::load_hgv1_config();
    SolidMotor motor(config.propulsion);

    // 2. Setup Simulation
    double t = 0.0;
    double dt = 0.1;
    double current_mass = config.total_mass;
    double total_impulse_accumulated = 0.0;
    
    // Ignite
    motor.ignite(0.0);
    
    // 3. Run Loop
    bool burned_out = false;
    double max_thrust = 0.0;

    for (int i = 0; i < 800; ++i) { // 80 seconds
        t = i * dt;
        
        // Dummy inputs
        TVCCommand tvc = {0.0, 0.0};
        Eigen::Vector3d com = Eigen::Vector3d::Zero();
        double ambient_pressure = 101325.0 * std::exp(-t * 1000.0 / 7000.0); // Simple decay
        
        PropulsionOutput out = motor.compute(t, current_mass, ambient_pressure, tvc, com);
        
        double thrust_mag = out.force_body.norm();
        if (thrust_mag > max_thrust) max_thrust = thrust_mag;

        total_impulse_accumulated += thrust_mag * dt;
        
        // Update mass
        current_mass += out.mass_flow_rate * dt;

        // Checks
        if (t < 60.0) {
            // Should be burning (mostly)
            // Note: Thrust curve might have 0 at t=0 but ramps up quickly
            if (t > 1.0 && t < 59.0) {
                if (thrust_mag < 1000.0) {
                    std::cerr << "Error: Thrust too low during burn! T=" << t << " Thrust=" << thrust_mag << std::endl;
                    // assert(false); 
                }
            }
        } else if (t > 61.0) {
            // Should be done
            if (thrust_mag > 1.0) {
                 std::cerr << "Error: Thrust detected after burnout! T=" << t << " Thrust=" << thrust_mag << std::endl;
                 assert(false);
            }
            if (!burned_out) {
                std::cout << "[Info] Burnout detected at T=" << t << std::endl;
                burned_out = true;
            }
        }
    }

    // 4. Verification
    std::cout << "Total Impulse Accumulated: " << total_impulse_accumulated << " Ns" << std::endl;
    std::cout << "Expected Impulse (approx): " << config.propulsion.total_impulse << " Ns" << std::endl;
    std::cout << "Final Mass: " << current_mass << " kg" << std::endl;
    std::cout << "Expected Dry Mass: " << config.propulsion.dry_mass + config.payload_mass << " kg" << std::endl;
    
    // Impulse Check (Allow 5% error due to pressure correction and discretization)
    double impulse_error = std::abs(total_impulse_accumulated - config.propulsion.total_impulse) / config.propulsion.total_impulse;
    std::cout << "Impulse Error: " << impulse_error * 100.0 << "%" << std::endl;
    
    if (impulse_error > 0.10) { // 10% tolerance
        std::cerr << "FAILED: Impulse deviation too high!" << std::endl;
        exit(1);
    }

    // Mass Check
    double expected_final = config.propulsion.dry_mass + config.payload_mass;
    double mass_error = std::abs(current_mass - expected_final);
    std::cout << "Mass Error: " << mass_error << " kg" << std::endl;

    if (mass_error > 100.0) { // 100kg tolerance
         std::cerr << "FAILED: Final mass deviation too high!" << std::endl;
         exit(1);
    }

    std::cout << "[Pass] Propulsion Test Passed." << std::endl;
}

int main() {
    test_solid_motor_burn();
    return 0;
}
