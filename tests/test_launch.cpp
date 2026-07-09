#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "sim/dynamics/dynamics_6dof.hpp"
#include "sim/propulsion/propulsion_model.hpp"
#include "aero/aerodynamics_model.hpp"
#include "sim/control/guidance.hpp"
#include "sim/control/autopilot.hpp"
#include "sim/gravity/gravity_model.hpp"
#include "sim/atmosphere/atmosphere_model.hpp"
#include "config/missile_config.hpp"
// #include "infra/util/integrator.hpp" // Removed

using namespace aerosp;

// Simplified System for Test
struct TestSystem {
    GravityModel& gravity;
    SolidMotor& motor;
    RCSModel& rcs;
    AerodynamicsModel& aero;
    sim::control::Autopilot& autopilot;
    sim::control::Guidance& guidance;
    config::HGV1Config& config;
    
    // State
    sim::control::Autopilot::AutopilotOutput current_cmd;
    InertialProps initial_inertia;

    TestSystem(GravityModel& g, SolidMotor& m, RCSModel& r, AerodynamicsModel& a, sim::control::Autopilot& ap, sim::control::Guidance& gd, config::HGV1Config& cfg)
        : gravity(g), motor(m), rcs(r), aero(a), autopilot(ap), guidance(gd), config(cfg) {
            initial_inertia.mass = config.total_mass;
            initial_inertia.com = Eigen::Vector3d(-6.0, 0, 0); // Approx
            initial_inertia.inertia = Eigen::Matrix3d::Identity() * 10000.0; // Dummy
            
            // Initialize Command
            current_cmd.tvc = {0,0};
            current_cmd.rcs = {0,0,0,false};
            current_cmd.aero = {0,0,0};
    }

    State6DOF operator()(const State6DOF& state, double t) {
        // 1. Gravity
        Eigen::Vector3d g_ecef = gravity.calculate_acceleration(state.pos_ecef);
        
        // 2. Atmosphere (Simplified)
        double alt = state.pos_ecef.norm() - 6378137.0;
        double rho = 1.225 * exp(-alt / 8500.0);
        
        // 4. Forces
        ForcesMoments fm;
        fm.force_body = Eigen::Vector3d::Zero();
        fm.moment_body = Eigen::Vector3d::Zero();
        
        // Propulsion
        auto prop_out = motor.compute(t, state.mass, 101325.0, current_cmd.tvc, initial_inertia.com);
        fm.force_body += prop_out.force_body;
        fm.moment_body += prop_out.moment_body;
        fm.mass_flow_rate = prop_out.mass_flow_rate;
        
        // Aero (Simplified Drag only for Launch)
        Eigen::Vector3d vel_body = state.quat_be.inverse() * state.vel_ecef;
        double v_mag = vel_body.norm();
        if (v_mag > 1.0) {
            double q = 0.5 * rho * v_mag * v_mag;
            double cd = 0.5; // High drag estimate
            double drag = q * config.aerodynamics.ref_area * cd;
            fm.force_body.x() -= drag; 
        }

        // 5. Dynamics
        InertialProps current_props = initial_inertia;
        current_props.mass = state.mass;
        
        // We need to return state derivative
        return Dynamics6DOF::compute_derivatives(state, fm, current_props, g_ecef);
    }
};

void test_launch_vertical() {
    std::cout << "[Test] Starting Launch Dynamics Test..." << std::endl;
    
    // 1. Setup
    auto config = config::load_hgv1_config();
    GravityModel gravity(4);
    SolidMotor motor(config.propulsion);
    RCSModel::Config rcs_cfg = {1000.0, 200.0, 10.0, 1.0};
    RCSModel rcs_model(rcs_cfg);
    
    AerodynamicsModel aero(config.aerodynamics);
    sim::control::Autopilot autopilot(config.autopilot);
    sim::control::Guidance::Config guide_cfg;
    guide_cfg.boost_pitch_start = 5.0;
    sim::control::Guidance guidance(guide_cfg);
    
    TestSystem system(gravity, motor, rcs_model, aero, autopilot, guidance, config);
    
    // 2. Initial State (Launch Pad)
    State6DOF state;
    state.pos_ecef = Eigen::Vector3d(6378137.0, 0, 0); // On Equator
    state.vel_ecef = Eigen::Vector3d(0, 0, 0);         
    state.quat_be = Eigen::Quaterniond::Identity(); 
    state.mass = config.total_mass;
    state.omega_body = Eigen::Vector3d::Zero(); // Corrected from ang_vel_body
    
    // Ignite Motor
    motor.ignite(0.0);
    
    // 3. Integrate for 10 seconds
    double t = 0.0;
    double dt = 0.01;
    
    std::cout << "t=0.0 Alt=0.0 Vel=0.0 Mass=" << state.mass << std::endl;
    
    for (int i=0; i<1000; ++i) {
        // Use Dynamics6DOF built-in RK4
        state = Dynamics6DOF::integrate_rk4(state, system, t, dt);
        t += dt;
        
        if (i % 100 == 0) {
             double alt = state.pos_ecef.norm() - 6378137.0;
             double vel = state.vel_ecef.norm();
             std::cout << "t=" << t << " Alt=" << alt << " Vel=" << vel << " Mass=" << state.mass << std::endl;
             
             // Checks
             if (std::isnan(alt) || std::isnan(vel)) {
                 std::cerr << "FAILED: NaN detected!" << std::endl;
                 exit(1);
             }
        }
    }
    
    // 4. Verify Ascent
    double final_alt = state.pos_ecef.norm() - 6378137.0;
    double final_vel = state.vel_ecef.norm();
    
    std::cout << "Final Alt: " << final_alt << " m" << std::endl;
    std::cout << "Final Vel: " << final_vel << " m/s" << std::endl;
    
    if (final_alt < 100.0) {
        std::cerr << "FAILED: Missile did not ascend significantly!" << std::endl;
        exit(1);
    }
    
    if (final_vel < 50.0) {
        std::cerr << "FAILED: Velocity too low!" << std::endl;
        exit(1);
    }

    std::cout << "[Pass] Launch Dynamics Test Passed." << std::endl;
}

int main() {
    test_launch_vertical();
    return 0;
}
