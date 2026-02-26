#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include "constants.hpp"
#include "gravity_model.hpp"
#include "atmosphere_model.hpp"
#include "dynamics_6dof.hpp"
#include "coordinate_transform.hpp"
#include "utils/progress_bar.hpp"
#include "propulsion_model.hpp"
#include "aerodynamics_model.hpp"
#include "gnc/autopilot.hpp"
#include "gnc/guidance.hpp" 
#include "missile_config.hpp" // Load Missile Design

using namespace AeroSim;
using namespace AeroSim::GNC;

// Helper to create NRLMSISE-00 input
NRLMSISE00Input get_atmosphere_input(double t, const LLA& lla) {
    NRLMSISE00Input input;
    input.year = 2026; 
    input.doy = 172;   
    input.sec = t;     
    input.alt = lla.alt / 1000.0; 
    input.lat = lla.lat;
    input.lon = lla.lon;
    input.lst = input.sec / 3600.0 + input.lon / 15.0; 
    input.f107A = 150.0;
    input.f107 = 150.0;
    input.ap = 4.0;
    for(int i=0; i<7; ++i) input.ap_vector[i] = 0.0;
    return input;
}

// Functor for system dynamics
struct MissileSystem {
    GravityModel& gravity;
    SolidMotor& motor;
        RCSModel& rcs; // New RCS
    AerodynamicsModel& aero; 
    InertialProps& initial_inertia;
    Autopilot& autopilot;
    Guidance& guidance; 
    Eigen::Quaterniond target_attitude; // Command
    double min_mass; // Minimum mass (dry + payload)

    // Constructor
    MissileSystem(GravityModel& g, SolidMotor& m, RCSModel& r, AerodynamicsModel& a, InertialProps& i, Autopilot& ap, Guidance& gd, double mm) 
        : gravity(g), motor(m), rcs(r), aero(a), initial_inertia(i), autopilot(ap), guidance(gd), target_attitude(Eigen::Quaterniond::Identity()), min_mass(mm) {}
        
    // Stored Control Command (computed once per step)
    Autopilot::AutopilotOutput current_cmd = {{0.0, 0.0}, {0.0, 0.0, 0.0, false}};

    // The operator() required by integrate_rk4
    State6DOF operator()(const State6DOF& state, double t) {
        // 1. Calculate Gravity (re-evaluated at current position)
        Eigen::Vector3d g_ecef = gravity.calculate_acceleration(state.pos_ecef);
        
        // 2. Calculate Atmosphere
        LLA lla = CoordinateTransform::ecef_to_lla(state.pos_ecef);
        // Use NRLMSISE-00 (High Fidelity)
        AtmosphereData atm = AtmosphereModel::calculate(get_atmosphere_input(t, lla));
        
        // 3. Calculate Aerodynamics (High Fidelity)
        ForcesMoments fm;
        fm.force_body = Eigen::Vector3d::Zero();
        fm.moment_body = Eigen::Vector3d::Zero();
        fm.mass_flow_rate = 0.0;
        
        // Transform velocity to body frame (assuming no wind)
        // state.vel_ecef is velocity relative to Earth (ECEF).
        // Atmosphere rotates with Earth, so Airspeed = vel_ecef (if wind = 0).
        Eigen::Vector3d v_air_ecef = state.vel_ecef;
        
        // Transform to Body Frame
        Eigen::Vector3d v_air_body = state.quat_be.inverse() * v_air_ecef;
        
        double v_mag = v_air_body.norm();
        double mach = v_mag / atm.sound_speed;
        double dynamic_pressure = 0.5 * atm.density * v_mag * v_mag;
        
        if (v_mag > 1.0) {
            // Calculate Alpha and Beta
            // v_air_body = [u, v, w]
            // alpha = atan2(w, u)
            // beta = asin(v / V)
            double u = v_air_body.x();
            double v = v_air_body.y();
            double w = v_air_body.z();
            
            double alpha = std::atan2(w, u);
            double beta = std::asin(v / v_mag);
            
            // Current CoM (approximation based on mass)
            // Linear interpolation of CoM based on mass ratio?
            // Let's assume CoM is constant for now as user didn't provide mass props table.
            // Using current state.mass to update inertia tensor scalar only.
            
            auto aero_forces = aero.compute_forces_moments(
                dynamic_pressure, mach, alpha, beta, initial_inertia.com,
                current_cmd.aero.pitch, current_cmd.aero.yaw, current_cmd.aero.roll
            );
            
            fm.force_body += aero_forces.first;
            fm.moment_body += aero_forces.second;
        }

        // 4. Calculate Propulsion
        // Main Motor (TVC)
        PropulsionOutput prop_main = motor.compute(t, state.mass, atm.pressure, current_cmd.tvc, initial_inertia.com);
        
        // RCS
        PropulsionOutput prop_rcs = rcs.compute(current_cmd.rcs, state.mass, min_mass);
        
        // Combine
        PropulsionOutput prop_total = prop_main + prop_rcs;
        
        fm.force_body += prop_total.force_body;
        fm.moment_body += prop_total.moment_body;
        fm.mass_flow_rate = prop_total.mass_flow_rate;
        
        // Update inertia mass for dynamics calculation
        InertialProps current_inertia = initial_inertia;
        current_inertia.mass = state.mass;
        // Scale inertia tensor: I_new = I_initial * (m_new / m_initial)
        if (initial_inertia.mass > 0) {
            double ratio = state.mass / initial_inertia.mass;
            current_inertia.inertia = initial_inertia.inertia * ratio;
        }
        
        // 5. Compute Derivatives using the core dynamics equation
        return Dynamics6DOF::compute_derivatives(state, fm, current_inertia, g_ecef);
    }
};

int main(int argc, char** argv) {
    std::cout << "===============================================" << std::endl;
    std::cout << "  AeroSim - High Fidelity Ballistic Simulation " << std::endl;
    std::cout << "===============================================" << std::endl;

    // Parse Command Line Arguments
    // Defaults set to optimized values for Qian Xuesen trajectory (Skip-Glide)
    // Adjusted to ensure low apogee (<100km) and Max Range
    double boost_pitch_start = 6.4375; // Optimized via Nelder-Mead (Low Gain Stable)
    double boost_pitch_rate = 2.775;   // Optimized via Nelder-Mead (Low Gain Stable)
    double boost_pitch_min = 5.0;     // Minimum pitch limit (Allow flattening)
    double t_end = 1200.0;           // Simulation duration

    if (argc >= 2) boost_pitch_start = std::atof(argv[1]);
    if (argc >= 3) boost_pitch_rate = std::atof(argv[2]);
    if (argc >= 4) boost_pitch_min = std::atof(argv[3]);
    if (argc >= 5) t_end = std::atof(argv[4]);

    std::cout << "[Args] Start: " << boost_pitch_start 
              << ", Rate: " << boost_pitch_rate 
              << ", Min: " << boost_pitch_min 
              << ", T_end: " << t_end << std::endl;
    
    // 1. Initialize Models
    std::cout << "[Init] Loading Gravity Model..." << std::endl;
    // Use lower degree for quick verification (N=20 for speed, N=70 for high precision, N=360 for full)
    // User requested < 5 mins. Degree 70 is too slow (~1hr).
    // Degree 4 (J2, J3, J4) captures primary oblateness and asymmetry, sufficient for trajectory shape.
    GravityModel gravity(4); 
    // Try to load coefficients, fallback if not found
    if (!gravity.load_coefficients("e:/missile/data/EGM2008.gfc")) {
        std::cerr << "[Warning] EGM2008.gfc not found, using default point mass + J2" << std::endl;
    } else {
        std::cout << "[Init] Loaded EGM2008 coefficients up to degree " << gravity.get_loaded_max_degree() << std::endl;
    }

    // Load Missile Configuration
    MissileDesign::HGV1Config hgv_config = MissileDesign::load_hgv1_config();
    std::cout << "[Init] Loaded Design: " << hgv_config.name << std::endl;
    std::cout << "[Init] Total Mass: " << hgv_config.total_mass << " kg, Payload: " << hgv_config.payload_mass << " kg" << std::endl;

    // Initialize Propulsion
    SolidMotor motor(hgv_config.propulsion);

    // Initialize RCS
    RCSModel::Config rcs_cfg;
    rcs_cfg.max_thrust = 5000.0; // 5kN thrusters
    rcs_cfg.isp = 220.0;
    rcs_cfg.lever_arm_x = 8.0;   // Near nose/tail
    rcs_cfg.lever_arm_r = 0.75;  // Body radius
    RCSModel rcs(rcs_cfg);

    // Initialize Aerodynamics
    AerodynamicsModel aero(hgv_config.aerodynamics);

    // Initialize Autopilot
    Autopilot autopilot(hgv_config.autopilot);
    
    // Guidance Config (Qian Xuesen Trajectory - Optimized for Skip)
    GNC::Guidance::Config guid_cfg;
    guid_cfg.boost_end_time = hgv_config.propulsion.burn_time;
    guid_cfg.boost_pitch_start = boost_pitch_start; 
    guid_cfg.boost_pitch_rate = boost_pitch_rate; 
    guid_cfg.boost_pitch_min = boost_pitch_min;

    std::cout << "[Config] Boost Pitch Start: " << guid_cfg.boost_pitch_start << std::endl;
    std::cout << "[Config] Boost Pitch Rate: " << guid_cfg.boost_pitch_rate << std::endl;

    guid_cfg.glide_alt_start = 50000.0; // Match burnout altitude
    guid_cfg.glide_alt_end = 20000.0;
    guid_cfg.glide_vel_min = 800.0;
    guid_cfg.hysteresis_margin = 5000.0; // Larger margin to prevent flickering

    guid_cfg.glide_aoa_bias = 5.0; // Low bias
    guid_cfg.glide_aoa_max = 20.0; // Increase max lift for skip
    guid_cfg.glide_aoa_min = -10.0; // Allow negative lift
    
    guid_cfg.kp_alt = 0.05; // Stronger altitude capture
    guid_cfg.kp_vz = 0.1;   // Stronger damping
    guid_cfg.max_climb_rate = 200.0; 
    guid_cfg.max_descent_rate = -300.0;

    guid_cfg.lateral_gain = 3.0; 
    guid_cfg.max_bank_angle = 60.0; 

    guid_cfg.target_range = 2200000.0;
    GNC::Guidance guidance(guid_cfg);
    
    // Set Target (Pacific Ocean - Guam Region approx)
    // Jiuquan (40.96N, 100.30E) -> ~2800km Range -> ~135E, 20N
    // Target: 20.0 N, 135.0 E
    LLA target_lla = {20.0 * AeroSim::Math::DEG2RAD(), 135.0 * AeroSim::Math::DEG2RAD(), 0.0};
    guidance.set_target(CoordinateTransform::lla_to_ecef(target_lla));

    // 2. Initial State Setup
    std::cout << "[Init] Setting up initial state..." << std::endl;
    // Launch Site: Jiuquan Satellite Launch Center (JSLC)
    // Lat: 40.960556 N, Lon: 100.298333 E, Alt: 1000m
    LLA init_lla = {40.960556 * AeroSim::Math::DEG2RAD(), 100.298333 * AeroSim::Math::DEG2RAD(), 1000.0}; 
    Eigen::Vector3d init_pos = CoordinateTransform::lla_to_ecef(init_lla);
    
    State6DOF state;
    state.pos_ecef = init_pos;
    state.vel_ecef = Eigen::Vector3d::Zero(); // Launch from rest
    
    // We want Body Z aligned with Local East (Launch Azimuth = 90 deg East)
    // At Launch Site (Lat=40.96, Lon=100.3):
    // Up, North, East are local vectors.
    // Launch Vertical: Body X = Up.
    // Body Y = South (-North) or similar?
    // Let's use Guidance to initialize orientation? Or just construct it.
    
    // Construct Initial Quaternion: Vertical, with Body Z pointing East.
    // ECEF Frame vectors at launch site:
    Eigen::Vector3d up_init = init_pos.normalized();
    Eigen::Vector3d earth_z(0,0,1);
    Eigen::Vector3d north_init = (earth_z - (earth_z.dot(up_init) * up_init)).normalized();
    Eigen::Vector3d east_init = north_init.cross(up_init).normalized();
    
    // Body Frame:
    // X_b = Up
    // Z_b = East
    // Y_b = Z_b x X_b = East x Up = North? No.
    // East x Up = (North x Up) x Up? No.
    // East = North x Up.
    // East x Up = (North x Up) x Up = North x (Up x Up) ... wait.
    // Right Hand Rule: North(Index) x Up(Middle) = East(Thumb).
    // East(Index) x Up(Middle) = North?
    // Let's check: (1,0,0) x (0,0,1) = (0,-1,0) = -Y.
    // So East x Up = -North = South.
    // So Y_b = South.
    
    Eigen::Matrix3d init_rot;
    init_rot.col(0) = up_init;    // Body X (Nose)
    init_rot.col(1) = -north_init; // Body Y (Right Wing) -> South
    init_rot.col(2) = east_init;   // Body Z (Belly/Top?) -> East
    
    state.quat_be = Eigen::Quaterniond(init_rot);
    
    state.omega_body.setZero();
    state.mass = hgv_config.total_mass;

    InertialProps inertia;
    inertia.mass = state.mass;
    // Cylinder Inertia: I = 1/12 * m * (3*r^2 + h^2)
    // Estimate based on config mass
    double estimated_inertia = (1.0/12.0) * state.mass * (100.0); // Approx length^2
    inertia.inertia = Eigen::Matrix3d::Identity() * estimated_inertia; 
    inertia.com = Eigen::Vector3d(-4.8, 0, 0); // Move CG back to -4.8m (CP ~5.2-5.5) to reduce static margin for better turn authority

    // 3. Create System Dynamics
    double min_mass = hgv_config.propulsion.dry_mass + hgv_config.payload_mass;
    MissileSystem missile(gravity, motor, rcs, aero, inertia, autopilot, guidance, min_mass);
    
    // No manual target attitude command anymore - Guidance handles it.
    
    // Ignite motor at t=0.1s
    motor.ignite(0.1);

    // 4. Simulation Loop
    double t = 0.0;
    double dt = 0.02; // 20ms step (50Hz) - Compromise between speed and accuracy
    // Run longer to see re-entry/glide
    // t_end is defined above

    std::cout << "[Sim] Starting integration (RK4)..." << std::endl;
    std::cout << "Time(s) | Alt(m) | Vel(m/s) | Mach | Phase | Pitch(deg) | Thrust(N) | T_body_y | M_body_y | Pitch_Err" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    // Open CSV file
    std::ofstream csv_file("simulation_data.csv");
    if (csv_file.is_open()) {
        csv_file << "Time(s),Alt(m),Vel(m/s),Mach,Phase,Pitch(deg),Thrust(N),T_body_y,M_body_y,Pitch_Err,Lat(deg),Lon(deg),Mass(kg),Pitch_Cmd(deg)\n";
    }

    size_t total_steps = (size_t)(t_end/dt);
    ProgressBar sim_progress(total_steps, 50, "Simulation");

    for (int i = 0; i <= total_steps; ++i) {
        
        // Ground Interaction / Clamp (Prevent negative altitude)
        LLA lla_check = CoordinateTransform::ecef_to_lla(state.pos_ecef);
        if (lla_check.alt < 0.0) {
            lla_check.alt = 0.0;
            state.pos_ecef = CoordinateTransform::lla_to_ecef(lla_check);
            // If moving down (towards center), stop
            Eigen::Vector3d up = state.pos_ecef.normalized();
            if (state.vel_ecef.dot(up) < 0) {
                state.vel_ecef.setZero();
                state.omega_body.setZero();
                std::cout << "Impact detected at t=" << t << "s. Simulation aborted." << std::endl;
                break;
            }
        }

        // CSV Logging (High Frequency: 1s)
        if (i % 100 == 0) {
            LLA current_lla = CoordinateTransform::ecef_to_lla(state.pos_ecef);
            
            // Re-calculate some values for logging
            AtmosphereData atm = AtmosphereModel::calculate(get_atmosphere_input(t, current_lla));
            // Recalculate forces for logging (Motor + RCS)
            PropulsionOutput prop_main = motor.compute(t, state.mass, atm.pressure, missile.current_cmd.tvc, inertia.com);
            PropulsionOutput prop_rcs = rcs.compute(missile.current_cmd.rcs, state.mass, min_mass);
            PropulsionOutput prop = prop_main + prop_rcs;
            
            double mach = state.vel_ecef.norm() / atm.sound_speed;
            
            Eigen::Vector3d up = state.pos_ecef.normalized();
            Eigen::Vector3d body_x = state.quat_be * Eigen::Vector3d::UnitX();
            double dot = body_x.dot(up);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            double angle_from_vertical = std::acos(dot) * AeroSim::Math::RAD2DEG();
            double pitch_deg = 90.0 - angle_from_vertical;

            double t_body_y = prop.force_body.y();
            double m_body_y = prop.moment_body.y();
            
            Eigen::Quaterniond guid_target = guidance.update(t, state.pos_ecef, state.vel_ecef);
            Eigen::Quaterniond q_err = state.quat_be.inverse() * guid_target;
            if (q_err.w() < 0) q_err.coeffs() = -q_err.coeffs();
            double pitch_err = 2.0 * q_err.y();

            // Debug: Log Guidance Target Pitch
            Eigen::Vector3d target_x = guid_target * Eigen::Vector3d::UnitX();
            double target_dot = target_x.dot(up);
            if (target_dot > 1.0) target_dot = 1.0;
            if (target_dot < -1.0) target_dot = -1.0;
            double target_pitch_deg = 90.0 - std::acos(target_dot) * AeroSim::Math::RAD2DEG();

            if (csv_file.is_open()) {
                csv_file << t << "," 
                         << current_lla.alt << "," 
                         << state.vel_ecef.norm() << "," 
                         << mach << ","
                         << (int)guidance.get_phase() << ","
                         << pitch_deg << ","
                         << prop.force_body.norm() << ","
                         << t_body_y << ","
                         << m_body_y << ","
                         << pitch_err << ","
                         << current_lla.lat * AeroSim::Math::RAD2DEG() << ","
                         << current_lla.lon * AeroSim::Math::RAD2DEG() << ","
                         << state.mass << ","
                         << target_pitch_deg // Added Log
                         << "\n";
            }

            // Console Output (Reduced Frequency: 5s to avoid scroll spam)
            if (i % 500 == 0) {
                // Clear current line to remove progress bar artifact
                std::cout << "\r" << std::string(100, ' ') << "\r";
                
                std::cout << std::fixed << std::setprecision(2) 
                          << t << " | " 
                          << current_lla.alt << " | " 
                          << state.vel_ecef.norm() << " | " 
                          << mach << " | "
                          << (int)guidance.get_phase() << " | "
                          << pitch_deg << " (Tgt: " << target_pitch_deg << ") | "
                          << prop.force_body.norm() << " | "
                          << t_body_y << " | "
                          << m_body_y << " | "
                          << pitch_err << std::endl;
                
                // Force redraw of progress bar on next update
                sim_progress.update(i);
            }
        }

        sim_progress.update(i);

        // Check for Ground Collision
        if (lla_check.alt < 0.0) {
            std::cout << "[Sim] IMPACT DETECTED at t=" << t << "s" << std::endl;
            
            // Recalculate for logging
            AtmosphereData atm = AtmosphereModel::calculate(get_atmosphere_input(t, lla_check));
            double mach = state.vel_ecef.norm() / atm.sound_speed;
            
            Eigen::Vector3d up = state.pos_ecef.normalized();
            Eigen::Vector3d body_x = state.quat_be * Eigen::Vector3d::UnitX();
            double dot = body_x.dot(up);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            double pitch_deg = 90.0 - std::acos(dot) * AeroSim::Math::RAD2DEG();
            
            PropulsionOutput prop = motor.compute(t, state.mass, atm.pressure, missile.current_cmd.tvc, inertia.com);
            
            Eigen::Quaterniond guid_target = guidance.update(t, state.pos_ecef, state.vel_ecef);
            Eigen::Quaterniond q_err = state.quat_be.inverse() * guid_target;
            double pitch_err = 2.0 * q_err.y();
            
            Eigen::Vector3d target_x = guid_target * Eigen::Vector3d::UnitX();
            double target_dot = target_x.dot(up);
            if (target_dot > 1.0) target_dot = 1.0;
            if (target_dot < -1.0) target_dot = -1.0;
            double target_pitch_deg = 90.0 - std::acos(target_dot) * AeroSim::Math::RAD2DEG();

            if (csv_file.is_open()) {
                csv_file << t << "," 
                         << 0.0 << "," 
                         << state.vel_ecef.norm() << "," 
                         << mach << ","
                         << (int)guidance.get_phase() << ","
                         << pitch_deg << ","
                         << prop.force_body.norm() << ","
                         << prop.force_body.y() << ","
                         << prop.moment_body.y() << ","
                         << pitch_err << ","
                         << lla_check.lat * AeroSim::Math::RAD2DEG() << ","
                         << lla_check.lon * AeroSim::Math::RAD2DEG() << ","
                         << state.mass << ","
                         << target_pitch_deg 
                         << "\n";
            }
            break;
        }

        // CONTROL STEP (Execute before integration)
        // Update Guidance and Autopilot once per step
        Eigen::Quaterniond guid_target = guidance.update(t, state.pos_ecef, state.vel_ecef);
        missile.current_cmd = autopilot.update(state.quat_be, state.omega_body, guid_target, dt);
        
        // Note: Autopilot handles TVC sign inversion internally.
        // No manual inversion needed here.

        // INTEGRATION STEP
        // Now calling the new template-based RK4 which re-evaluates forces internally
        state = Dynamics6DOF::integrate_rk4(state, missile, t, dt);
        
        // Update time
        t += dt;
    }

    sim_progress.finish();
    if (csv_file.is_open()) {
        csv_file.close();
        std::cout << "[Sim] Data saved to simulation_data.csv" << std::endl;
    }
    std::cout << "[Sim] Simulation complete." << std::endl;
    
    // Final State
    std::cout << "Final State | Time: " << t << "s" << std::endl;
    std::cout << "Alt: " << CoordinateTransform::ecef_to_lla(state.pos_ecef).alt << " m" << std::endl;
    std::cout << "Vel: " << state.vel_ecef.norm() << " m/s" << std::endl;
    
    return 0;
}
