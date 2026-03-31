#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h> // For CPU parallel tasks if needed (but we focus on GPU gravity)

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
#include "missile_config.hpp"

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

// Functor for system dynamics - Per Trajectory Instance
struct MissileSystem {
    SolidMotor& motor;         // Shared
    RCSModel& rcs;             // Shared
    AerodynamicsModel& aero;   // Shared
    InertialProps& initial_inertia; // Shared (Initial props)
    double min_mass;

    // Per-Step State
    Eigen::Vector3d cached_gravity; // Gravity acceleration in ECEF (m/s^2)
    Autopilot::AutopilotOutput current_cmd = {{0.0, 0.0}, {0.0, 0.0, 0.0, false}, {0.0, 0.0, 0.0}};

    // Constructor
    MissileSystem(SolidMotor& m, RCSModel& r, AerodynamicsModel& a, InertialProps& i, double mm) 
        : motor(m), rcs(r), aero(a), initial_inertia(i), min_mass(mm) {
        cached_gravity.setZero();
    }
        
    void set_gravity(const Eigen::Vector3d& g) {
        cached_gravity = g;
    }

    // The operator() required by RK4 integration
    State6DOF operator()(const State6DOF& state, double t) {
        // 1. Gravity (Pre-calculated)
        Eigen::Vector3d g_ecef = cached_gravity;
        
        // 2. Atmosphere
        LLA lla = CoordinateTransform::ecef_to_lla(state.pos_ecef);
        // Use USSA76 for speed in Monte Carlo, or NRLMSISE-00 if precision needed
        // For 100 trajectories, USSA76 is much faster.
        // But let's use NRLMSISE-00 for the first trajectory (detailed logging) and USSA76 for others?
        // To keep dynamics consistent, use USSA76 for all or NRLMSISE-00 for all.
        // USSA76 is ~100x faster than NRLMSISE-00.
        // Let's use USSA76 for performance as requested ("reduce performance overhead").
        AtmosphereData atm = AtmosphereModel::calculate_ussa76(lla.alt);

        // 3. Aerodynamics
        ForcesMoments fm;
        fm.force_body = Eigen::Vector3d::Zero();
        fm.moment_body = Eigen::Vector3d::Zero();
        fm.mass_flow_rate = 0.0;
        
        Eigen::Vector3d v_air_ecef = state.vel_ecef; // Assume no wind
        Eigen::Vector3d v_air_body = state.quat_be.inverse() * v_air_ecef;
        
        double v_mag = v_air_body.norm();
        double mach = v_mag / atm.sound_speed;
        double dynamic_pressure = 0.5 * atm.density * v_mag * v_mag;
        
        if (v_mag > 1.0) {
            double u = v_air_body.x();
            double v = v_air_body.y();
            double w = v_air_body.z();
            
            double alpha = std::atan2(w, u);
            double beta = std::asin(v / v_mag);
            
            auto aero_forces = aero.compute_forces_moments(
                dynamic_pressure, mach, alpha, beta, initial_inertia.com,
                current_cmd.aero.pitch, current_cmd.aero.yaw, current_cmd.aero.roll
            );
            
            fm.force_body += aero_forces.first;
            fm.moment_body += aero_forces.second;
        }

        // 4. Propulsion
        PropulsionOutput prop_main = motor.compute(t, state.mass, atm.pressure, current_cmd.tvc, initial_inertia.com);
        PropulsionOutput prop_rcs = rcs.compute(current_cmd.rcs, state.mass, min_mass);
        PropulsionOutput prop_total = prop_main + prop_rcs;
        
        fm.force_body += prop_total.force_body;
        fm.moment_body += prop_total.moment_body;
        fm.mass_flow_rate = prop_total.mass_flow_rate;
        
        // Update inertia
        InertialProps current_inertia = initial_inertia;
        current_inertia.mass = state.mass;
        if (initial_inertia.mass > 0) {
            double ratio = state.mass / initial_inertia.mass;
            current_inertia.inertia = initial_inertia.inertia * ratio;
        }
        
        // 5. Derivatives
        return Dynamics6DOF::compute_derivatives(state, fm, current_inertia, g_ecef, SimulationProfile::GLOBAL_BALLISTIC);
    }
};

struct Trajectory {
    int id;
    bool active = true;
    State6DOF state;
    Autopilot autopilot;
    Guidance guidance;
    MissileSystem system;
    
    // Stats
    double max_alt = 0.0;
    double max_speed = 0.0;
    double range = 0.0;

    Trajectory(int i, const State6DOF& s, const Autopilot::Config& ap_cfg, const Guidance::Config& gd_cfg,
               SolidMotor& m, RCSModel& r, AerodynamicsModel& a, InertialProps& ip, double mm)
        : id(i), state(s), autopilot(ap_cfg), guidance(gd_cfg), system(m, r, a, ip, mm) {}
};

int main(int argc, char** argv) {
    std::cout << "===============================================" << std::endl;
    std::cout << "  AeroSim - Multi-Trajectory Monte Carlo Mode  " << std::endl;
    std::cout << "===============================================" << std::endl;

    // Config
    int num_trajectories = 100; // Monte Carlo Count
    double t_end = 3000.0;
    
    // Load Models
    std::cout << "[Init] Loading Gravity Model (Batch GPU Enabled)..." << std::endl;
    GravityModel gravity(4); // J2-J4
    if (!gravity.load_coefficients("e:/missile/data/EGM2008.gfc")) {
        std::cerr << "[Warning] EGM2008.gfc not found, using default." << std::endl;
    }

    MissileDesign::HGV1Config hgv_config = MissileDesign::load_hgv1_config();
    SolidMotor motor(hgv_config.propulsion);
    RCSModel::Config rcs_cfg;
    rcs_cfg.max_thrust = 5000.0;
    rcs_cfg.isp = 220.0;
    rcs_cfg.lever_arm_x = 8.0;
    rcs_cfg.lever_arm_r = 0.75;
    RCSModel rcs(rcs_cfg);
    AerodynamicsModel aero(hgv_config.aerodynamics);
    
    // Initial State Template
    LLA init_lla = {40.960556 * AeroSim::Math::DEG2RAD(), 100.298333 * AeroSim::Math::DEG2RAD(), 1000.0}; 
    Eigen::Vector3d init_pos = CoordinateTransform::lla_to_ecef(init_lla);
    
    // Orientation
    Eigen::Vector3d up_init = init_pos.normalized();
    Eigen::Vector3d earth_z(0,0,1);
    Eigen::Vector3d north_init = (earth_z - (earth_z.dot(up_init) * up_init)).normalized();
    Eigen::Vector3d east_init = north_init.cross(up_init).normalized();
    Eigen::Matrix3d init_rot;
    init_rot.col(0) = up_init;
    init_rot.col(1) = -north_init;
    init_rot.col(2) = east_init;
    
    State6DOF base_state;
    base_state.pos_ecef = init_pos;
    base_state.vel_ecef = Eigen::Vector3d::Zero();
    base_state.quat_be = Eigen::Quaterniond(init_rot);
    base_state.omega_body.setZero();
    base_state.mass = hgv_config.total_mass;
    
    // Inertial Properties (Cylinder approximation)
    InertialProps inertia;
    inertia.mass = base_state.mass;
    double radius = 0.26; // 0.52m diameter
    double length = 6.0;
    double Ix = 0.5 * inertia.mass * radius * radius;
    double Iy = (1.0/12.0) * inertia.mass * (3*radius*radius + length*length);
    inertia.inertia = Eigen::Matrix3d::Zero();
    inertia.inertia(0,0) = Ix;
    inertia.inertia(1,1) = Iy;
    inertia.inertia(2,2) = Iy;
    inertia.com = Eigen::Vector3d(3.0, 0, 0); // Move CG to mid-body (3.0m) for stability

    // Guidance Config
    GNC::Guidance::Config guid_cfg;
    guid_cfg.boost_end_time = 50.0;
    guid_cfg.boost_pitch_start = 10.0;
    guid_cfg.boost_pitch_rate = 1.0;
    guid_cfg.boost_pitch_min = 30.0;
    guid_cfg.glide_alt_start = 50000.0;
    guid_cfg.glide_alt_end = 20000.0;
    guid_cfg.glide_vel_min = 800.0;
    guid_cfg.target_range = 4000000.0;
    LLA target_lla = {20.0 * AeroSim::Math::DEG2RAD(), 135.0 * AeroSim::Math::DEG2RAD(), 0.0};
    Eigen::Vector3d target_ecef = CoordinateTransform::lla_to_ecef(target_lla);

    // Initialize Trajectories with Dispersion
    std::cout << "[Init] Initializing " << num_trajectories << " trajectories..." << std::endl;
    std::vector<Trajectory> trajectories;
    trajectories.reserve(num_trajectories);
    
    std::mt19937 rng(42);
    std::normal_distribution<double> dist_pos(0.0, 10.0); // 10m position error
    std::normal_distribution<double> dist_mass(0.0, 50.0); // 50kg mass error
    std::normal_distribution<double> dist_pitch(0.0, 0.5); // 0.5 deg pitch error

    for (int i = 0; i < num_trajectories; ++i) {
        State6DOF s = base_state;
        // Apply Dispersion
        s.pos_ecef += Eigen::Vector3d(dist_pos(rng), dist_pos(rng), dist_pos(rng));
        s.mass += dist_mass(rng);
        
        // Apply pitch dispersion to orientation
        Eigen::AngleAxisd pitch_err(dist_pitch(rng) * AeroSim::Math::DEG2RAD(), Eigen::Vector3d::UnitY());
        s.quat_be = s.quat_be * pitch_err;

        GNC::Guidance::Config g_cfg = guid_cfg; // Copy config
        // Could disperse guidance parameters here too

        Autopilot::Config ap_cfg = hgv_config.autopilot;

        Trajectory t(i, s, ap_cfg, g_cfg, motor, rcs, aero, inertia, hgv_config.propulsion.dry_mass);
        t.guidance.set_target(target_ecef);
        trajectories.push_back(std::move(t));
    }

    // Simulation Loop
    double t = 0.0;
    double dt = 0.002; // 500 Hz for stability during boost/transonic
    int active_count = num_trajectories;
    
    // Performance metrics
    auto sim_start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Sim] Starting Batch Simulation (Fixed dt=" << dt << "s)..." << std::endl;
    ProgressBar sim_progress(100, 50, "Monte Carlo");

    // Logging Buffer
    struct LogPoint {
        double t;
        double alt;
        double vel;
        double mach;
        int phase;
        double pitch;
        double mass;
        double thrust;
    };
    std::vector<std::vector<LogPoint>> trajectory_logs(num_trajectories);
    // Reserve memory (approx 3000s / 0.1s = 30000 points)
    for(auto& log : trajectory_logs) log.reserve(30000);

    motor.ignite(0.1); // Ignite all at 0.1s

    while (t < t_end && active_count > 0) {
        active_count = 0;
        
        // 1. GNC Update & Status Check
        for (int i=0; i<num_trajectories; ++i) {
            auto& traj = trajectories[i];
            if (!traj.active) continue;
            
            LLA lla = CoordinateTransform::ecef_to_lla(traj.state.pos_ecef);
            
            // Stats
            if (lla.alt > traj.max_alt) traj.max_alt = lla.alt;
            double v = traj.state.vel_ecef.norm();
            if (v > traj.max_speed) traj.max_speed = v;

            // Ground Collision
            if (lla.alt < 0.0) {
                traj.active = false;
                traj.range = (CoordinateTransform::lla_to_ecef(lla) - init_pos).norm();
                continue;
            }
            active_count++;

            // GNC
            Eigen::Quaterniond guid_target = traj.guidance.update(t, traj.state.pos_ecef, traj.state.vel_ecef);
            traj.system.current_cmd = traj.autopilot.update(traj.state.quat_be, traj.state.omega_body, guid_target, dt);

            // Logging (Every 0.1s)
            if (std::fmod(t, 0.1) < dt) {
                 double mach = v / 340.0; // Approx
                 Eigen::Vector3d up = traj.state.pos_ecef.normalized();
                 Eigen::Vector3d body_x = traj.state.quat_be * Eigen::Vector3d::UnitX();
                 double pitch = 90.0 - std::acos(std::clamp(body_x.dot(up), -1.0, 1.0)) * 180.0/3.14159;
                 
                 // Estimate Thrust (not stored in state, need to re-compute or cache? Just assume motor curve for now or skip)
                 double thrust = 0.0; // Placeholder
                 
                 trajectory_logs[i].push_back({t, lla.alt, v, mach, (int)traj.guidance.get_phase(), pitch, traj.state.mass, thrust});
            }
        }

        if (active_count == 0) break;

        // 2. Physics Integration (RK4 Unrolled with Batch Gravity)
        // Only process active trajectories to save time? 
        // Or keep indices consistent? Keeping consistent is easier for batch vector construction.
        // We'll build a list of pointers or indices to active trajectories.
        std::vector<int> active_indices;
        active_indices.reserve(num_trajectories);
        for(int i=0; i<num_trajectories; ++i) {
            if(trajectories[i].active) active_indices.push_back(i);
        }
        
        int N = active_indices.size();
        std::vector<Eigen::Vector3d> positions(N);

        // --- Stage 1 ---
        for(int i=0; i<N; ++i) positions[i] = trajectories[active_indices[i]].state.pos_ecef;
        auto grav1 = gravity.calculate_accelerations_cuda(positions);
        
        std::vector<State6DOF> k1(N);
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            trajectories[idx].system.set_gravity(grav1[i]);
            k1[i] = trajectories[idx].system(trajectories[idx].state, t);
        }

        // --- Stage 2 ---
        std::vector<State6DOF> temp_states(N);
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            State6DOF s = trajectories[idx].state + k1[i] * (0.5 * dt);
            s.normalize();
            temp_states[i] = s;
            positions[i] = s.pos_ecef;
        }
        auto grav2 = gravity.calculate_accelerations_cuda(positions);

        std::vector<State6DOF> k2(N);
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            trajectories[idx].system.set_gravity(grav2[i]);
            k2[i] = trajectories[idx].system(temp_states[i], t + 0.5*dt);
        }

        // --- Stage 3 ---
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            State6DOF s = trajectories[idx].state + k2[i] * (0.5 * dt);
            s.normalize();
            temp_states[i] = s;
            positions[i] = s.pos_ecef;
        }
        auto grav3 = gravity.calculate_accelerations_cuda(positions);

        std::vector<State6DOF> k3(N);
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            trajectories[idx].system.set_gravity(grav3[i]);
            k3[i] = trajectories[idx].system(temp_states[i], t + 0.5*dt);
        }

        // --- Stage 4 ---
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            State6DOF s = trajectories[idx].state + k3[i] * dt;
            s.normalize();
            temp_states[i] = s;
            positions[i] = s.pos_ecef;
        }
        auto grav4 = gravity.calculate_accelerations_cuda(positions);

        std::vector<State6DOF> k4(N);
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            trajectories[idx].system.set_gravity(grav4[i]);
            k4[i] = trajectories[idx].system(temp_states[i], t + dt);
        }

        // --- Update ---
        for(int i=0; i<N; ++i) {
            int idx = active_indices[i];
            trajectories[idx].state = trajectories[idx].state + (k1[i] + k2[i]*2.0 + k3[i]*2.0 + k4[i]) * (dt / 6.0);
            trajectories[idx].state.normalize();
        }

        t += dt;
        sim_progress.update((size_t)((t / t_end) * 100.0));
    }
    
    sim_progress.finish();

    // Report & CSV Export
    std::cout << "[Sim] Monte Carlo Complete." << std::endl;
    
    // 1. Compute Stats & Find Best Trajectory
    double mean_range = 0.0;
    double max_range = 0.0;
    int success_count = 0;
    int best_idx = 0;
    
    for(int i=0; i<num_trajectories; ++i) {
        const auto& tr = trajectories[i];
        if (tr.range > 1000.0) { // Filter out launch failures
            mean_range += tr.range;
            if (tr.range > max_range) {
                max_range = tr.range;
                best_idx = i;
            }
            success_count++;
        }
    }
    if (success_count > 0) mean_range /= success_count;
    
    std::cout << "Success Count: " << success_count << "/" << num_trajectories << std::endl;
    std::cout << "Mean Range: " << mean_range / 1000.0 << " km" << std::endl;
    std::cout << "Best Range: " << max_range / 1000.0 << " km (Trajectory " << best_idx << ")" << std::endl;

    // 2. Export Best Trajectory
    std::ofstream csv_file("simulation_data.csv");
    if (csv_file.is_open()) {
        csv_file << "Time,Alt,Vel,Mach,Phase,Pitch,Thrust,Mass\n";
        const auto& best_log = trajectory_logs[best_idx];
        for(const auto& pt : best_log) {
            csv_file << pt.t << "," << pt.alt << "," << pt.vel << "," << pt.mach << "," 
                     << pt.phase << "," << pt.pitch << "," << pt.thrust << "," << pt.mass << "\n";
        }
        csv_file.close();
        std::cout << "[Output] Best trajectory data saved to simulation_data.csv" << std::endl;
    }
    std::cout << "Max Range: " << max_range / 1000.0 << " km" << std::endl;

    return 0;
}
