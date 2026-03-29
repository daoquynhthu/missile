#include "missile_config.hpp"
#include "propulsion_model.hpp"
#include "aerodynamics_model.hpp"
#include <cmath>
#include <vector>

namespace AeroSim {
namespace MissileDesign {

    HGV1Config load_hgv1_config() {
        HGV1Config config;
        
        // --- Basic Parameters ---
        config.name = "IRBM (Optimized Geometry)";
        // Total Mass 15000 kg (IRBM Class)
        config.total_mass = 15000.0; 
        // Payload Mass 1000 kg
        config.payload_mass = 1000.0;

        // --- Inertia Calculation ---
        // Simplified Component Buildup:
        // 1. Payload (Point Mass) at Nose
        // 2. Motor (Hollow Cylinder) Main Body
        // 3. Fins (Cruciform Plates) at Tail
        
        double x_nose = 0.0;
        double x_tail = 12.0;
        double r_body = 0.6;
        
        // Component 1: Payload
        double m_payload = config.payload_mass; // 1000 kg
        double x_payload = 1.0; // 1m from nose
        
        // Component 2: Motor + Structure
        double m_motor = config.total_mass - m_payload; // 14000 kg
        double L_motor = 10.0;
        double x_motor = 2.0 + L_motor/2.0; // Centered at 7.0m
        
        // Component 3: Fins (Modeled as part of structure mass for now, but add specific inertia term)
        // Assume 4 fins, 50kg each, effective radius 1.2m
        double m_fin = 50.0; 
        double x_fin = 11.5; // Near tail
        // Adjust motor mass to keep total mass correct (subtract fins)
        m_motor -= 4 * m_fin; 
        
        // Calculate CG
        double cg_x = (m_payload * x_payload + m_motor * x_motor + 4 * m_fin * x_fin) / config.total_mass;
        config.center_of_mass = Eigen::Vector3d(cg_x, 0.0, 0.0);
        
        // Calculate Inertia Tensor about CG
        double Ixx = 0.0;
        double Iyy = 0.0;
        double Izz = 0.0;
        
        // 1. Payload
        double dx = x_payload - cg_x;
        Ixx += 0.0; // Point mass on axis
        Iyy += m_payload * dx * dx;
        Izz += m_payload * dx * dx;
        
        // 2. Motor (Cylinder)
        dx = x_motor - cg_x;
        double Ixx_cyl = 0.5 * m_motor * r_body * r_body;
        double Iyy_cyl = (1.0/12.0) * m_motor * (3*r_body*r_body + L_motor*L_motor);
        Ixx += Ixx_cyl;
        Iyy += Iyy_cyl + m_motor * dx * dx;
        Izz += Iyy_cyl + m_motor * dx * dx;
        
        // 3. Fins (4x Flat Plates)
        // Treat as point masses at radius r_fin_eff for Roll, and point masses at x_fin for Pitch/Yaw
        double r_fin_eff = 1.2; // Effective radius of gyration for fins
        dx = x_fin - cg_x;
        double Ixx_fin = 4 * m_fin * r_fin_eff * r_fin_eff; // Roll inertia from fins
        double Iyy_fin = 4 * m_fin * dx * dx; // Pitch inertia from fins (parallel axis)
        // Add fin self-inertia? (1/12 m c^2) - Negligible compared to parallel axis
        Ixx += Ixx_fin;
        Iyy += Iyy_fin;
        Izz += Iyy_fin; // Symmetric
        
        config.inertia_tensor = Eigen::Matrix3d::Zero();
        config.inertia_tensor(0,0) = Ixx;
        config.inertia_tensor(1,1) = Iyy;
        config.inertia_tensor(2,2) = Izz;

        // --- Propulsion System Design ---
        // IRBM Scale
        
        AeroSim::SolidMotor::Config& prop = config.propulsion;
        prop.payload_mass = config.payload_mass;
        prop.dry_mass = 1500.0; // Structure mass
        prop.propellant_mass = config.total_mass - config.payload_mass - prop.dry_mass; // 12500 kg propellant
        prop.casing_radius = 0.6; // Diameter 1.2m
        prop.casing_length = 10.0; 
        // Nozzle at the back (negative X)
        prop.nozzle_pos = Eigen::Vector3d(-12.0, 0.0, 0.0);
        prop.exit_area = 0.8; // Exit area
        prop.isp = 285.0; // Isp
        
        prop.burn_time = 60.0; // Longer burn for IRBM
        prop.total_impulse = prop.propellant_mass * prop.isp * 9.80665;
        
        prop.time_knots = {0.0, 5.0, 15.0, 50.0, 58.0, 60.0, 60.1};
        // Scaled Thrust to match Total Impulse (~35 MNs)
        prop.thrust_knots = {
            800000.0,  // t=0: Ignition (800kN)
            1200000.0, // t=5: Max Thrust (1200kN)
            800000.0,  // t=15: End Boost
            400000.0,  // t=50: Sustain
            100000.0,  // t=58: Tailing off
            0.0,       // t=60: Burnout
            0.0        // t=60.1: Off
        };

        // --- Aerodynamics Design (Standard Missile) ---
        // High L/D for hypersonic glide
        AeroSim::AerodynamicsModel::Config& aero = config.aerodynamics;
        
        // Updated values from Shape Optimizer (IRBM)
        aero.ref_area = 1.1310; // Reference Area (pi * 0.6^2)
        aero.ref_length = 12.0; // Reference Length (m)
        aero.ref_span = 3.0;    // Reference Span (m)
        
        // Use Pre-computed Aerodynamic Table for speed and accuracy
        aero.use_csv_table = true;
        aero.csv_path = "e:/missile/aerodynamics_table.csv";
        aero.use_cuda_solver = false; // Disable runtime solver
        aero.stl_path = "e:/missile/hgv_model_optimized.stl"; 

        // Mach Grid (0.0 to 25.0)
        aero.mach_grid = {0.0, 0.8, 1.2, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0};
        
        // Alpha Grid (degrees, -10 to 30)
        aero.alpha_grid = {-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0};

        // Generate Hypersonic Aero Data
        // Simplified model based on Newtonian impact theory
        
        int n_mach = aero.mach_grid.size();
        int n_alpha = aero.alpha_grid.size();
        
        aero.cl_table_2d.resize(n_mach * n_alpha);
        aero.cd_table_2d.resize(n_mach * n_alpha);
        aero.cm_table_2d.resize(n_mach * n_alpha); 

        for (int i = 0; i < n_mach; ++i) {
            double mach = aero.mach_grid[i];
            
            // Base Drag Coefficient
            double cd0_base = 0.02;
            if (mach < 0.8) cd0_base = 0.02;
            else if (mach < 1.2) cd0_base = 0.05; // Transonic wave drag
            else cd0_base = 0.04 * (1.0 / std::sqrt(mach*mach - 1.0)) + 0.01; // Supersonic decay

            // Lift Slope
            double cl_alpha_slope = 0.0;
            if (mach < 1.0) cl_alpha_slope = 0.1 * (180.0/3.14159); // ~5.7 per rad
            else cl_alpha_slope = (4.0 / std::sqrt(mach*mach - 1.0)); // Linearized supersonic

            for (int j = 0; j < n_alpha; ++j) {
                double alpha_deg = aero.alpha_grid[j];
                double alpha_rad = alpha_deg * 3.14159 / 180.0;
                
                // 1. Lift Coefficient Cl
                double cl = 0.0;
                if (mach < 5.0) {
                    cl = cl_alpha_slope * std::sin(alpha_rad);
                    if (std::abs(alpha_deg) > 15.0) cl *= 0.8; // Stall
                } else {
                    // Hypersonic Waverider Lift
                    // Cl approx K * alpha * cos(alpha)
                    double K_hypersonic = 1.2; 
                    cl = K_hypersonic * std::sin(2.0 * alpha_rad); 
                }
                
                // 2. Drag Coefficient Cd
                double k_ind = 0.0;
                if (mach < 1.0) k_ind = 0.05;
                else k_ind = 0.3 + 0.1 * mach; 
                
                double cd = cd0_base + std::abs(cl * std::sin(alpha_rad)) + 0.1 * (1.0 - std::cos(alpha_rad));
                
                // Ensure High L/D at Hypersonic
                if (mach > 5.0 && alpha_deg > 0.0) {
                     // L/D Correction if needed
                }

                // 3. Pitching Moment Coefficient Cm
                double static_margin = 0.05; // 5% length
                if (mach > 1.0) static_margin = 0.1 + 0.05 * mach; // Rearward shift
                if (static_margin > 0.2) static_margin = 0.2; 
                
                double cm = -static_margin * cl; 

                // Fill Tables
                int idx = i * n_alpha + j;
                aero.cl_table_2d[idx] = cl;
                aero.cd_table_2d[idx] = cd;
                aero.cm_table_2d[idx] = cm;
            }
        }
        
        // Center of Pressure Table (Normalized Length 0.0-1.0 from Nose)
        // Adjusted to 5.2 (5.2m from Nose) to reduce static margin.
        // CG is approx -4.8m. CP at -5.2m gives 0.4m margin (4% L), which is stable but maneuverable.
        // Previous value 6.0 (1.2m margin) was too stable, preventing pitch-up at high altitude.
        aero.xcp_table = {5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2};

        // Control Derivatives (per radian)
        aero.cm_delta_pitch = -0.8; // Pitch moment due to elevator (negative for stability)
        aero.cn_delta_yaw = -0.5;   // Yaw moment due to rudder
        aero.cl_delta_roll = 0.5;   // Roll moment due to aileron

        // Autopilot Config
        AeroSim::GNC::Autopilot::Config ap_cfg;
        
        // Pitch PID
        // Tuned for HGV-1 (High Authority TVC & Aero)
        // Reduced gains to prevent Phase 0 instability/oscillation
        ap_cfg.pitch_pid.kp = 1.0;   // Reduced from 2.0
        ap_cfg.pitch_pid.ki = 0.05;  // Reduced from 0.1
        ap_cfg.pitch_pid.kd = 1.0;   // Reduced from 3.0
        ap_cfg.pitch_pid.output_min = -0.5; // -28 deg
        ap_cfg.pitch_pid.output_max = 0.5;  // +28 deg
        ap_cfg.pitch_pid.integrator_min = -0.2;
        ap_cfg.pitch_pid.integrator_max = 0.2;
        
        // Yaw PID (Same as Pitch for symmetric missile)
        ap_cfg.yaw_pid = ap_cfg.pitch_pid;
        
        // Aero Control PID (Elevator/Rudder/Aileron)
        // These are critical for Glide Phase where TVC is off
        ap_cfg.aero_pitch_pid.kp = 5.0;  // Stronger response for aero surfaces (Increased from 3.0)
        ap_cfg.aero_pitch_pid.ki = 0.2;  // Increased from 0.05 to eliminate steady-state error
        ap_cfg.aero_pitch_pid.kd = 1.5;  // Damping
        ap_cfg.aero_pitch_pid.output_min = -0.5; // -28 deg deflection limit
        ap_cfg.aero_pitch_pid.output_max = 0.5;
        ap_cfg.aero_pitch_pid.integrator_min = -0.1;
        ap_cfg.aero_pitch_pid.integrator_max = 0.1;

        ap_cfg.aero_yaw_pid = ap_cfg.aero_pitch_pid;

        ap_cfg.aero_roll_pid.kp = 2.0;
        ap_cfg.aero_roll_pid.ki = 0.0;
        ap_cfg.aero_roll_pid.kd = 0.5;
        ap_cfg.aero_roll_pid.output_min = -0.5;
        ap_cfg.aero_roll_pid.output_max = 0.5;

        // RCS PID (Higher gains for vacuum operation)
        ap_cfg.rcs_pitch_pid.kp = 2.5;
        ap_cfg.rcs_pitch_pid.ki = 0.0;
        ap_cfg.rcs_pitch_pid.kd = 5.0;
        ap_cfg.rcs_pitch_pid.output_min = -1.0;
        ap_cfg.rcs_pitch_pid.output_max = 1.0;
        
        ap_cfg.rcs_yaw_pid = ap_cfg.rcs_pitch_pid;
        
        ap_cfg.rcs_roll_pid.kp = 5.0;
        ap_cfg.rcs_roll_pid.ki = 0.0;
        ap_cfg.rcs_roll_pid.kd = 2.5;
        ap_cfg.rcs_roll_pid.output_min = -1.0;
        ap_cfg.rcs_roll_pid.output_max = 1.0;
        
        // Roll PID (Unused for TVC but good to init)
        ap_cfg.roll_pid.kp = 2.0;
        ap_cfg.roll_pid.ki = 0.0;
        ap_cfg.roll_pid.kd = 0.5;
        ap_cfg.roll_pid.output_min = -1.0;
        ap_cfg.roll_pid.output_max = 1.0;
        
        config.autopilot = ap_cfg;

        return config;
    }
}
}
