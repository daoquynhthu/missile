#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>
#include <utility> // for std::pair
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include <Eigen/Core>

#include "aero/panel/aero_solver.hpp"
#include "infra/util/utils.hpp"

namespace aerosp {

// Unified Coefficient Structure
struct AeroCoeffs {
    double CX; // Body X Force Coeff
    double CY; // Body Y Force Coeff
    double CZ; // Body Z Force Coeff
    double Cl; // Rolling Moment Coeff
    double Cm; // Pitching Moment Coeff
    double Cn; // Yawing Moment Coeff
    double CD; // Drag Coeff (Wind)
    double CL; // Lift Coeff (Wind)
};

/**
 * @brief Aerodynamic Model with Mach-dependent coefficients
 * Configurable to use either Analytical/Table-based or GPU-Solver-based methods.
 */
class AerodynamicsModel {
public:
    struct Config {
        double ref_area = 1.0;    // Reference Area (m^2)
        double ref_length = 1.0;  // Reference Length (m)
        double ref_span = 1.0;    // Reference Span (m)
        
        // --- 3D Modeling Configuration ---
        double body_length = 10.0;    // Total length (m)
        double body_width = 3.0;     // Max width (m)
        double body_height = 1.2;    // Max height (m)
        double sweep_angle = 75.0;   // Sweep angle (deg)
        double nose_radius = 0.05;   // Blunting radius (m)
        
        // --- Solver Configuration ---
        bool use_cuda_solver = true; // Enable/Disable GPU solver
        std::string stl_path = "hgv_model.stl";
        
        // --- Table Lookup Configuration ---
        bool use_csv_table = false;
        std::string csv_path = "aerodynamics_table.csv";

        // Control Derivatives (per radian)
        double cm_delta_pitch = -0.8; // Pitch moment due to elevator
        double cn_delta_yaw = -0.5;   // Yaw moment due to rudder
        double cl_delta_roll = 0.5;   // Roll moment due to aileron
        
        // 1D Tables (Mach dependent) for Analytical Fallback
        std::vector<double> mach_grid;
        std::vector<double> cd0_table;
        std::vector<double> cl_alpha_table;
        std::vector<double> xcp_table; // Added back for compatibility

        // 2D Tables (Mach, Alpha) - For High Fidelity Analytical Fallback
        std::vector<double> alpha_grid;
        std::vector<double> cl_table_2d; // Flattened [mach][alpha]
        std::vector<double> cd_table_2d;
        std::vector<double> cm_table_2d;
    };

    struct TableData {
        std::vector<double> mach_grid;
        std::vector<double> alpha_grid;
        std::map<std::pair<int, int>, AeroCoeffs> data_map; // (mach_idx, alpha_idx) -> Coeffs
        bool loaded = false;
    };

    AerodynamicsModel(const Config& config) : m_config(config) {
        if (m_config.use_csv_table) {
            load_csv_table(m_config.csv_path);
        }

        if (m_config.use_cuda_solver && !m_table.loaded) {
            m_solver = std::make_unique<aero::panel::AeroSolver>();
            if (!m_solver->load_model(m_config.stl_path, m_config.ref_area, m_config.ref_length, m_config.ref_span)) {
                std::cerr << "Warning: Failed to load STL model " << m_config.stl_path 
                          << ". Falling back to analytical model." << std::endl;
                m_config.use_cuda_solver = false; 
            }
        }
    }

    void load_csv_table(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open aerodynamic table: " << path << std::endl;
            return;
        }

        std::string line;
        // Skip header
        std::getline(file, line);

        std::vector<double> unique_machs;
        std::vector<double> unique_alphas;
        struct Entry { double m; double a; AeroCoeffs c; };
        std::vector<Entry> entries;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<double> row;
            while (std::getline(ss, token, ',')) {
                try {
                    row.push_back(std::stod(token));
                } catch (...) {
                    row.push_back(0.0);
                }
            }
            if (row.size() >= 12) {
                // Full format: mach,alpha,beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn[,Fidelity]
                // Fidelity column (index 12) is informational; all coefficients
                // are stored uniformly regardless of source.
                double m = row[0];
                double a = row[1];
                
                AeroCoeffs c;
                c.CX = row[3];
                c.CY = row[4];
                c.CZ = row[5];
                c.CL = row[6];
                c.CD = row[7];
                c.Cl = row[9];
                c.Cm = row[10];
                c.Cn = row[11];

                entries.push_back({m, a, c});
                unique_machs.push_back(m);
                unique_alphas.push_back(a);
            } else if (row.size() >= 6) {
                // Simplified format: Mach,Alpha,CL,CD,Cm,L_D
                double m = row[0];
                double a = row[1]; // Degrees in CSV
                
                // Convert alpha to radians for transformation
                double alpha_rad = a * 3.14159265359 / 180.0;
                
                double cl = row[2];
                double cd = row[3];
                double cm = row[4];
                
                // Compute Body Frame Forces
                // Drag is opposite to Velocity Vector
                // Lift is perpendicular to Velocity Vector (in lift plane)
                // Body X is Nose.
                // Velocity Vector V_inf is at angle alpha to Body X.
                // Drag D is along V_inf.
                // Lift L is perpendicular to V_inf.
                
                // Force along Body X (Axial): Fx = L sin(a) - D cos(a)
                // Force along Body Z (Normal): Fz = -L cos(a) - D sin(a)
                // (Using standard aerospace convention where Z is down, Lift is Up (-Z_wind))
                
                // CX = CL * sin(alpha) - CD * cos(alpha)
                // CZ = -CL * cos(alpha) - CD * sin(alpha)
                
                double cx = cl * std::sin(alpha_rad) - cd * std::cos(alpha_rad);
                double cz = -cl * std::cos(alpha_rad) - cd * std::sin(alpha_rad);
                
                AeroCoeffs c;
                c.CX = cx;
                c.CY = 0.0;
                c.CZ = cz;
                c.CL = cl;
                c.CD = cd;
                c.Cl = 0.0;
                c.Cm = cm;
                c.Cn = 0.0;
                
                entries.push_back({m, a, c});
                unique_machs.push_back(m);
                unique_alphas.push_back(a);
            }
        }

        // Sort and remove duplicates
        std::sort(unique_machs.begin(), unique_machs.end());
        unique_machs.erase(std::unique(unique_machs.begin(), unique_machs.end()), unique_machs.end());
        
        std::sort(unique_alphas.begin(), unique_alphas.end());
        unique_alphas.erase(std::unique(unique_alphas.begin(), unique_alphas.end()), unique_alphas.end());

        m_table.mach_grid = unique_machs;
        m_table.alpha_grid = unique_alphas;

        // Fill map
        for (const auto& e : entries) {
            // Find indices (using simple search or map, but here we need grid indices)
            auto it_m = std::lower_bound(m_table.mach_grid.begin(), m_table.mach_grid.end(), e.m);
            auto it_a = std::lower_bound(m_table.alpha_grid.begin(), m_table.alpha_grid.end(), e.a);
            
            if (it_m != m_table.mach_grid.end() && it_a != m_table.alpha_grid.end() && 
                std::abs(*it_m - e.m) < 1e-4 && std::abs(*it_a - e.a) < 1e-4) {
                
                int idx_m = std::distance(m_table.mach_grid.begin(), it_m);
                int idx_a = std::distance(m_table.alpha_grid.begin(), it_a);
                m_table.data_map[{idx_m, idx_a}] = e.c;
            }
        }

        m_table.loaded = true;
        std::cout << "Loaded Aerodynamic Table: " << m_table.mach_grid.size() << " Mach points, " 
                  << m_table.alpha_grid.size() << " Alpha points." << std::endl;
    }

    AeroCoeffs compute_coeffs_table(double mach, double alpha) const {
        if (!m_table.loaded) return AeroCoeffs{};

        // Clamp inputs
        mach = std::max(m_table.mach_grid.front(), std::min(m_table.mach_grid.back(), mach));
        alpha = std::max(m_table.alpha_grid.front(), std::min(m_table.alpha_grid.back(), alpha));

        // Find indices
        auto it_m = std::lower_bound(m_table.mach_grid.begin(), m_table.mach_grid.end(), mach);
        int i = std::distance(m_table.mach_grid.begin(), it_m);
        if (i > 0 && (it_m == m_table.mach_grid.end() || *it_m > mach)) i--;

        auto it_a = std::lower_bound(m_table.alpha_grid.begin(), m_table.alpha_grid.end(), alpha);
        int j = std::distance(m_table.alpha_grid.begin(), it_a);
        if (j > 0 && (it_a == m_table.alpha_grid.end() || *it_a > alpha)) j--;

        // Indices for interpolation
        int i0 = i, i1 = std::min(i + 1, (int)m_table.mach_grid.size() - 1);
        int j0 = j, j1 = std::min(j + 1, (int)m_table.alpha_grid.size() - 1);

        double m0 = m_table.mach_grid[i0];
        double m1 = m_table.mach_grid[i1];
        double a0 = m_table.alpha_grid[j0];
        double a1 = m_table.alpha_grid[j1];

        // Weights
        double tm = (i1 == i0) ? 0.0 : (mach - m0) / (m1 - m0);
        double ta = (j1 == j0) ? 0.0 : (alpha - a0) / (a1 - a0);

        // Fetch corner values
        auto get_coeff = [&](int mi, int ai) {
            auto it = m_table.data_map.find({mi, ai});
            if (it != m_table.data_map.end()) return it->second;
            return AeroCoeffs{}; // Should not happen if grid is complete
        };

        AeroCoeffs c00 = get_coeff(i0, j0);
        AeroCoeffs c10 = get_coeff(i1, j0);
        AeroCoeffs c01 = get_coeff(i0, j1);
        AeroCoeffs c11 = get_coeff(i1, j1);

        // Bilinear Interpolation
        auto interp = [&](double v00, double v10, double v01, double v11) {
            return (1-tm)*(1-ta)*v00 + tm*(1-ta)*v10 + (1-tm)*ta*v01 + tm*ta*v11;
        };

        AeroCoeffs res;
        res.CX = interp(c00.CX, c10.CX, c01.CX, c11.CX);
        res.CY = interp(c00.CY, c10.CY, c01.CY, c11.CY);
        res.CZ = interp(c00.CZ, c10.CZ, c01.CZ, c11.CZ);
        res.CL = interp(c00.CL, c10.CL, c01.CL, c11.CL);
        res.CD = interp(c00.CD, c10.CD, c01.CD, c11.CD);
        res.Cl = interp(c00.Cl, c10.Cl, c01.Cl, c11.Cl);
        res.Cm = interp(c00.Cm, c10.Cm, c01.Cm, c11.Cm);
        res.Cn = interp(c00.Cn, c10.Cn, c01.Cn, c11.Cn);

        return res;
    }

    AeroCoeffs compute_coeffs(double mach, double alpha, double beta = 0.0) const {
        if (m_config.use_csv_table && m_table.loaded) {
            return compute_coeffs_table(mach, alpha * 180.0 / 3.14159265359); // Table uses degrees
        }

        if (m_config.use_cuda_solver && m_solver) {
            // Use CUDA Solver
            aerosp::aero::panel::AeroCoefficients res = m_solver->compute_coefficients(
                static_cast<float>(mach), 
                static_cast<float>(alpha * 180.0 / 3.14159265359),
                static_cast<float>(beta * 180.0 / 3.14159265359)
            );
            
            return {
                res.CX, res.CY, res.CZ,
                res.Cl, res.Cm, res.Cn,
                res.CD, res.CL
            };
        }

        // Fallback to Analytical Model
        return compute_coeffs_analytical(mach, alpha, beta);
    }
    
    AeroCoeffs compute_coeffs_analytical(double mach, double alpha, double beta) const {
        // Simple Analytical Model
        // CD = CD0 + k * alpha^2
        // CL = CL_alpha * alpha
        // Cm = Cm0 + Cm_alpha * alpha
        
        double cd0 = 0.1;
        double cl_alpha = 2.0;
        
        if (!m_config.cd0_table.empty() && !m_config.mach_grid.empty()) {
            cd0 = aerosp::infra::util::interpolate_1d(m_config.mach_grid, m_config.cd0_table, mach);
        }
        if (!m_config.cl_alpha_table.empty() && !m_config.mach_grid.empty()) {
            cl_alpha = aerosp::infra::util::interpolate_1d(m_config.mach_grid, m_config.cl_alpha_table, mach);
        }

        double cl = cl_alpha * std::sin(alpha); // Better than linear for high alpha
        double cd = cd0 + 1.5 * std::sin(alpha) * std::sin(alpha); // Induced drag approximation
        
        // Body Frame Conversion (Simplified)
        // CX = -CD cos(a) + CL sin(a)
        // CZ = -CD sin(a) - CL cos(a)
        double cx = -cd * std::cos(alpha) + cl * std::sin(alpha);
        double cz = -cd * std::sin(alpha) - cl * std::cos(alpha);
        
        // Moments (Assumed stable static margin)
        // Cm = -SM * CL
        double static_margin = 0.1; // 10% MAC
        double cm = -static_margin * cl;

        return {cx, 0.0, cz, 0.0, cm, 0.0, cd, cl};
    }

    /**
     * @brief Compute total forces and moments in Body Frame
     * @param com Center of Mass (relative to Nose, if Solver assumes Nose=Origin)
     */
    std::pair<Eigen::Vector3d, Eigen::Vector3d> compute_forces_moments(
        double dynamic_pressure,
        double mach,
        double alpha,
        double beta,
        const Eigen::Vector3d& com,
        double control_pitch = 0.0,
        double control_yaw = 0.0,
        double control_roll = 0.0
    ) const {
        
        AeroCoeffs coeffs = compute_coeffs(mach, alpha, beta);
        
        double Q_S = dynamic_pressure * m_config.ref_area;
        double Q_S_L = Q_S * m_config.ref_length;
        double Q_S_b = Q_S * m_config.ref_span;
        
        // 1. Aerodynamic Forces (Body Frame)
        Eigen::Vector3d force_body;
        force_body.x() = coeffs.CX * Q_S;
        force_body.y() = coeffs.CY * Q_S;
        force_body.z() = coeffs.CZ * Q_S;
        
        // 2. Aerodynamic Moments (About Nose/Origin)
        Eigen::Vector3d moment_aero_nose;
        moment_aero_nose.x() = coeffs.Cl * Q_S_b;
        moment_aero_nose.y() = coeffs.Cm * Q_S_L;
        moment_aero_nose.z() = coeffs.Cn * Q_S_b;
        
        // 3. Transfer Moment to CG
        // M_cg = M_nose - r_cg x F_aero
        // (Assuming Solver moments are about (0,0,0) and com is position of CG relative to (0,0,0))
        Eigen::Vector3d moment_body = moment_aero_nose - com.cross(force_body);
        
        // 4. Add Control Moments (Damping / Deflection)
        // Simplified: Additive control effectiveness
        moment_body.y() += m_config.cm_delta_pitch * control_pitch * Q_S_L;
        moment_body.z() += m_config.cn_delta_yaw * control_yaw * Q_S_b;
        moment_body.x() += m_config.cl_delta_roll * control_roll * Q_S_b;

        // Debug Output
        static int call_count = 0;
        if (call_count++ % 1000 == 0) { // Reduced frequency
             /*
             double alpha_deg = alpha * 180.0 / 3.14159265359;
             double beta_deg = beta * 180.0 / 3.14159265359;
             std::cout << "[Aero] M=" << mach << " A=" << alpha_deg << " B=" << beta_deg 
                       << " | CX=" << coeffs.CX << " CY=" << coeffs.CY << " CZ=" << coeffs.CZ 
                       << " | Fx=" << force_body.x() << " Fy=" << force_body.y() << " Fz=" << force_body.z() 
                       << " | q=" << dynamic_pressure << std::endl;
             */
        }

        return {force_body, moment_body};
    }

private:
    Config m_config;
    std::unique_ptr<aero::panel::AeroSolver> m_solver;
    TableData m_table;
};

} // namespace aerosp
