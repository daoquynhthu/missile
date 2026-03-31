#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <Eigen/Dense>

namespace AeroSim {
namespace RM {

/**
 * @brief POD version of AeroCoeffs for GPU
 */
struct AeroCoeffsGPU {
    double CX, CY, CZ;
    double Cl, Cm, Cn;
    double Cmq, Cnr;
};

/**
 * @brief GPU-compatible table data
 */
struct DartAeroTableGPU {
    int num_mach;
    int num_alpha;
    double* mach_grid;
    double* alpha_grid;
    AeroCoeffsGPU* data; // Flattened [mach_idx * num_alpha + alpha_idx]
};

/**
 * @brief High-Fidelity Aerodynamic Table for RM Dart
 * Loads coefficients from a CSV file and performs 2D interpolation (Mach, Alpha).
 */
class DartAeroTable {
public:
    struct AeroCoeffs {
        double CX, CY, CZ; // Forces
        double Cl, Cm, Cn; // Moments
        double Cmq, Cnr;   // Damping
        double CD, CL;     // Wind frame
    };

    struct TableData {
        std::vector<double> mach_grid;
        std::vector<double> alpha_grid;
        std::map<std::pair<int, int>, AeroCoeffs> data_map; // (mach_idx, alpha_idx) -> Coeffs
        bool loaded = false;
    };

    DartAeroTable(const std::string& csv_path) {
        load(csv_path);
    }

    void load(const std::string& csv_path) {
        load_csv_table(csv_path);
    }

    ~DartAeroTable() {
        if (m_gpu_data.mach_grid) cudaFree(m_gpu_data.mach_grid);
        if (m_gpu_data.alpha_grid) cudaFree(m_gpu_data.alpha_grid);
        if (m_gpu_data.data) cudaFree(m_gpu_data.data);
    }

    bool is_loaded() const { return m_table.loaded; }

    /**
     * @brief Prepare GPU data for Monte Carlo
     */
    void prepare_gpu() {
        if (!m_table.loaded) return;
        
        m_gpu_data.num_mach = m_table.mach_grid.size();
        m_gpu_data.num_alpha = m_table.alpha_grid.size();
        
        cudaMalloc(&m_gpu_data.mach_grid, m_gpu_data.num_mach * sizeof(double));
        cudaMalloc(&m_gpu_data.alpha_grid, m_gpu_data.num_alpha * sizeof(double));
        cudaMalloc(&m_gpu_data.data, m_gpu_data.num_mach * m_gpu_data.num_alpha * sizeof(AeroCoeffsGPU));
        
        cudaMemcpy(m_gpu_data.mach_grid, m_table.mach_grid.data(), m_gpu_data.num_mach * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(m_gpu_data.alpha_grid, m_table.alpha_grid.data(), m_gpu_data.num_alpha * sizeof(double), cudaMemcpyHostToDevice);
        
        std::vector<AeroCoeffsGPU> flat_data(m_gpu_data.num_mach * m_gpu_data.num_alpha);
        for (int i = 0; i < m_gpu_data.num_mach; ++i) {
            for (int j = 0; j < m_gpu_data.num_alpha; ++j) {
                AeroCoeffs c = m_table.data_map.at({i, j});
                flat_data[i * m_gpu_data.num_alpha + j] = {c.CX, c.CY, c.CZ, c.Cl, c.Cm, c.Cn, c.Cmq, c.Cnr};
            }
        }
        cudaMemcpy(m_gpu_data.data, flat_data.data(), flat_data.size() * sizeof(AeroCoeffsGPU), cudaMemcpyHostToDevice);
    }

    DartAeroTableGPU get_gpu_data() const { return m_gpu_data; }

    /**
     * @brief Host-side bilinear interpolation for debugging
     */
    AeroCoeffs get_coeffs(double mach, double alpha_deg) const {
        if (!m_table.loaded) return {0,0,0,0,0,0,0,0,0,0};

        auto it_mach = std::lower_bound(m_table.mach_grid.begin(), m_table.mach_grid.end(), mach);
        int m_idx = std::distance(m_table.mach_grid.begin(), it_mach);
        if (m_idx > 0) m_idx--;
        if (m_idx >= m_table.mach_grid.size() - 1) m_idx = m_table.mach_grid.size() - 2;

        auto it_alpha = std::lower_bound(m_table.alpha_grid.begin(), m_table.alpha_grid.end(), alpha_deg);
        int a_idx = std::distance(m_table.alpha_grid.begin(), it_alpha);
        if (a_idx > 0) a_idx--;
        if (a_idx >= m_table.alpha_grid.size() - 1) a_idx = m_table.alpha_grid.size() - 2;

        double m0 = m_table.mach_grid[m_idx];
        double m1 = m_table.mach_grid[m_idx + 1];
        double a0 = m_table.alpha_grid[a_idx];
        double a1 = m_table.alpha_grid[a_idx + 1];

        double rm = (mach - m0) / (m1 - m0);
        double ra = (alpha_deg - a0) / (a1 - a0);

        auto c00 = m_table.data_map.at({m_idx, a_idx});
        auto c01 = m_table.data_map.at({m_idx, a_idx + 1});
        auto c10 = m_table.data_map.at({m_idx + 1, a_idx});
        auto c11 = m_table.data_map.at({m_idx + 1, a_idx + 1});

        auto interp = [&](double v00, double v01, double v10, double v11) {
            return (1 - rm) * (1 - ra) * v00 + (1 - rm) * ra * v01 + rm * (1 - ra) * v10 + rm * ra * v11;
        };

        return {
            interp(c00.CX, c01.CX, c10.CX, c11.CX),
            interp(c00.CY, c01.CY, c10.CY, c11.CY),
            interp(c00.CZ, c01.CZ, c10.CZ, c11.CZ),
            interp(c00.Cl, c01.Cl, c10.Cl, c11.Cl),
            interp(c00.Cm, c01.Cm, c10.Cm, c11.Cm),
            interp(c00.Cn, c01.Cn, c10.Cn, c11.Cn),
            interp(c00.Cmq, c01.Cmq, c10.Cmq, c11.Cmq),
            interp(c00.Cnr, c01.Cnr, c10.Cnr, c11.Cnr),
            interp(c00.CD, c01.CD, c10.CD, c11.CD),
            interp(c00.CL, c01.CL, c10.CL, c11.CL)
        };
    }

    /**
     * @brief GPU-compatible bilinear interpolation
     */
    static __device__ AeroCoeffsGPU interpolate_gpu(const DartAeroTableGPU& table, double mach, double alpha_deg) {
        // 1. Find Mach interval
        int m0 = 0, m1 = 0;
        for (int i = 1; i < table.num_mach; ++i) {
            if (mach <= table.mach_grid[i]) {
                m1 = i; m0 = i - 1; break;
            }
        }
        if (m1 == 0) { m0 = table.num_mach - 2; m1 = table.num_mach - 1; }

        // 2. Find Alpha interval
        int a0 = 0, a1 = 0;
        for (int i = 1; i < table.num_alpha; ++i) {
            if (alpha_deg <= table.alpha_grid[i]) {
                a1 = i; a0 = i - 1; break;
            }
        }
        if (a1 == 0) { a0 = table.num_alpha - 2; a1 = table.num_alpha - 1; }

        double dm = (mach - table.mach_grid[m0]) / (table.mach_grid[m1] - table.mach_grid[m0]);
        double da = (alpha_deg - table.alpha_grid[a0]) / (table.alpha_grid[a1] - table.alpha_grid[a0]);

        auto c00 = table.data[m0 * table.num_alpha + a0];
        auto c01 = table.data[m0 * table.num_alpha + a1];
        auto c10 = table.data[m1 * table.num_alpha + a0];
        auto c11 = table.data[m1 * table.num_alpha + a1];

        auto interp = [&](double v00, double v01, double v10, double v11) {
            double v0 = v00 * (1.0 - da) + v01 * da;
            double v1 = v10 * (1.0 - da) + v11 * da;
            return v0 * (1.0 - dm) + v1 * dm;
        };

        return {
            interp(c00.CX, c01.CX, c10.CX, c11.CX),
            interp(c00.CY, c01.CY, c10.CY, c11.CY),
            interp(c00.CZ, c01.CZ, c10.CZ, c11.CZ),
            interp(c00.Cl, c01.Cl, c10.Cl, c11.Cl),
            interp(c00.Cm, c01.Cm, c10.Cm, c11.Cm),
            interp(c00.Cn, c01.Cn, c10.Cn, c11.Cn),
            interp(c00.Cmq, c01.Cmq, c10.Cmq, c11.Cmq),
            interp(c00.Cnr, c01.Cnr, c10.Cnr, c11.Cnr)
        };
    }

    /**
     * @brief Get aerodynamic coefficients via 2D interpolation
     * @param mach Mach number
     * @param alpha_deg Angle of Attack (deg)
     * @param beta_deg Side-slip angle (deg) - Simplified: uses alpha interpolation for now
     */
    AeroCoeffs get_coeffs(double mach, double alpha_deg, double beta_deg) const {
        if (!m_table.loaded) return {0,0,0,0,0,0,0,0,0,0};

        // 1. Find Mach interval
        auto it_m = std::lower_bound(m_table.mach_grid.begin(), m_table.mach_grid.end(), mach);
        int m1 = std::distance(m_table.mach_grid.begin(), it_m);
        if (m1 == 0) m1 = 1;
        if (m1 >= m_table.mach_grid.size()) m1 = m_table.mach_grid.size() - 1;
        int m0 = m1 - 1;

        // 2. Find Alpha interval
        auto it_a = std::lower_bound(m_table.alpha_grid.begin(), m_table.alpha_grid.end(), alpha_deg);
        int a1 = std::distance(m_table.alpha_grid.begin(), it_a);
        if (a1 == 0) a1 = 1;
        if (a1 >= m_table.alpha_grid.size()) a1 = m_table.alpha_grid.size() - 1;
        int a0 = a1 - 1;

        // 3. Perform 2D Bilinear Interpolation
        double dm = (mach - m_table.mach_grid[m0]) / (m_table.mach_grid[m1] - m_table.mach_grid[m0]);
        double da = (alpha_deg - m_table.alpha_grid[a0]) / (m_table.alpha_grid[a1] - m_table.alpha_grid[a0]);

        AeroCoeffs c00 = m_table.data_map.at({m0, a0});
        AeroCoeffs c01 = m_table.data_map.at({m0, a1});
        AeroCoeffs c10 = m_table.data_map.at({m1, a0});
        AeroCoeffs c11 = m_table.data_map.at({m1, a1});

        auto interpolate = [&](double v00, double v01, double v10, double v11) {
            double v0 = v00 * (1 - da) + v01 * da;
            double v1 = v10 * (1 - da) + v11 * da;
            return v0 * (1 - dm) + v1 * dm;
        };

        AeroCoeffs result;
        result.CX = interpolate(c00.CX, c01.CX, c10.CX, c11.CX);
        result.CY = interpolate(c00.CY, c01.CY, c10.CY, c11.CY);
        result.CZ = interpolate(c00.CZ, c01.CZ, c10.CZ, c11.CZ);
        result.Cl = interpolate(c00.Cl, c01.Cl, c10.Cl, c11.Cl);
        result.Cm = interpolate(c00.Cm, c01.Cm, c10.Cm, c11.Cm);
        result.Cn = interpolate(c00.Cn, c01.Cn, c10.Cn, c11.Cn);
        result.Cmq = interpolate(c00.Cmq, c01.Cmq, c10.Cmq, c11.Cmq);
        result.Cnr = interpolate(c00.Cnr, c01.Cnr, c10.Cnr, c11.Cnr);
        result.CD = interpolate(c00.CD, c01.CD, c10.CD, c11.CD);
        result.CL = interpolate(c00.CL, c01.CL, c10.CL, c11.CL);

        return result;
    }

private:
    void load_csv_table(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open dart aero table " << path << std::endl;
            return;
        }

        std::string line, header;
        std::getline(file, header); // Skip header

        std::map<double, int> m_map, a_map;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stod(cell));
            }

            if (row.size() < 13) continue;

            double m = row[0];
            double a = row[1];
            
            if (m_map.find(m) == m_map.end()) {
                m_map[m] = m_table.mach_grid.size();
                m_table.mach_grid.push_back(m);
            }
            if (a_map.find(a) == a_map.end()) {
                a_map[a] = m_table.alpha_grid.size();
                m_table.alpha_grid.push_back(a);
            }

            AeroCoeffs c;
            c.CX = row[3]; c.CY = row[4]; c.CZ = row[5];
            c.CL = row[6]; c.CD = row[7];
            c.Cl = row[8]; c.Cm = row[9]; c.Cn = row[10];
            c.Cmq = row[11]; c.Cnr = row[12];

            m_table.data_map[{m_map[m], a_map[a]}] = c;
        }

        std::sort(m_table.mach_grid.begin(), m_table.mach_grid.end());
        std::sort(m_table.alpha_grid.begin(), m_table.alpha_grid.end());
        
        // Re-map indices because sort might have changed order
        std::map<std::pair<int, int>, AeroCoeffs> sorted_data;
        for (auto const& [key, val] : m_table.data_map) {
            // Find new indices
            double m = 0, a = 0;
            // This is slow but only done once at load
            for (auto const& [mv, mi] : m_map) if (mi == key.first) m = mv;
            for (auto const& [av, ai] : a_map) if (ai == key.second) a = av;
            
            int new_m = std::distance(m_table.mach_grid.begin(), std::find(m_table.mach_grid.begin(), m_table.mach_grid.end(), m));
            int new_a = std::distance(m_table.alpha_grid.begin(), std::find(m_table.alpha_grid.begin(), m_table.alpha_grid.end(), a));
            sorted_data[{new_m, new_a}] = val;
        }
        m_table.data_map = sorted_data;
        m_table.loaded = true;
        std::cout << "[Aero] Loaded dart table: " << m_table.mach_grid.size() << " Mach x " << m_table.alpha_grid.size() << " Alpha points." << std::endl;
    }

    TableData m_table;
    DartAeroTableGPU m_gpu_data = {0, 0, nullptr, nullptr, nullptr};
};

} // namespace RM
} // namespace AeroSim
