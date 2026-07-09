#ifndef GRAVITY_MODEL_HPP
#define GRAVITY_MODEL_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

namespace AeroSim {

    class GravityModel {
    public:
        GravityModel(int max_degree = 360);
        ~GravityModel();
        
        bool load_coefficients(const std::string& filepath);
        Eigen::Vector3d calculate_acceleration(const Eigen::Vector3d& r_ecef) const;
        
        // Batch processing for multi-trajectory simulation
        std::vector<Eigen::Vector3d> calculate_accelerations_cuda(const std::vector<Eigen::Vector3d>& r_ecef_list) const;
        
        void prepare_cuda();
        // Deprecated single-point CUDA call (kept for compatibility but implementation will change)
        Eigen::Vector3d calculate_acceleration_cuda(const Eigen::Vector3d& r_ecef) const;
        int get_loaded_max_degree() const { return m_loaded_max_degree; }

    private:
        int m_max_degree;
        int m_loaded_max_degree;
        double m_mu;
        double m_radius;

        std::vector<double> m_C;
        std::vector<double> m_S;

        double* d_C;
        double* d_S;
        bool m_cuda_ready;

        inline int idx(int n, int m) const {
            return n * (n + 1) / 2 + m;
        }
    };
}

#endif // GRAVITY_MODEL_HPP
