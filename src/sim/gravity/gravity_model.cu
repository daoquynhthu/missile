#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sim/gravity/gravity_model.hpp"
#include "infra/math/constants.hpp"
#include "infra/cuda/cuda_utils.cuh"
#include "infra/util/progress_bar.hpp"

namespace aerosp {

// CUDA kernel to compute gravity using spherical harmonics (Pines' Algorithm - Fully Normalized)
// Reference: Pines, S. (1973). "Uniform Representation of the Gravitational Potential and its Derivatives"
// This implementation supports full-degree EGM2008 (up to N=360 or higher).
// Helper function for both Host and Device
// This function encapsulates the core gravity calculation logic (Pines' Algorithm)
__host__ __device__ void compute_gravity_single_point(const Vec3& r_ecef, double mu, double R, int N, 
                               const double* C, const double* S, Vec3* g_out) {
    double x = r_ecef.x;
    double y = r_ecef.y;
    double z = r_ecef.z;
    double r_sq = x*x + y*y + z*z;
    double r = sqrt(r_sq);
    
    if (r < 6000000.0) { // Simple safety check for Earth surface
        *g_out = Vec3(0, 0, 0);
        return;
    }

    // Constants
    double u = z / r; // sin(phi)

    // Implementation of the "Pines" algorithm (singular-free at poles)
    // using Forward Column Recurrence for Normalized Associated Legendre Functions P_nm

    double r_inv = 1.0 / r;
    
    // Acceleration components in Spherical-like frame (but projected to ECEF)
    // We accumulate potential derivatives.
    double dV_dr = 0.0;
    double dV_dphi = 0.0;
    double dV_dlam = 0.0;
    
    // Precompute sectorial terms (cos(m*lambda), sin(m*lambda))
    double r_xy_sq = x*x + y*y;
    double r_xy = sqrt(r_xy_sq);
    double c_lambda = x / r_xy;
    double s_lambda = y / r_xy;
    
    if (r_xy < 1e-10) { // Polar singularity check
        c_lambda = 1.0;
        s_lambda = 0.0;
    }
    
    // We will use a fixed maximum buffer size.
    const int MAX_DEGREE_BUFFER = 71; // Reduced from 361 to save registers/local memory for speed check
    double P_col[MAX_DEGREE_BUFFER]; // P_{n,m}
    
    double cos_phi = r_xy * r_inv; // sqrt(x^2+y^2)/r
    
    // But we need to handle N up to 360.
    int max_n = N;
    
    // Central term
    // Scale factor for R/r
    double R_r = R * r_inv;

    // We'll limit N inside kernel to MAX_DEGREE_BUFFER-1
    if (max_n >= MAX_DEGREE_BUFFER) max_n = MAX_DEGREE_BUFFER - 1;

    // --- Iterate m from 0 to max_n ---
    
    // Initialize P_mm
    // P_00 = 1
    P_col[0] = 1.0; 
    
    // We need to maintain P_mm for next step
    double P_mm_curr = 1.0; 
    
    // We also need cos(m*lambda) and sin(m*lambda)
    double cm = 1.0;
    double sm = 0.0;
    
    for (int m = 0; m <= max_n; ++m) {
        
        // 1. Compute P_nm for n=m..max_n
        // For n=m: P_mm is already known (P_mm_curr)
        P_col[m] = P_mm_curr;
        
        // For n=m+1 .. max_n
        if (m < max_n) {
            // P_{m+1, m} = u * sqrt(2m+3) * P_{m,m}
            double anm = sqrt((double)(2*m + 3));
            P_col[m+1] = anm * u * P_col[m];
        }
        
        for (int n = m + 2; n <= max_n; ++n) {
            // P_{n,m} = a_nm * u * P_{n-1,m} - b_nm * P_{n-2,m}
            double n_d = (double)n;
            double m_d = (double)m;
            double anm = sqrt(((2.0*n_d - 1.0)*(2.0*n_d + 1.0)) / ((n_d - m_d)*(n_d + m_d)));
            double bnm = sqrt(((2.0*n_d + 1.0)*(n_d + m_d - 1.0)*(n_d - m_d - 1.0)) / ((2.0*n_d - 3.0)*(n_d + m_d)*(n_d - m_d)));
            
            P_col[n] = anm * u * P_col[n-1] - bnm * P_col[n-2];
        }
        
        // 2. Accumulate contributions
        // Precompute powers of R_r for this m-loop
        // R_r_m = (R/r)^m
        // We will update it in the n-loop
        double R_r_n = pow(R_r, m); 

        for (int n = m; n <= max_n; ++n) {
            if (n < 2) {
                 // Update R_r_n for next n even if we skip
                 R_r_n *= R_r;
                 continue; 
            }
            
            // Index in C/S arrays
            // Mapping (n,m) to linear index
            // Typically: idx = n*(n+1)/2 + m
            int idx = n * (n + 1) / 2 + m;
            double C_nm = C[idx];
            double S_nm = S[idx];
            
            double P_nm = P_col[n];
            
            // Term V = (GM/r) * (R/r)^n * P_nm * (C cos + S sin)
            
            double term_common = (mu * r_inv) * R_r_n;
            double trig_term = C_nm * cm + S_nm * sm;
            double trig_term_deriv = S_nm * cm - C_nm * sm; // d(trig)/d(m*lam)
            
            // Radial derivative contribution
            // dV_dr += -(n+1)/r * term_common * P_nm * trig_term
            dV_dr -= (n + 1.0) * r_inv * term_common * P_nm * trig_term;
            
            // Lambda derivative contribution
            // dV/dlam = term_common * P_nm * m * (S cos - C sin)
            dV_dlam += term_common * P_nm * m * trig_term_deriv;
            
            // Phi derivative contribution
            // Since we are inside the atmosphere/space (r > R), we are rarely at exact poles.
            // We can clamp cos_phi.
            double cos_phi_safe = (cos_phi < 1e-10) ? 1e-10 : cos_phi;
            
            double N_nm = sqrt( ((2.0*n+1.0)/(2.0*n-1.0)) * ((double)(n-m)/(double)(n+m)) );
            
            double dP_dphi = (n * u * P_nm - (n + m) * N_nm * P_col[n-1]) / cos_phi_safe;
            
            if (n == m) {
                 // Special case for n=m
                 dP_dphi = -m * u / cos_phi_safe * P_nm;
            }
            
            dV_dphi += term_common * dP_dphi * trig_term;

            // Prepare for next n
            R_r_n *= R_r;
        }
        
        // Update P_mm for next column (m+1)
        double fact = sqrt((2.0*m + 3.0) / (2.0*m + 2.0));
        P_mm_curr = fact * cos_phi * P_mm_curr;
        
        // Update trig terms
        double cm_new = cm * c_lambda - sm * s_lambda;
        double sm_new = sm * c_lambda + cm * s_lambda;
        cm = cm_new;
        sm = sm_new;
    }
    
    // --- 3. Convert Potential Derivatives to Acceleration in ECEF ---
    double g_r = dV_dr;
    double g_phi = r_inv * dV_dphi;
    double cos_phi_safe_2 = (cos_phi < 1e-10) ? 1e-10 : cos_phi;
    double g_lam = r_inv / cos_phi_safe_2 * dV_dlam;
    
    double gx = g_r * (c_lambda * cos_phi) + g_phi * (-c_lambda * u) + g_lam * (-s_lambda);
    double gy = g_r * (s_lambda * cos_phi) + g_phi * (-s_lambda * u) + g_lam * (c_lambda);
    double gz = g_r * u + g_phi * cos_phi;
    
    // Add central body term (if C00=1 is not handled or handled differently)
    // EGM2008 typically has C00=1. We skipped n < 2.
    // So we must ADD the central term.
    gx += -mu * x / (r_sq * r);
    gy += -mu * y / (r_sq * r);
    gz += -mu * z / (r_sq * r);

    *g_out = Vec3(gx, gy, gz);
}

// Single-point kernel (wrapper)
__global__ void gravity_kernel(Vec3 r_ecef, double mu, double R, int N, 
                               const double* C, const double* S, Vec3* g_out) {
    compute_gravity_single_point(r_ecef, mu, R, N, C, S, g_out);
}

// Batch kernel for multi-trajectory simulation
__global__ void gravity_kernel_batch(const Vec3* r_ecef_arr, double mu, double R, int N, 
                               const double* C, const double* S, Vec3* g_out_arr, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        compute_gravity_single_point(r_ecef_arr[idx], mu, R, N, C, S, &g_out_arr[idx]);
    }
}

// Wrapper for the kernel
void launch_gravity_kernel(const Vec3& r_ecef, double mu, double R, int N, 
                           const double* d_C, const double* d_S, Vec3* h_g_out) {
    Vec3* d_g_out;
    cudaMalloc(&d_g_out, sizeof(Vec3));
    
    // Launch configuration: 1 block, 1 thread (since we process 1 missile)
    // For batch processing, we would change this.
    // Increasing stack size if necessary for large N
    size_t stackSize = 0;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (stackSize < 4096) {
        cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    }

    gravity_kernel<<<1, 1>>>(r_ecef, mu, R, N, d_C, d_S, d_g_out);
    
    cudaMemcpy(h_g_out, d_g_out, sizeof(Vec3), cudaMemcpyDeviceToHost);
    cudaFree(d_g_out);
}

} // namespace aerosp

// Explicit qualification for class methods outside namespace block
aerosp::GravityModel::GravityModel(int max_degree) 
    : m_max_degree(max_degree), m_loaded_max_degree(0), m_mu(0), m_radius(0),
      d_C(nullptr), d_S(nullptr), m_cuda_ready(false) {
    
    // int num_coeffs = (max_degree + 1) * (max_degree + 2) / 2;
    // m_C.assign(num_coeffs, 0.0);
    // m_S.assign(num_coeffs, 0.0);
}

aerosp::GravityModel::~GravityModel() {
    if (d_C) cudaFree(d_C);
    if (d_S) cudaFree(d_S);
}

bool aerosp::GravityModel::load_coefficients(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    // Count lines for progress bar
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // aerosp::ProgressBar progress(file_size, 50, "Loading EGM2008");
    std::cout << "[Init] Loading EGM2008..." << std::endl;

    std::string line;
    bool in_data = false;
    size_t bytes_read = 0;
    
    while (std::getline(file, line)) {
        bytes_read += line.length() + 1; // +1 for newline
        // progress.update(bytes_read);
        
        if (!in_data) {
            if (line.find("earth_gravity_constant") != std::string::npos) {
                size_t pos = line.find_last_of(" ");
                if (pos != std::string::npos) {
                    std::string val = line.substr(pos+1);
                    for (char &c : val) if (c == 'd' || c == 'D') c = 'e';
                    m_mu = std::stod(val);
                }
            } else if (line.find("radius") != std::string::npos) {
                size_t pos = line.find_last_of(" ");
                if (pos != std::string::npos) {
                    std::string val = line.substr(pos+1);
                    for (char &c : val) if (c == 'd' || c == 'D') c = 'e';
                    m_radius = std::stod(val);
                }
            } else if (line.find("max_degree") != std::string::npos) {
                size_t pos = line.find_last_of(" ");
                if (pos != std::string::npos) {
                    int deg = std::stoi(line.substr(pos+1));
                    m_loaded_max_degree = std::min(deg, m_max_degree);
                }
            } else if (line.find("end_of_head") != std::string::npos) {
                in_data = true;
                // Initialize vectors now that we know max degree
                int num_coeffs = (m_loaded_max_degree + 1) * (m_loaded_max_degree + 2) / 2;
                m_C.resize(num_coeffs, 0.0);
                m_S.resize(num_coeffs, 0.0);
            }
            continue;
        }
        
        // Data lines:  gfc   n   m   Cnm   Snm   dC   dS
        // Skip empty lines
        if (line.empty()) continue;
        
        // Replace 'd' with 'e' for FORTRAN style scientific notation
        for (char &c : line) if (c == 'd' || c == 'D') c = 'e';
        
        std::stringstream ss(line);
        std::string type; int n, m; double c, s;
        if (ss >> type >> n >> m >> c >> s) {
            if (n <= m_loaded_max_degree && m <= n) {
                int index = n * (n + 1) / 2 + m;
                if (index < m_C.size()) {
                    m_C[index] = c;
                    m_S[index] = s;
                }
            }
        }
    }
    // progress.finish();
    return true;
}

void aerosp::GravityModel::prepare_cuda() {
    if (m_cuda_ready) return;
    if (m_C.empty()) return;
    
    size_t size = m_C.size() * sizeof(double);
    cudaError_t err;
    
    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) std::cerr << "CUDA Malloc C failed: " << cudaGetErrorString(err) << std::endl;
    
    err = cudaMalloc(&d_S, size);
    if (err != cudaSuccess) std::cerr << "CUDA Malloc S failed: " << cudaGetErrorString(err) << std::endl;
    
    err = cudaMemcpy(d_C, m_C.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cerr << "CUDA Memcpy C failed: " << cudaGetErrorString(err) << std::endl;
    
    err = cudaMemcpy(d_S, m_S.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cerr << "CUDA Memcpy S failed: " << cudaGetErrorString(err) << std::endl;
    
    m_cuda_ready = true;
}

Eigen::Vector3d aerosp::GravityModel::calculate_acceleration_cuda(const Eigen::Vector3d& r_ecef) const {
    // Need to cast away constness conceptually or use mutable if we were modifying logic, 
    // but here we just need to ensure d_C/d_S are allocated.
    // The prepare_cuda() should be called before simulation loop.
    // Ideally check here:
    if (!m_cuda_ready) {
        // This is a hack, in real code handle initialization better
        const_cast<aerosp::GravityModel*>(this)->prepare_cuda();
    }

    aerosp::Vec3 h_r(r_ecef.x(), r_ecef.y(), r_ecef.z());

    // Launch configuration: 1 thread for single point calculation
    // Note: This is inefficient due to per-call cudaMalloc. 
    // Use calculate_accelerations_cuda (batch) or calculate_acceleration (CPU) instead.
    aerosp::Vec3* d_g;
    cudaMalloc(&d_g, sizeof(aerosp::Vec3));

    // Increase stack size if needed (for large N recursion/stack variables)
    size_t stackSize = 0;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (stackSize < 16384) { // Increase to 16KB for safety with N=360
        cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    }

    aerosp::gravity_kernel<<<1, 1>>>(h_r, m_mu, m_radius, m_loaded_max_degree, d_C, d_S, d_g);
    
    aerosp::Vec3 h_g;
    cudaMemcpy(&h_g, d_g, sizeof(aerosp::Vec3), cudaMemcpyDeviceToHost);
    cudaFree(d_g);
    
    return Eigen::Vector3d(h_g.x, h_g.y, h_g.z);
}


// Host-side CPU implementation (Fixed: No longer calls inefficient GPU kernel)
Eigen::Vector3d aerosp::GravityModel::calculate_acceleration(const Eigen::Vector3d& r_ecef) const {
    aerosp::Vec3 r(r_ecef.x(), r_ecef.y(), r_ecef.z());
    aerosp::Vec3 g;
    
    // Direct CPU call using shared logic
    compute_gravity_single_point(r, m_mu, m_radius, m_loaded_max_degree, m_C.data(), m_S.data(), &g);
    
    return Eigen::Vector3d(g.x, g.y, g.z);
}

// Batch GPU implementation for multi-trajectory simulation
std::vector<Eigen::Vector3d> aerosp::GravityModel::calculate_accelerations_cuda(const std::vector<Eigen::Vector3d>& r_ecef_list) const {
    if (r_ecef_list.empty()) return {};
    
    if (!m_cuda_ready) {
        const_cast<aerosp::GravityModel*>(this)->prepare_cuda();
    }

    size_t count = r_ecef_list.size();
    
    // Allocate device memory for inputs/outputs
    // Ideally, these should be persistent buffers to avoid malloc overhead every step.
    // But for batch=1000, one malloc is negligible compared to 1000 mallocs.
    
    aerosp::Vec3* d_r_list;
    aerosp::Vec3* d_g_list;
    
    cudaMalloc(&d_r_list, count * sizeof(aerosp::Vec3));
    cudaMalloc(&d_g_list, count * sizeof(aerosp::Vec3));
    
    // Prepare input data
    std::vector<aerosp::Vec3> h_r_list(count);
    for(size_t i=0; i<count; ++i) {
        h_r_list[i] = aerosp::Vec3(r_ecef_list[i].x(), r_ecef_list[i].y(), r_ecef_list[i].z());
    }
    
    cudaMemcpy(d_r_list, h_r_list.data(), count * sizeof(aerosp::Vec3), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    // Ensure stack size
    size_t stackSize = 0;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (stackSize < 16384) { 
        cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    }
    
    aerosp::gravity_kernel_batch<<<blocksPerGrid, threadsPerBlock>>>(d_r_list, m_mu, m_radius, m_loaded_max_degree, d_C, d_S, d_g_list, count);
    
    // Copy back
    std::vector<aerosp::Vec3> h_g_list(count);
    cudaMemcpy(h_g_list.data(), d_g_list, count * sizeof(aerosp::Vec3), cudaMemcpyDeviceToHost);
    
    cudaFree(d_r_list);
    cudaFree(d_g_list);
    
    // Convert to Eigen
    std::vector<Eigen::Vector3d> result(count);
    for(size_t i=0; i<count; ++i) {
        result[i] = Eigen::Vector3d(h_g_list[i].x, h_g_list[i].y, h_g_list[i].z);
    }
    
    return result;
}
