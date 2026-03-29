#include "aero_solver/aero_solver.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>

// Helper for float3 operations
__device__ __host__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __host__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __host__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ __host__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ __host__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

namespace AeroSim {
namespace Solver {

    // --- CUDA Kernel Definitions ---
    // Modified Newtonian Theory
    __global__ void compute_forces_moments_kernel(const Triangle* triangles, int num_triangles, 
                                          float mach, float alpha_rad, float beta_rad,
                                          float3* forces_out, float3* moments_out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_triangles) return;

        Triangle tri = triangles[idx];
        
        // Flow direction vector in body frame
        // V_inf vector comes from:
        // alpha (pitch), beta (sideslip)
        // V_inf_body = V_mag * [cos(a)cos(b), sin(b), sin(a)cos(b)]
        // Flow direction d = -V_inf_body / V_mag
        float ca = cosf(alpha_rad);
        float sa = sinf(alpha_rad);
        float cb = cosf(beta_rad);
        float sb = sinf(beta_rad);

        float3 flow_dir = make_float3(-ca * cb, -sb, -sa * cb);
        
        // Angle between surface normal and flow direction
        // cos(theta) = dot(n, flow_dir)
        float dot_val = dot(tri.normal, flow_dir);
                        
        float Cp = 0.0f;
        
        if (dot_val < 0.0f) { // Compression side (facing the flow)
            // dot_val is negative cosine of incidence angle
            // Modified Newtonian: Cp = Cp_max * cos^2(theta) = Cp_max * dot_val^2
            
            // Cp_max approx 2.0 for hypersonic flow
            // More accurate: Cp_max = (gamma+3)/(gamma+1) * [1 - 1/(gamma*M^2)] ??
            // Simplified: 2.0 * sin^2(deflection)
            Cp = 2.0f * dot_val * dot_val; 
        } else {
            // Shadow region
            // Cp = -1/M^2 (Vacuum limit approx) or 0
            if (mach > 0.1f)
                Cp = -1.0f / (mach * mach);
            else
                Cp = 0.0f;
        }
        
        // Force Term (Contribution to Coefficient Sum)
        // F_term = -Cp * Area * Normal
        // We sum these up and divide by S_ref later
        float3 force_term = tri.normal * (-Cp * tri.area);
        
        forces_out[idx] = force_term;
        
        // Moment Term (Contribution to Coefficient Sum)
        // M_term = r x F_term
        // r = triangle centroid (assuming moment ref point is origin 0,0,0)
        float3 moment_term = cross(tri.center, force_term);
        
        moments_out[idx] = moment_term;
    }

    AeroSolver::AeroSolver() {}

    AeroSolver::~AeroSolver() {
        if (d_triangles) cudaFree(d_triangles);
        if (d_forces) cudaFree(d_forces);
        if (d_moments) cudaFree(d_moments);
    }

    bool AeroSolver::load_mesh(const std::vector<Triangle>& mesh, float ref_area_in, float ref_length_in, float ref_span_in) {
        this->ref_area = ref_area_in;
        this->ref_length = ref_length_in;
        this->ref_span = ref_span_in;

        if (mesh.empty()) return false;

        // Free existing memory if any
        if (d_triangles) cudaFree(d_triangles);
        if (d_forces) cudaFree(d_forces);
        if (d_moments) cudaFree(d_moments);

        num_triangles = mesh.size();

        cudaMalloc(&d_triangles, num_triangles * sizeof(Triangle));
        cudaMalloc(&d_forces, num_triangles * sizeof(float3));
        cudaMalloc(&d_moments, num_triangles * sizeof(float3));

        cudaMemcpy(d_triangles, mesh.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);

        return true;
    }

    bool AeroSolver::load_model(const std::string& stl_path, float ref_area_in, float ref_length_in, float ref_span_in) {
        std::vector<Triangle> host_triangles = parse_stl(stl_path);
        if (host_triangles.empty()) return false;

        if (load_mesh(host_triangles, ref_area_in, ref_length_in, ref_span_in)) {
             std::cout << "Loaded STL Model: " << stl_path << " (" << num_triangles << " triangles) to GPU." << std::endl;
             return true;
        }
        return false;
    }

    AeroCoefficients AeroSolver::compute_coefficients(float mach, float alpha_deg, float beta_deg) {
        float alpha_rad = alpha_deg * 3.14159265359f / 180.0f;
        float beta_rad = beta_deg * 3.14159265359f / 180.0f;

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_triangles + threadsPerBlock - 1) / threadsPerBlock;

        compute_forces_moments_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_triangles, num_triangles, mach, alpha_rad, beta_rad, d_forces, d_moments
        );
        cudaDeviceSynchronize();

        // Copy back results (for small number of triangles, CPU sum is fine)
        std::vector<float3> h_forces(num_triangles);
        std::vector<float3> h_moments(num_triangles);
        
        cudaMemcpy(h_forces.data(), d_forces, num_triangles * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_moments.data(), d_moments, num_triangles * sizeof(float3), cudaMemcpyDeviceToHost);

        float3 total_force_term = {0,0,0};
        float3 total_moment_term = {0,0,0};

        for(int i=0; i<num_triangles; ++i) {
            total_force_term = total_force_term + h_forces[i];
            total_moment_term = total_moment_term + h_moments[i];
        }

        AeroCoefficients coeffs;
        
        // Body Frame Coefficients
        // CX = Fx / S_ref
        coeffs.CX = total_force_term.x / ref_area;
        coeffs.CY = total_force_term.y / ref_area;
        coeffs.CZ = total_force_term.z / ref_area;
        
        // Moment Coefficients
        // Cl = Mx / (S_ref * b_ref)
        // Cm = My / (S_ref * c_ref)
        // Cn = Mz / (S_ref * b_ref)
        coeffs.Cl = total_moment_term.x / (ref_area * ref_span);
        coeffs.Cm = total_moment_term.y / (ref_area * ref_length);
        coeffs.Cn = total_moment_term.z / (ref_area * ref_span);

        // Wind Frame Coefficients (Lift/Drag)
        // Drag is force component parallel to V_inf
        // Lift is force component perpendicular to V_inf (in vertical plane usually)
        // Transformation from Body to Wind (at beta=0):
        // CD = -CX cos(a) - CZ sin(a)  (Wait, CX is usually negative for drag? No, CX is forward force)
        // Standard Body Frame: X forward, Z down.
        // Drag is backwards.
        // F_drag = -F_wind_x
        // F_wind_x = F_body_x cos(a)cos(b) + F_body_y sin(b) + F_body_z sin(a)cos(b) ??
        
        // Let's use simple alpha rotation for now (beta=0 approx)
        float ca = cosf(alpha_rad);
        float sa = sinf(alpha_rad);
        
        // Drag D = -F_stability_x
        // Lift L = -F_stability_z
        // Stability Frame is Body Frame rotated by alpha around Y.
        // F_stab_x = F_body_x cos(a) + F_body_z sin(a)
        // F_stab_z = -F_body_x sin(a) + F_body_z cos(a)
        
        // CD is usually positive. If F_body_x is negative (drag), then:
        // CD = -(CX cos a + CZ sin a)
        coeffs.CD = -(coeffs.CX * ca + coeffs.CZ * sa);
        
        // CL is usually positive (up). If F_body_z is negative (lift up in Z-down), then:
        // CL = -(-CX sin a + CZ cos a) = CX sin a - CZ cos a
        coeffs.CL = coeffs.CX * sa - coeffs.CZ * ca;
        
        if (std::abs(coeffs.CD) > 1e-6f)
            coeffs.L_D = coeffs.CL / coeffs.CD;
        else
            coeffs.L_D = 0.0f;

        return coeffs;
    }

    std::vector<Triangle> AeroSolver::parse_stl(const std::string& path) {
        std::vector<Triangle> triangles;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open STL file: " << path << std::endl;
            return triangles;
        }

        // Skip header
        file.seekg(80);

        uint32_t count;
        file.read(reinterpret_cast<char*>(&count), 4);

        for (uint32_t i = 0; i < count; ++i) {
            Triangle t;
            float normal[3];
            float v0[3], v1[3], v2[3];
            uint16_t attr;

            file.read(reinterpret_cast<char*>(normal), 12);
            file.read(reinterpret_cast<char*>(v0), 12);
            file.read(reinterpret_cast<char*>(v1), 12);
            file.read(reinterpret_cast<char*>(v2), 12);
            file.read(reinterpret_cast<char*>(&attr), 2);

            t.normal = make_float3(normal[0], normal[1], normal[2]);
            t.v0 = make_float3(v0[0], v0[1], v0[2]);
            t.v1 = make_float3(v1[0], v1[1], v1[2]);
            t.v2 = make_float3(v2[0], v2[1], v2[2]);

            // Calculate Area
            float3 e1 = t.v1 - t.v0;
            float3 e2 = t.v2 - t.v0;
            float3 cp = cross(e1, e2);
            t.area = 0.5f * sqrtf(dot(cp, cp));

            // Calculate Centroid
            t.center = (t.v0 + t.v1 + t.v2) * (1.0f/3.0f);
            
            // Re-normalize normal if needed (some STLs have bad normals)
            // But usually we trust the file or recompute.
            // Let's recompute normal to be safe and ensure consistent winding
            float len_cp = sqrtf(dot(cp, cp));
            if (len_cp > 1e-8f) {
                t.normal = cp * (1.0f / len_cp);
            }

            triangles.push_back(t);
        }
        return triangles;
    }

}
}
