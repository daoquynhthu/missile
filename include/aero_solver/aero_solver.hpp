#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Solver {

    struct Triangle {
        float3 v0, v1, v2;
        float3 center; // Centroid for moment calculation
        float3 normal;
        float area;
    };

    struct AeroCoefficients {
        float CX, CY, CZ; // Body frame force coefficients
        float CL, CD;     // Wind frame lift/drag coefficients
        float Cl, Cm, Cn; // Moment coefficients (Roll, Pitch, Yaw)
        float L_D;        // Lift-to-Drag ratio
    };

    class AeroSolver {
    public:
        AeroSolver();
        ~AeroSolver();

        // Initialize solver with STL file path
        // ref_area: Reference area (m^2) for coefficients
        // ref_length: Reference length (m) for pitching moment
        // ref_span: Reference span (m) for rolling/yawing moments
        bool load_model(const std::string& stl_path, float ref_area = 1.0f, float ref_length = 1.0f, float ref_span = 1.0f);

        // Load mesh directly from memory
        bool load_mesh(const std::vector<Triangle>& mesh, float ref_area = 1.0f, float ref_length = 1.0f, float ref_span = 1.0f);

        // Compute aerodynamic coefficients
        AeroCoefficients compute_coefficients(float mach, float alpha_deg, float beta_deg = 0.0f);

    private:
        Triangle* d_triangles = nullptr; // GPU memory for triangles
        float3* d_forces = nullptr;      // GPU memory for per-triangle forces
        float3* d_moments = nullptr;     // GPU memory for per-triangle moments
        int num_triangles = 0;
        
        float ref_area = 1.0f;
        float ref_length = 1.0f;
        float ref_span = 1.0f;
        float3 moment_ref_point = {0.0f, 0.0f, 0.0f}; // Moment reference point (e.g. nose or CG)

        // Helper to parse STL
        std::vector<Triangle> parse_stl(const std::string& path);
    };

}
}
