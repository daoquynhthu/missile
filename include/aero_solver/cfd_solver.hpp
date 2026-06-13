#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "aero_solver/mesh_generator.hpp"

namespace AeroSim {
namespace Solver {

struct CfdConfig {
    float cfl = 0.5f;
    int max_iter = 20000;
    float convergence_tol = 1e-8f;
    float gamma = 1.4f;
    int checkpoint_interval = 1000;
    bool muscl = true;  // MUSCL reconstruction (reduces dissipation, requires finer mesh)
};

// Per-condition FVM result: integrated forces on the body surface.
struct CfdResult {
    float CX, CY, CZ;
    float Cl, Cm, Cn;
    float CD, CL;
    int iterations;
    float residual;
};

class CfdSolver {
public:
    CfdSolver();
    ~CfdSolver();

    // Upload a tetrahedral mesh to the GPU
    bool load_mesh(const TetMesh& mesh);

    // Allocate persistent scratch buffers (Q, Q0, grad) for warm-start reuse.
    // Once allocated, solve() reuses these instead of per-call alloc/free.
    bool alloc_scratch();
    void free_scratch();

    // Solve a single freestream condition.
    // If scratch buffers are allocated and d_Q_init is not null, uses it as
    // initial condition (warm-start from previous converged solution).
    // d_Q_out (optional): returns device pointer to converged Q for warm-start.
    CfdResult solve(
        float mach, float alpha_deg, float beta_deg,
        float ref_area, float ref_length, float ref_span,
        float com_x, float com_y, float com_z,
        const CfdConfig& cfg = CfdConfig{},
        float* d_Q_init = nullptr,
        float** d_Q_out = nullptr);

    // Batch solve multiple conditions (one Mach×Alpha per block)
    std::vector<CfdResult> solve_batch(
        const std::vector<float>& machs,
        const std::vector<float>& alphas,
        float beta_deg,
        float ref_area, float ref_length, float ref_span,
        float com_x, float com_y, float com_z,
        const CfdConfig& cfg = CfdConfig{});

    // Sequential batch solve with warm-start.
    // Solves one condition at a time, using the previous converged Q as
    // initial condition for the next. Requires less total iterations.
    // Condition order: mach outer, alpha middle, beta inner.
    std::vector<CfdResult> solve_batch_warm(
        const std::vector<float>& machs,
        const std::vector<float>& alphas,
        const std::vector<float>& betas,
        float ref_area, float ref_length, float ref_span,
        float com_x, float com_y, float com_z,
        const CfdConfig& cfg = CfdConfig{});

private:
    float* d_nodes = nullptr;
    int4* d_tets = nullptr;
    int4* d_tet_neighbors = nullptr;
    float* d_tet_volumes = nullptr;
    float* d_tet_centers = nullptr;
    int* d_boundary_type = nullptr;  // per tet face: 0=interior, 1=wall, 2=farfield
    int num_tets = 0;
    int num_nodes = 0;

    // Persistent scratch buffers (allocated by alloc_scratch)
    float* d_persist_Q = nullptr;
    float* d_persist_Q0 = nullptr;
    float* d_persist_grad = nullptr;
    float* d_persist_CX = nullptr;
    float* d_persist_CY = nullptr;
    float* d_persist_CZ = nullptr;
    float* d_persist_Cl = nullptr;
    float* d_persist_Cm = nullptr;
    float* d_persist_Cn = nullptr;
    int*   d_persist_iter = nullptr;
    float* d_persist_res = nullptr;
};

} // namespace Solver
} // namespace AeroSim
