#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/cfd_solver.hpp"
#include "aero_cfd/reconstruction.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static bool near(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::max(std::fabs(a), std::fabs(b)));
}

static int test_residual_equivalence_single_face() {
    TEST("CFD-GPU-1 Euler residual matches CPU reference for interior face");
    {
        CfdMesh mesh;
        mesh.cells.resize(2);
        CfdFace face;
        face.left_cell = 0;
        face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 0.75f;
        face.nx = 0.577350269f;
        face.ny = 0.577350269f;
        face.nz = 0.577350269f;
        mesh.faces.push_back(face);

        PrimitiveState left;
        left.rho = 1.0f; left.u = 2.0f; left.v = -0.2f; left.w = 0.1f; left.p = 0.9f;
        PrimitiveState right;
        right.rho = 0.8f; right.u = -0.4f; right.v = 0.3f; right.w = -0.1f; right.p = 0.7f;

        std::vector<ConservativeState> q;
        q.push_back(primitive_to_conservative(left, 1.4f));
        q.push_back(primitive_to_conservative(right, 1.4f));

        std::vector<EulerFlux> cpu;
        std::vector<EulerFlux> gpu;
        if (!compute_euler_residual_cpu(mesh, q, left, 1.4f, cpu)) FAIL("cpu residual failed");

        std::string error;
        if (!compute_euler_residual_gpu(mesh, q, left, 1.4f, gpu, &error)) FAIL("%s", error.c_str());
        if (gpu.size() != cpu.size()) FAIL("gpu size=%zu cpu size=%zu", gpu.size(), cpu.size());

        for (std::size_t i = 0; i < cpu.size(); ++i) {
            if (!near(gpu[i].mass, cpu[i].mass, 1e-6f)) FAIL("cell=%zu mass gpu=%g cpu=%g", i, gpu[i].mass, cpu[i].mass);
            if (!near(gpu[i].mom_x, cpu[i].mom_x, 1e-6f)) FAIL("cell=%zu mom_x gpu=%g cpu=%g", i, gpu[i].mom_x, cpu[i].mom_x);
            if (!near(gpu[i].mom_y, cpu[i].mom_y, 1e-6f)) FAIL("cell=%zu mom_y gpu=%g cpu=%g", i, gpu[i].mom_y, cpu[i].mom_y);
            if (!near(gpu[i].mom_z, cpu[i].mom_z, 1e-6f)) FAIL("cell=%zu mom_z gpu=%g cpu=%g", i, gpu[i].mom_z, cpu[i].mom_z);
            if (!near(gpu[i].energy, cpu[i].energy, 1e-6f)) FAIL("cell=%zu energy gpu=%g cpu=%g", i, gpu[i].energy, cpu[i].energy);
        }
        PASS;
    }
    return 0;
}

static int test_device_mesh_move() {
    TEST("CFD-GPU-2 DeviceMesh move ownership");
    {
        CfdMesh mesh;
        mesh.cells.resize(2);
        CfdFace face;
        face.left_cell = 0; face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 1.0f; face.nx = 1.0f;
        mesh.faces.push_back(face);

        PrimitiveState left;
        left.rho = 1.0f; left.u = 1.0f; left.p = 1.0f;
        PrimitiveState right = left;
        right.u = -0.5f; right.p = 0.75f;

        std::vector<ConservativeState> q;
        q.push_back(primitive_to_conservative(left, 1.4f));
        q.push_back(primitive_to_conservative(right, 1.4f));

        DeviceMesh mesh1;
        std::string error;
        if (!mesh1.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (mesh1.cell_count() != 2 || mesh1.face_count() != 1) FAIL("counts cells=%zu faces=%zu", mesh1.cell_count(), mesh1.face_count());
        if (mesh1.state_device() == nullptr || mesh1.residual_device() == nullptr) FAIL("missing device buffers");
        if (!mesh1.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (!compute_euler_residual_gpu(mesh1, left, 1.4f, &error)) FAIL("%s", error.c_str());

        DeviceMesh moved = std::move(mesh1);
        if (moved.cell_count() != 2) FAIL("moved cell_count=%zu", moved.cell_count());
        if (mesh1.cell_count() != 0) FAIL("source cell_count=%zu after move", mesh1.cell_count());
        if (mesh1.state_device() != nullptr) FAIL("source state not null after move");

        std::vector<EulerFlux> residual;
        if (!moved.download_residual(residual, &error)) FAIL("%s", error.c_str());
        if (residual.size() != 2) FAIL("residual size=%zu", residual.size());
        if (std::fabs(residual[0].mass + residual[1].mass) > 1e-6f) {
            FAIL("mass conservation residual=[%g,%g]", residual[0].mass, residual[1].mass);
        }
        PASS;
    }
    return 0;
}

static int test_gpu_limiter() {
    TEST("CFD-GPU-3 GPU limiter application matches CPU reference");
    {
        CfdMesh mesh;
        mesh.cells.resize(3);
        CfdFace face;
        face.left_cell = 0; face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 1.0f;
        mesh.faces.push_back(face);

        std::vector<PrimitiveGradient> gradients(3);
        std::vector<PrimitiveLimiter> limiters(3);
        for (int i = 0; i < 3; ++i) {
            gradients[i].drho_dx = 1.0f + static_cast<float>(i);
            gradients[i].drho_dy = -2.0f;
            gradients[i].du_dx = 0.5f;
            gradients[i].dv_dy = -0.25f;
            gradients[i].dw_dz = 0.75f;
            gradients[i].dp_dx = -1.5f;
            gradients[i].dp_dy = 2.5f;
            limiters[i].rho = 0.25f;
            limiters[i].u = 0.5f;
            limiters[i].v = 0.75f;
            limiters[i].w = 1.0f;
            limiters[i].p = 0.1f * static_cast<float>(i + 1);
        }

        DeviceMesh mesh_d;
        std::string error;
        if (!mesh_d.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!mesh_d.upload_gradients(gradients, &error)) FAIL("%s", error.c_str());
        if (!mesh_d.upload_limiters(limiters, &error)) FAIL("%s", error.c_str());
        if (!apply_limiter_gpu(mesh_d, &error)) FAIL("%s", error.c_str());

        std::vector<PrimitiveGradient> gpu;
        if (!mesh_d.download_gradients(gpu, &error)) FAIL("%s", error.c_str());
        if (gpu.size() != gradients.size()) FAIL("gradient size=%zu", gpu.size());
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            auto cpu = apply_limiter(gradients[i], limiters[i]);
            if (!near(gpu[i].drho_dx, cpu.drho_dx, 1e-6f)) FAIL("cell=%zu drho_dx gpu=%g cpu=%g", i, gpu[i].drho_dx, cpu.drho_dx);
            if (!near(gpu[i].du_dx, cpu.du_dx, 1e-6f)) FAIL("cell=%zu du_dx gpu=%g cpu=%g", i, gpu[i].du_dx, cpu.du_dx);
            if (!near(gpu[i].dp_dx, cpu.dp_dx, 1e-6f)) FAIL("cell=%zu dp_dx gpu=%g cpu=%g", i, gpu[i].dp_dx, cpu.dp_dx);
        }
        PASS;
    }
    return 0;
}

static int test_gpu_timing() {
    TEST("CFD-GPU-4 timed Euler residual returns non-negative kernel time");
    {
        CfdMesh mesh;
        mesh.cells.resize(2);
        CfdFace face;
        face.left_cell = 0; face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 0.75f;
        face.nx = 1.0f;
        mesh.faces.push_back(face);

        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.5f; w.p = 1.0f;
        std::vector<ConservativeState> q(2, primitive_to_conservative(w, 1.4f));

        DeviceMesh mesh_d;
        std::string error;
        if (!mesh_d.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!mesh_d.upload_state(q, &error)) FAIL("%s", error.c_str());
        float elapsed_ms = -1.0f;
        if (!compute_euler_residual_gpu_timed(mesh_d, w, 1.4f, &elapsed_ms, &error)) FAIL("%s", error.c_str());
        if (elapsed_ms < 0.0f) FAIL("elapsed_ms=%g", elapsed_ms);
        std::size_t bytes = estimate_euler_residual_gpu_bytes(mesh);
        if (bytes == 0) FAIL("estimated bytes=%zu", bytes);
        float bandwidth_gb_s = elapsed_ms > 0.0f ? (static_cast<float>(bytes) / (elapsed_ms * 1.0e6f)) : 0.0f;
        if (bandwidth_gb_s < 0.0f) FAIL("bandwidth=%g", bandwidth_gb_s);
        PASS;
    }
    return 0;
}

static int test_real_mesh_equivalence() {
    TEST("CFD-GPU-5 Euler residual matches CPU on full cube mesh");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        std::vector<EulerFlux> cpu;
        if (!compute_euler_residual_cpu(mesh, q, w, 1.4f, cpu)) FAIL("cpu residual failed");

        std::vector<EulerFlux> gpu;
        std::string error;
        if (!compute_euler_residual_gpu(mesh, q, w, 1.4f, gpu, &error)) FAIL("%s", error.c_str());

        if (gpu.size() != cpu.size()) FAIL("size mismatch gpu=%zu cpu=%zu", gpu.size(), cpu.size());

        float max_rel = 0.0f;
        int bad_cell = -1;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            float dm = std::fabs(gpu[i].mass - cpu[i].mass) / (1.0f + std::max(std::fabs(gpu[i].mass), std::fabs(cpu[i].mass)));
            if (dm > max_rel) { max_rel = dm; bad_cell = static_cast<int>(i); }
        }
        if (max_rel > 1e-6f) FAIL("max relative diff=%g at cell=%d", max_rel, bad_cell);
        PASS;
    }
    return 0;
}

static int test_gpu_solver_equivalence_cube() {
    TEST("CFD-GPU-6 GPU solver L2 matches CPU after 1 iteration on cube mesh");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 0.0f;
        cond.beta_deg = 0.0f;

        CfdConfig gpu_cfg;
        gpu_cfg.use_gpu = true;
        gpu_cfg.max_iter = 1;

        CfdConfig cpu_cfg = gpu_cfg;
        cpu_cfg.use_gpu = false;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu_result = solver.solve(cond, gpu_cfg);
        if (gpu_result.failed) FAIL("GPU solve failed");

        CfdSolveSummary cpu_result = solver.solve(cond, cpu_cfg);
        if (cpu_result.failed) FAIL("CPU solve failed");

        float gpu_l2 = gpu_result.residual_history.empty() ? 0.0f : gpu_result.residual_history.back();
        float cpu_l2 = cpu_result.residual_history.empty() ? 0.0f : cpu_result.residual_history.back();

        if (std::fabs(gpu_l2 - cpu_l2) > 1e-6f) FAIL("L2 mismatch GPU=%g CPU=%g", gpu_l2, cpu_l2);
        PASS;
    }
    return 0;
}

static int test_gpu_cpu_convergence_match() {
    TEST("CFD-GPU-7 GPU-CPU convergence match: 20 iterations on cube mesh");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.cfl = 0.4f;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-12f;

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 3.0f;
        cond.beta_deg = 0.0f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdConfig gpu_cfg = cfg; gpu_cfg.use_gpu = true;
        CfdConfig cpu_cfg = cfg; cpu_cfg.use_gpu = false;

        CfdSolveSummary gpu_result = solver.solve(cond, gpu_cfg);
        if (gpu_result.failed) FAIL("GPU solve failed");

        CfdSolveSummary cpu_result = solver.solve(cond, cpu_cfg);
        if (cpu_result.failed) FAIL("CPU solve failed");

        if (cpu_result.residual_history.empty() || gpu_result.residual_history.empty()) FAIL("no convergence history");

        std::size_t n = std::min(gpu_result.residual_history.size(), cpu_result.residual_history.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (std::fabs(gpu_result.residual_history[i] - cpu_result.residual_history[i]) > 1e-6f) {
                FAIL("iter=%zu L2 GPU=%g CPU=%g diff=%g", i,
                    gpu_result.residual_history[i], cpu_result.residual_history[i],
                    std::fabs(gpu_result.residual_history[i] - cpu_result.residual_history[i]));
            }
        }

        PASS;
    }
    return 0;
}

static int test_gpu_flat_plate_convergence() {
    TEST("CFD-GPU-8 GPU and CPU converge to similar residual level on flat plate");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 0.0f;
        cond.beta_deg = 0.0f;

        CfdConfig gpu_cfg;
        gpu_cfg.use_gpu = true;
        gpu_cfg.cfl = 0.4f;
        gpu_cfg.max_iter = 500;
        gpu_cfg.convergence_tol = 1e-8f;

        CfdConfig cpu_cfg = gpu_cfg;
        cpu_cfg.use_gpu = false;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu_result = solver.solve(cond, gpu_cfg);
        if (gpu_result.failed) FAIL("GPU solve failed");

        CfdSolveSummary cpu_result = solver.solve(cond, cpu_cfg);
        if (cpu_result.failed) FAIL("CPU solve failed");

        if (cpu_result.residual_history.empty() || gpu_result.residual_history.empty()) FAIL("no convergence history");

        float cpu_l2 = cpu_result.residual_history.back();
        float gpu_l2 = gpu_result.residual_history.back();
        float ratio = cpu_l2 > 0.0f ? gpu_l2 / cpu_l2 : 1.0f;

        if (ratio > 1e3f) FAIL("GPU/Cpu L2 ratio=%g (GPU=%g CPU=%g)", ratio, gpu_l2, cpu_l2);
        if (gpu_l2 > 1.0f) FAIL("GPU L2=%g not converged", gpu_l2);

        float cx_tol = 0.1f;
        if (std::fabs(gpu_result.forces.CX - cpu_result.forces.CX) > cx_tol) FAIL("CX GPU=%g CPU=%g", gpu_result.forces.CX, cpu_result.forces.CX);
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_residual_equivalence_single_face();
    result |= test_device_mesh_move();
    result |= test_gpu_limiter();
    result |= test_gpu_timing();
    result |= test_real_mesh_equivalence();
    result |= test_gpu_solver_equivalence_cube();
    result |= test_gpu_cpu_convergence_match();
    result |= test_gpu_flat_plate_convergence();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
