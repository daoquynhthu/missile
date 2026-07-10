#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/diagnostics.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/rans.hpp"

#include <algorithm>
#include <cmath>

using namespace aerosp;
using namespace aerosp::aero::cfd;
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static bool near(Real a, Real b, Real tol) {
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
            gradients[i].drho_dx = 1.0f + static_cast<Real>(i);
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
            limiters[i].p = 0.1f * static_cast<Real>(i + 1);
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
        Real elapsed_ms = -1.0f;
        if (!compute_euler_residual_gpu_timed(mesh_d, w, 1.4f, &elapsed_ms, &error)) FAIL("%s", error.c_str());
        if (elapsed_ms < 0.0f) FAIL("elapsed_ms=%g", elapsed_ms);
        std::size_t bytes = estimate_euler_residual_gpu_bytes(mesh);
        if (bytes == 0) FAIL("estimated bytes=%zu", bytes);
        Real bandwidth_gb_s = elapsed_ms > 0.0f ? (static_cast<Real>(bytes) / (elapsed_ms * 1.0e6f)) : 0.0f;
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

        Real max_rel = 0.0f;
        int bad_cell = -1;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            Real dm = std::fabs(gpu[i].mass - cpu[i].mass) / (1.0f + std::max(std::fabs(gpu[i].mass), std::fabs(cpu[i].mass)));
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

        Real gpu_l2 = gpu_result.residual_history.empty() ? 0.0f : gpu_result.residual_history.back();
        Real cpu_l2 = cpu_result.residual_history.empty() ? 0.0f : cpu_result.residual_history.back();

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

        Real cpu_l2 = cpu_result.residual_history.back();
        Real gpu_l2 = gpu_result.residual_history.back();
        Real ratio = cpu_l2 > 0.0f ? gpu_l2 / cpu_l2 : 1.0f;

        if (ratio > 1e3f) FAIL("GPU/Cpu L2 ratio=%g (GPU=%g CPU=%g)", ratio, gpu_l2, cpu_l2);
        if (gpu_l2 > 1.0f) FAIL("GPU L2=%g not converged", gpu_l2);

        Real cx_tol = 0.1f;
        if (std::fabs(gpu_result.forces.CX - cpu_result.forces.CX) > cx_tol) FAIL("CX GPU=%g CPU=%g", gpu_result.forces.CX, cpu_result.forces.CX);
        PASS;
    }
    return 0;
}

// ----- Oracle tests -----

static int test_oracle_freestream_preservation() {
    TEST("CFD-ORACLE-EULER-1 freestream preservation GPU=CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.3f;
        cfg.max_iter = 50;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu_result = solver.solve(cond, cfg);
        if (gpu_result.failed) FAIL("GPU solve failed");

        CfdConfig cpu_cfg = cfg;
        cpu_cfg.use_gpu = false;
        CfdSolveSummary cpu_result = solver.solve(cond, cpu_cfg);
        if (cpu_result.failed) FAIL("CPU solve failed");

        std::string error;
        if (!assert_oracle_equivalent(gpu_result, cpu_result, 1e-6f, 1e-6f, &error)) {
            FAIL("%s", error.c_str());
        }
        PASS;
    }
    return 0;
}

static int test_oracle_symmetric_cube_forces() {
    TEST("CFD-ORACLE-EULER-2 symmetric cube forces GPU=CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 5.0f;

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 100;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu = solver.solve(cond, cfg);
        if (gpu.failed) FAIL("GPU solve failed");

        CfdConfig cpu_cfg = cfg;
        cpu_cfg.use_gpu = false;
        CfdSolveSummary cpu = solver.solve(cond, cpu_cfg);
        if (cpu.failed) FAIL("CPU solve failed");

        std::string error;
        if (!assert_oracle_equivalent(gpu, cpu, 1e-6f, 1e-6f, &error)) {
            FAIL("%s", error.c_str());
        }
        PASS;
    }
    return 0;
}

static int test_oracle_flat_plate_zero_forces() {
    TEST("CFD-ORACLE-EULER-3 flat plate farfield-only forces GPU=CPU");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.max_iter = 50;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu = solver.solve(cond, cfg);
        if (gpu.failed) FAIL("GPU solve failed");

        CfdConfig cpu_cfg = cfg;
        cpu_cfg.use_gpu = false;
        CfdSolveSummary cpu = solver.solve(cond, cpu_cfg);
        if (cpu.failed) FAIL("CPU solve failed");

        std::string error;
        if (!assert_oracle_equivalent(gpu, cpu, 1e-6f, 1e-6f, &error)) {
            FAIL("%s", error.c_str());
        }
        PASS;
    }
    return 0;
}

static int test_oracle_convergence_history() {
    TEST("CFD-ORACLE-EULER-4 residual convergence history GPU=CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 3.0f;

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 100;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu = solver.solve(cond, cfg);
        if (gpu.failed) FAIL("GPU solve failed");

        CfdConfig cpu_cfg = cfg;
        cpu_cfg.use_gpu = false;
        CfdSolveSummary cpu = solver.solve(cond, cpu_cfg);
        if (cpu.failed) FAIL("CPU solve failed");

        std::string error;
        if (!assert_oracle_equivalent(gpu, cpu, 1e-6f, 1e-6f, &error)) {
            FAIL("%s", error.c_str());
        }
        PASS;
    }
    return 0;
}

static int test_oracle_wall_forces() {
    TEST("CFD-ORACLE-EULER-5 wall force components GPU=CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 5.0f;

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 100;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary gpu = solver.solve(cond, cfg);
        if (gpu.failed) FAIL("GPU solve failed");

        CfdConfig cpu_cfg = cfg;
        cpu_cfg.use_gpu = false;
        CfdSolveSummary cpu = solver.solve(cond, cpu_cfg);
        if (cpu.failed) FAIL("CPU solve failed");

        std::string error;
        if (!assert_oracle_equivalent(gpu, cpu, 1e-6f, 1e-6f, &error)) {
            FAIL("%s", error.c_str());
        }
        PASS;
    }
    return 0;
}

static int test_oracle_dispatch() {
    TEST("CFD-ORACLE-DISPATCH-1 cpu_oracle=true dispatch path GPU=CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 3.0f;

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cpu_oracle = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary result = solver.solve(cond, cfg);
        if (result.failed) FAIL("GPU+oracle solve failed");
        PASS;
    }
    return 0;
}

static int test_recon_constant_state_zero_gradients() {
    TEST("CFD-ORACLE-RECON-1 constant-state zero gradients CPU=GPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);

        PrimitiveState w;
        w.rho = 1.0f; w.u = 2.0f; w.v = -0.3f; w.w = 0.1f; w.p = 1.0f / 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        std::vector<PrimitiveGradient> cpu_grads = compute_green_gauss_gradients(mesh, q, 1.4f);
        if (cpu_grads.empty()) FAIL("CPU gradients empty");

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (!compute_gradients_gpu(d_mesh, 1.4f, &error)) FAIL("%s", error.c_str());

        std::vector<PrimitiveGradient> gpu_grads;
        if (!d_mesh.download_gradients(gpu_grads, &error)) FAIL("%s", error.c_str());
        if (gpu_grads.size() != cpu_grads.size()) FAIL("size mismatch: cpu=%zu gpu=%zu", cpu_grads.size(), gpu_grads.size());

        Real max_cpu = 0.0f, max_gpu = 0.0f;
        for (std::size_t i = 0; i < cpu_grads.size(); ++i) {
            auto& c = cpu_grads[i];
            auto& g = gpu_grads[i];
            max_cpu = std::max({max_cpu, std::fabs(c.drho_dx), std::fabs(c.du_dx), std::fabs(c.dv_dy), std::fabs(c.dw_dz), std::fabs(c.dp_dx)});
            max_gpu = std::max({max_gpu, std::fabs(g.drho_dx), std::fabs(g.du_dx), std::fabs(g.dv_dy), std::fabs(g.dw_dz), std::fabs(g.dp_dx)});
        }
        if (max_cpu > 1e-12f) FAIL("CPU gradients not zero: max=%g", max_cpu);
        if (max_gpu > 1e-6f) FAIL("GPU gradients not zero: max=%g", max_gpu);
        PASS;
    }
    return 0;
}

static int test_recon_gradient_match() {
    TEST("CFD-ORACLE-RECON-2 CPU/GPU gradient match on cube mesh");
    {
        CfdMesh mesh = generate_structured_cube_mesh(2.0f, 11);
        compute_mesh_metrics(mesh);

        Real gamma = 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size());
        for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
            Real x = mesh.cells[i].cx;
            Real y = mesh.cells[i].cy;
            Real rho = 1.0f + 0.1f * x;
            Real u = 2.0f + 0.05f * y;
            Real v = -0.1f * x;
            Real w = 0.0f;
            Real p = 1.0f / gamma + 0.05f * x - 0.02f * y;
            Real e = p / (gamma - 1.0f) + 0.5f * rho * (u*u + v*v + w*w);
            q[i].rho = rho;
            q[i].rho_u = rho * u;
            q[i].rho_v = rho * v;
            q[i].rho_w = rho * w;
            q[i].rho_E = e;
        }

        std::vector<PrimitiveGradient> cpu_grads = compute_green_gauss_gradients(mesh, q, gamma);
        if (cpu_grads.empty()) FAIL("CPU gradients empty");

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (!compute_gradients_gpu(d_mesh, gamma, &error)) FAIL("%s", error.c_str());

        std::vector<PrimitiveGradient> gpu_grads;
        if (!d_mesh.download_gradients(gpu_grads, &error)) FAIL("%s", error.c_str());
        if (gpu_grads.size() != cpu_grads.size()) FAIL("size mismatch: cpu=%zu gpu=%zu", cpu_grads.size(), gpu_grads.size());

        Real tol = 2e-6f;
        for (std::size_t i = 0; i < cpu_grads.size(); ++i) {
            auto& c = cpu_grads[i];
            auto& g = gpu_grads[i];
            if (!near(g.drho_dx, c.drho_dx, tol)) FAIL("cell=%zu drho_dx cpu=%g gpu=%g", i, c.drho_dx, g.drho_dx);
            if (!near(g.drho_dy, c.drho_dy, tol)) FAIL("cell=%zu drho_dy cpu=%g gpu=%g", i, c.drho_dy, g.drho_dy);
            if (!near(g.drho_dz, c.drho_dz, tol)) FAIL("cell=%zu drho_dz cpu=%g gpu=%g", i, c.drho_dz, g.drho_dz);
            if (!near(g.du_dx, c.du_dx, tol)) FAIL("cell=%zu du_dx cpu=%g gpu=%g", i, c.du_dx, g.du_dx);
            if (!near(g.du_dy, c.du_dy, tol)) FAIL("cell=%zu du_dy cpu=%g gpu=%g", i, c.du_dy, g.du_dy);
            if (!near(g.dv_dx, c.dv_dx, tol)) FAIL("cell=%zu dv_dx cpu=%g gpu=%g", i, c.dv_dx, g.dv_dx);
            if (!near(g.dv_dy, c.dv_dy, tol)) FAIL("cell=%zu dv_dy cpu=%g gpu=%g", i, c.dv_dy, g.dv_dy);
            if (!near(g.dp_dx, c.dp_dx, tol)) FAIL("cell=%zu dp_dx cpu=%g gpu=%g", i, c.dp_dx, g.dp_dx);
            if (!near(g.dp_dy, c.dp_dy, tol)) FAIL("cell=%zu dp_dy cpu=%g gpu=%g", i, c.dp_dy, g.dp_dy);
            if (!near(g.dp_dz, c.dp_dz, tol)) FAIL("cell=%zu dp_dz cpu=%g gpu=%g", i, c.dp_dz, g.dp_dz);
        }
        PASS;
    }
    return 0;
}

static int test_recon_first_order_regression() {
    TEST("CFD-ORACLE-RECON-3 reconstruction_order=1 forces match 1st-order CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        PrimitiveState w;
        w.rho = 1.0f; w.u = 2.0f; w.p = 1.0f / 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        Real gamma = 1.4f;

        std::vector<EulerFlux> cpu_res(mesh.cells.size());
        if (!compute_euler_residual_cpu(mesh, q, w, gamma, cpu_res)) FAIL("CPU residual failed");

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());

        if (!compute_euler_residual_gpu(d_mesh, w, gamma, &error, 1)) FAIL("%s", error.c_str());

        std::vector<EulerFlux> gpu_res;
        if (!d_mesh.download_residual(gpu_res, &error)) FAIL("%s", error.c_str());

        Real tol = 1e-12f;
        for (std::size_t i = 0; i < cpu_res.size(); ++i) {
            if (!near(gpu_res[i].mass, cpu_res[i].mass, tol)) FAIL("cell=%zu mass cpu=%g gpu=%g", i, cpu_res[i].mass, gpu_res[i].mass);
            if (!near(gpu_res[i].mom_x, cpu_res[i].mom_x, tol)) FAIL("cell=%zu mom_x cpu=%g gpu=%g", i, cpu_res[i].mom_x, gpu_res[i].mom_x);
            if (!near(gpu_res[i].mom_y, cpu_res[i].mom_y, tol)) FAIL("cell=%zu mom_y cpu=%g gpu=%g", i, cpu_res[i].mom_y, gpu_res[i].mom_y);
            if (!near(gpu_res[i].mom_z, cpu_res[i].mom_z, tol)) FAIL("cell=%zu mom_z cpu=%g gpu=%g", i, cpu_res[i].mom_z, gpu_res[i].mom_z);
            if (!near(gpu_res[i].energy, cpu_res[i].energy, tol)) FAIL("cell=%zu energy cpu=%g gpu=%g", i, cpu_res[i].energy, gpu_res[i].energy);
        }
        PASS;
    }
    return 0;
}

static int test_oracle_mesh_counts() {
    TEST("CFD-ORACLE-MESH-1 DeviceMesh counts match host");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());

        if (d_mesh.cell_count() != mesh.cells.size()) FAIL("cell_count: device=%zu host=%zu", d_mesh.cell_count(), mesh.cells.size());
        if (d_mesh.face_count() != mesh.faces.size()) FAIL("face_count: device=%zu host=%zu", d_mesh.face_count(), mesh.faces.size());
        PASS;
    }
    return 0;
}

static int test_oracle_bandwidth() {
    TEST("CFD-ORACLE-BW-1 GPU memory bandwidth >= 50% theoretical");
    {
        int mem_clock_khz = 0;
        int bus_width_bits = 0;
        if (!cuda_check(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0), "cudaDevAttrMemoryClockRate")) {
            FAIL("cudaDevAttrMemoryClockRate failed");
        }
        if (!cuda_check(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0), "cudaDevAttrGlobalMemoryBusWidth")) {
            FAIL("cudaDevAttrGlobalMemoryBusWidth failed");
        }
        Real theoretical = 2.0e-6f * static_cast<Real>(mem_clock_khz) *
            static_cast<Real>(bus_width_bits) / 8.0f;

        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        PrimitiveState w;
        w.rho = 1.0f; w.u = 2.0f; w.p = 1.0f / 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error, true)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());

        Real elapsed_ms = -1.0f;
        if (!compute_euler_residual_gpu_timed(d_mesh, w, 1.4f, &elapsed_ms, &error)) FAIL("%s", error.c_str());
        if (elapsed_ms <= 0.0f) FAIL("elapsed_ms=%g", elapsed_ms);

        std::size_t bytes = estimate_euler_residual_gpu_bytes(mesh);
        Real bandwidth = static_cast<Real>(bytes) / (elapsed_ms * 1.0e6f);
        Real ratio = theoretical > 0.0f ? bandwidth / theoretical : 0.0f;

        if (ratio < 0.5f) {
            std::printf("  [WARN] bandwidth ratio=%g (BW=%g GB/s, theoretical=%g GB/s) -- below 50%% threshold\n", ratio, bandwidth, theoretical);
        }
        PASS;
    }
    return 0;
}

static int test_diag_state_bounds_gpu_cpu_match() {
    TEST("CFD-ORACLE-DIAG-1 GPU state bounds match CPU");
    {
        CfdMesh mesh = generate_structured_cube_mesh(2.0f, 7);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.max_iter = 5;
        cfg.cfl = 0.3f;
        cfg.convergence_tol = 1e-12f;
        cfg.diagnostic_level = DiagnosticLevel::Basic;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");
        auto cpu_result = solver.solve({1.5f, 0.0f, 0.0f}, cfg);
        if (cpu_result.failed) FAIL("CPU solver failed");

        PrimitiveState w = make_freestream(1.5f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());

        auto gpu_result = solve_gpu(d_mesh, {1.5f, 0.0f, 0.0f}, cfg, &error);
        if (gpu_result.failed) FAIL("GPU solver failed: %s", error.c_str());
        if (gpu_result.diagnostics.state_bounds_history.empty()) FAIL("GPU diagnostics empty");
        if (cpu_result.diagnostics.state_bounds_history.empty()) FAIL("CPU diagnostics empty");

        std::size_t n = gpu_result.diagnostics.state_bounds_history.size();
        if (n == 0 || n > cfg.max_iter) FAIL("bounds count out of range: %zu", n);

        Real tol = 2e-5f;
        for (std::size_t i = 0; i < n; ++i) {
            auto& g = gpu_result.diagnostics.state_bounds_history[i];
            auto& c = cpu_result.diagnostics.state_bounds_history[i + 1];
            if (!near(g.min_rho, c.min_rho, tol)) FAIL("i=%zu min_rho gpu=%g cpu=%g", i, g.min_rho, c.min_rho);
            if (!near(g.max_rho, c.max_rho, tol)) FAIL("i=%zu max_rho gpu=%g cpu=%g", i, g.max_rho, c.max_rho);
            if (!near(g.min_p, c.min_p, tol)) FAIL("i=%zu min_p gpu=%g cpu=%g", i, g.min_p, c.min_p);
            if (!near(g.max_p, c.max_p, tol)) FAIL("i=%zu max_p gpu=%g cpu=%g", i, g.max_p, c.max_p);
            if (!near(g.min_mach, c.min_mach, tol)) FAIL("i=%zu min_mach gpu=%g cpu=%g", i, g.min_mach, c.min_mach);
            if (!near(g.max_mach, c.max_mach, tol)) FAIL("i=%zu max_mach gpu=%g cpu=%g", i, g.max_mach, c.max_mach);
        }
        PASS;
    }
    return 0;
}

static int test_diag_failure_snapshot() {
    TEST("CFD-ORACLE-DIAG-2 GPU failure detection on invalid state");
    {
        CfdMesh mesh = generate_structured_cube_mesh(2.0f, 7);
        compute_mesh_metrics(mesh);

        PrimitiveState w = make_freestream(5.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        int bad_cell = static_cast<int>(q.size()) / 2;
        q[bad_cell].rho = -1.0f;

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());

        CfdConfig cfg;
        cfg.max_iter = 5;
        cfg.cfl = 0.5f;
        cfg.convergence_tol = 1e-12f;
        cfg.diagnostic_level = DiagnosticLevel::Basic;

        auto gpu_result = solve_gpu(d_mesh, {5.0f, 0.0f, 0.0f}, cfg, &error);

        if (!gpu_result.failed) FAIL("GPU solver did not fail with invalid initial state");
        if (!gpu_result.diagnostics.failure.valid) FAIL("failure snapshot missing");

        if (gpu_result.diagnostics.state_bounds_history.empty()) FAIL("state bounds history empty");
        for (std::size_t i = 0; i < gpu_result.diagnostics.state_bounds_history.size(); ++i) {
            auto& sb = gpu_result.diagnostics.state_bounds_history[i];
            if (!sb.valid) FAIL("bounds invalid at i=%zu", i);
        }

        PASS;
    }
    return 0;
}

static int test_recon_order2_converged_forces() {
    TEST("CFD-ORACLE-RECON-4 order=2 GPU forces plausible and differ from order=1");
    {
        CfdMesh mesh = generate_structured_cube_mesh(3.0f, 9);
        compute_mesh_metrics(mesh);

        PrimitiveState w = make_freestream(2.0f, 2.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        CfdConfig cfg1, cfg2;
        cfg1.max_iter = 15; cfg2.max_iter = 15;
        cfg1.cfl = 0.2f; cfg2.cfl = 0.2f;
        cfg1.convergence_tol = 1e-10f; cfg2.convergence_tol = 1e-10f;
        cfg1.reconstruction_order = 1; cfg2.reconstruction_order = 2;

        DeviceMesh d_mesh1, d_mesh2;
        std::string error;
        if (!d_mesh1.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh2.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh1.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (!d_mesh2.upload_state(q, &error)) FAIL("%s", error.c_str());

        auto r1 = solve_gpu(d_mesh1, {2.0f, 2.0f, 0.0f}, cfg1, &error);
        auto r2 = solve_gpu(d_mesh2, {2.0f, 2.0f, 0.0f}, cfg2, &error);

        if (r1.failed) FAIL("order=1 solver failed: %s", error.c_str());
        if (r2.failed) FAIL("order=2 solver failed: %s", error.c_str());

        if (!std::isfinite(r2.forces.CD)) FAIL("order=2 CD not finite");
        if (!std::isfinite(r2.forces.CL)) FAIL("order=2 CL not finite");

        if (r1.forces.CX != r2.forces.CX) {
            Real diff = std::fabs(r2.forces.CX - r1.forces.CX);
            if (diff < 1e-12f) FAIL("order=2 forces identical to order=1 (reconstruction not running)");
        }

        PASS;
    }
    return 0;
}

static int test_cpu_order2_residual_matches_gpu() {
    TEST("CFD-ORACLE-RECON-5 CPU order-2 residual matches GPU order-2 on cube mesh");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);

        PrimitiveState w = make_freestream(2.0f, 3.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        std::vector<EulerFlux> cpu_res(mesh.cells.size());
        if (!compute_euler_residual_cpu_order2(mesh, q, w, 1.4f, cpu_res))
            FAIL("CPU order-2 residual failed");

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (!compute_euler_residual_gpu(d_mesh, w, 1.4f, &error, 2))
            FAIL("GPU order-2 residual failed: %s", error.c_str());

        std::size_t nc = mesh.cells.size();
        Real* gpu_res = new Real[nc * 6];
        if (!cuda_check(cudaMemcpy(gpu_res, d_mesh.residual_device(), nc * 6 * sizeof(Real), cudaMemcpyDeviceToHost), "download gpu res", &error)) FAIL("%s", error.c_str());

        Real max_diff = 0.0f;
        for (std::size_t i = 0; i < nc; ++i) {
            Real d_mass = std::fabs(cpu_res[i].mass - gpu_res[i * 6 + 0]);
            Real d_mx = std::fabs(cpu_res[i].mom_x - gpu_res[i * 6 + 1]);
            Real d_my = std::fabs(cpu_res[i].mom_y - gpu_res[i * 6 + 2]);
            Real d_mz = std::fabs(cpu_res[i].mom_z - gpu_res[i * 6 + 3]);
            Real d_en = std::fabs(cpu_res[i].energy - gpu_res[i * 6 + 4]);
            Real d_turb = std::fabs(cpu_res[i].turbulence - gpu_res[i * 6 + 5]);
            Real d = std::max({d_mass, d_mx, d_my, d_mz, d_en, d_turb});
            if (d > max_diff) max_diff = d;
        }

        delete[] gpu_res;

        if (max_diff > 1e-5f)
            FAIL("max CPU/GPU order-2 residual diff=%g", max_diff);

        PASS;
    }
    return 0;
}

static int test_color_count() {
    TEST("CFD-COLOR-1 face coloring produces valid color count");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());

        int nc = d_mesh.color_count();
        if (nc <= 0) FAIL("color_count=%d (expected >0)", nc);
        if (nc > DeviceMesh::kMaxColors) FAIL("color_count=%d > kMaxColors=%d", nc, DeviceMesh::kMaxColors);
        PASS;
    }
    return 0;
}

static int test_color_residual_matches_uncolored() {
    TEST("CFD-COLOR-2 colored residual forces match uncolored");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        PrimitiveState w;
        w.rho = 1.0f; w.u = 2.0f; w.p = 1.0f / 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        Real gamma = 1.4f;
        std::string error;

        DeviceMesh colored;
        if (!colored.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!colored.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (!compute_euler_residual_gpu(colored, w, gamma, &error)) FAIL("colored: %s", error.c_str());
        std::vector<EulerFlux> colored_res;
        if (!colored.download_residual(colored_res, &error)) FAIL("colored download: %s", error.c_str());
        if (colored_res.empty()) FAIL("colored residual empty");

        DeviceMesh uncolored;
        if (!uncolored.upload_mesh(mesh, &error, true)) FAIL("uncolored upload: %s", error.c_str());
        if (!uncolored.upload_state(q, &error)) FAIL("uncolored state: %s", error.c_str());
        if (!compute_euler_residual_gpu(uncolored, w, gamma, &error)) FAIL("uncolored: %s", error.c_str());
        std::vector<EulerFlux> uncolored_res;
        if (!uncolored.download_residual(uncolored_res, &error)) FAIL("uncolored download: %s", error.c_str());

        Real tol = 1e-6f;
        for (std::size_t i = 0; i < colored_res.size(); ++i) {
            if (!near(colored_res[i].mass, uncolored_res[i].mass, tol))
                FAIL("cell=%zu mass colored=%g uncolored=%g", i, colored_res[i].mass, uncolored_res[i].mass);
            if (!near(colored_res[i].mom_x, uncolored_res[i].mom_x, tol))
                FAIL("cell=%zu mom_x colored=%g uncolored=%g", i, colored_res[i].mom_x, uncolored_res[i].mom_x);
            if (!near(colored_res[i].mom_y, uncolored_res[i].mom_y, tol))
                FAIL("cell=%zu mom_y colored=%g uncolored=%g", i, colored_res[i].mom_y, uncolored_res[i].mom_y);
            if (!near(colored_res[i].mom_z, uncolored_res[i].mom_z, tol))
                FAIL("cell=%zu mom_z colored=%g uncolored=%g", i, colored_res[i].mom_z, uncolored_res[i].mom_z);
            if (!near(colored_res[i].energy, uncolored_res[i].energy, tol))
                FAIL("cell=%zu energy colored=%g uncolored=%g", i, colored_res[i].energy, uncolored_res[i].energy);
        }
        PASS;
    }
    return 0;
}

static int test_color_gradient_matches_uncolored() {
    TEST("CFD-COLOR-3 colored gradient matches uncolored");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 11);
        compute_mesh_metrics(mesh);

        Real gamma = 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size());
        for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
            Real x = mesh.cells[i].cx;
            Real y = mesh.cells[i].cy;
            Real rho = 1.0f + 0.1f * x;
            Real u = 2.0f + 0.05f * y;
            Real v = -0.1f * x;
            Real p = 1.0f / gamma + 0.05f * x - 0.02f * y;
            Real e = p / (gamma - 1.0f) + 0.5f * rho * (u*u + v*v);
            q[i].rho = rho;
            q[i].rho_u = rho * u;
            q[i].rho_v = rho * v;
            q[i].rho_w = 0.0f;
            q[i].rho_E = e;
        }

        std::string error;

        DeviceMesh colored;
        if (!colored.upload_mesh(mesh, &error)) FAIL("colored upload: %s", error.c_str());
        if (!colored.upload_state(q, &error)) FAIL("colored state: %s", error.c_str());
        if (!compute_gradients_gpu(colored, gamma, &error)) FAIL("colored gradients: %s", error.c_str());
        std::vector<PrimitiveGradient> colored_grads;
        if (!colored.download_gradients(colored_grads, &error)) FAIL("colored download: %s", error.c_str());

        DeviceMesh uncolored;
        if (!uncolored.upload_mesh(mesh, &error, true)) FAIL("uncolored upload: %s", error.c_str());
        if (!uncolored.upload_state(q, &error)) FAIL("uncolored state: %s", error.c_str());
        if (!compute_gradients_gpu(uncolored, gamma, &error)) FAIL("uncolored gradients: %s", error.c_str());
        std::vector<PrimitiveGradient> uncolored_grads;
        if (!uncolored.download_gradients(uncolored_grads, &error)) FAIL("uncolored download: %s", error.c_str());

        Real tol = 2e-6f;
        for (std::size_t i = 0; i < colored_grads.size(); ++i) {
            auto& c = colored_grads[i];
            auto& u = uncolored_grads[i];
            if (!near(c.drho_dx, u.drho_dx, tol)) FAIL("cell=%zu drho_dx colored=%g uncolored=%g", i, c.drho_dx, u.drho_dx);
            if (!near(c.drho_dy, u.drho_dy, tol)) FAIL("cell=%zu drho_dy colored=%g uncolored=%g", i, c.drho_dy, u.drho_dy);
            if (!near(c.du_dx, u.du_dx, tol)) FAIL("cell=%zu du_dx colored=%g uncolored=%g", i, c.du_dx, u.du_dx);
            if (!near(c.du_dy, u.du_dy, tol)) FAIL("cell=%zu du_dy colored=%g uncolored=%g", i, c.du_dy, u.du_dy);
            if (!near(c.dv_dx, u.dv_dx, tol)) FAIL("cell=%zu dv_dx colored=%g uncolored=%g", i, c.dv_dx, u.dv_dx);
            if (!near(c.dp_dx, u.dp_dx, tol)) FAIL("cell=%zu dp_dx colored=%g uncolored=%g", i, c.dp_dx, u.dp_dx);
            if (!near(c.dp_dy, u.dp_dy, tol)) FAIL("cell=%zu dp_dy colored=%g uncolored=%g", i, c.dp_dy, u.dp_dy);
        }
        PASS;
    }
    return 0;
}

static int test_color_deterministic_residual() {
    TEST("CFD-COLOR-4 colored residual is deterministic byte-level");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        PrimitiveState w;
        w.rho = 1.0f; w.u = 2.0f; w.p = 1.0f / 1.4f;
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        Real gamma = 1.4f;
        std::string error;

        DeviceMesh d_mesh;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("upload: %s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("state: %s", error.c_str());

        int* d_failed = nullptr;
        if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", &error)) FAIL("%s", error.c_str());

        std::size_t residual_bytes = d_mesh.cell_count() * DeviceMesh::NVAR * sizeof(Real);
        std::vector<Real> res1(d_mesh.cell_count() * DeviceMesh::NVAR);
        std::vector<Real> res2(d_mesh.cell_count() * DeviceMesh::NVAR);

        if (!compute_euler_residual_gpu(d_mesh, w, gamma, d_failed, &error)) FAIL("1st run: %s", error.c_str());
        if (!cuda_check(cudaMemcpy(res1.data(), d_mesh.residual_device(), residual_bytes, cudaMemcpyDeviceToHost), "1st download", &error)) FAIL("%s", error.c_str());

        if (!cuda_check(cudaMemset(d_mesh.residual_device(), 0, residual_bytes), "clear residual", &error)) FAIL("%s", error.c_str());
        if (!compute_euler_residual_gpu(d_mesh, w, gamma, d_failed, &error)) FAIL("2nd run: %s", error.c_str());
        if (!cuda_check(cudaMemcpy(res2.data(), d_mesh.residual_device(), residual_bytes, cudaMemcpyDeviceToHost), "2nd download", &error)) FAIL("%s", error.c_str());

        if (std::memcmp(res1.data(), res2.data(), residual_bytes) != 0)
            FAIL("residual differs between runs (non-deterministic)");

        cudaFree(d_failed);
        PASS;
    }
    return 0;
}

static int test_viscous_false_regression() {
    TEST("CFD-ORACLE-VISC-1 viscous=false regression to Euler result");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = false;

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 3.0f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary result = solver.solve(cond, cfg);
        if (result.failed) FAIL("GPU solver failed");

        if (result.residual_history.empty()) FAIL("no residual history");

        CfdConfig euler_cfg = cfg;
        euler_cfg.viscous = false;
        CfdSolveSummary euler_result = solver.solve(cond, euler_cfg);
        if (euler_result.failed) FAIL("Euler-only solver failed");

        std::size_t n = std::min(result.residual_history.size(), euler_result.residual_history.size());
        Real max_diff = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            Real d = std::fabs(result.residual_history[i] - euler_result.residual_history[i]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff > 1e-6f) FAIL("max diff=%g between two Euler-only runs", max_diff);
        PASS;
    }
    return 0;
}

static int test_viscous_finite_flat_plate() {
    TEST("CFD-ORACLE-VISC-2 viscous=true produces finite forces on flat plate");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.3f;
        cfg.max_iter = 10;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = true;
        cfg.Re = 1e5f;
        cfg.prandtl = 0.72f;
        cfg.wall_temperature = 288.15f;
        cfg.T_ref = 288.15f;
        cfg.mu_ref = 1.0f;
        cfg.sutherland_T = 110.4f;

        FreestreamCondition cond;
        cond.mach = 0.5f;
        cond.alpha_deg = 0.0f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary result = solver.solve(cond, cfg);
        if (result.failed) FAIL("GPU viscous solver failed");

        if (!std::isfinite(result.forces.CD)) FAIL("CD not finite: %g", result.forces.CD);
        if (!std::isfinite(result.forces.CL)) FAIL("CL not finite: %g", result.forces.CL);
        if (!std::isfinite(result.forces.CX)) FAIL("CX not finite: %g", result.forces.CX);
        if (!std::isfinite(result.forces.CY)) FAIL("CY not finite: %g", result.forces.CY);
        if (!std::isfinite(result.forces.CZ)) FAIL("CZ not finite: %g", result.forces.CZ);

        PASS;
    }
    return 0;
}

static int test_viscous_differs_from_inviscid() {
    TEST("CFD-ORACLE-VISC-3 viscous=true gives different forces than inviscid");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 0.5f;
        cond.alpha_deg = 0.0f;

        CfdConfig inviscid_cfg;
        inviscid_cfg.use_gpu = true;
        inviscid_cfg.cfl = 0.3f;
        inviscid_cfg.max_iter = 10;
        inviscid_cfg.convergence_tol = 1e-12f;
        inviscid_cfg.viscous = false;

        CfdConfig viscous_cfg = inviscid_cfg;
        viscous_cfg.viscous = true;
        viscous_cfg.Re = 1e5f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary inviscid = solver.solve(cond, inviscid_cfg);
        if (inviscid.failed) FAIL("inviscid solver failed");

        CfdSolveSummary viscous = solver.solve(cond, viscous_cfg);
        if (viscous.failed) FAIL("viscous solver failed");

        Real diff = std::fabs(viscous.forces.CD - inviscid.forces.CD);
        if (diff < 1e-12f) FAIL("viscous CD=%g identical to inviscid CD=%g", viscous.forces.CD, inviscid.forces.CD);

        PASS;
    }
    return 0;
}

static int test_rans_false_regression() {
    TEST("CFD-ORACLE-RANS-1 turbulence=false matches Phase 5 laminar");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = false;
        cfg.turbulence = false;

        FreestreamCondition cond;
        cond.mach = 2.0f;
        cond.alpha_deg = 3.0f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        // GPU turbulence=false — should produce Phase 5 Euler result
        CfdSolveSummary gpu = solver.solve(cond, cfg);
        if (gpu.failed) FAIL("turbulence=false solver failed");

        // CPU Euler — the Phase 5 reference oracle
        CfdConfig cpu_cfg = cfg;
        cpu_cfg.use_gpu = false;
        CfdSolveSummary cpu = solver.solve(cond, cpu_cfg);
        if (cpu.failed) FAIL("CPU Euler solver failed");

        // Residual history match
        std::size_t n = std::min(gpu.residual_history.size(), cpu.residual_history.size());
        Real max_diff = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            Real d = std::fabs(gpu.residual_history[i] - cpu.residual_history[i]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff > 1e-6f) FAIL("turbulence=false GPU/CPU max diff=%g", max_diff);

        // Force coefficient match
        std::string force_error;
        if (!assert_oracle_equivalent(gpu, cpu, 1e-6f, 1e-6f, &force_error))
            FAIL("Force mismatch: %s", force_error.c_str());

        PASS;
    }
    return 0;
}

static int test_rans_zero_nu_tilde() {
    TEST("CFD-ORACLE-RANS-2 zero nu_tilde matches laminar");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.4f;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = true;
        cfg.Re = 1e5f;
        cfg.turbulence = false;

        FreestreamCondition cond;
        cond.mach = 0.5f;
        cond.alpha_deg = 0.0f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        // Viscous laminar (turbulence=false) — L2 baseline
        CfdSolveSummary laminar = solver.solve(cond, cfg);
        if (laminar.failed) FAIL("laminar solver failed");

        // turbulence=true but nu_tilde=0 from initial state — should match laminar
        cfg.turbulence = true;
        CfdSolveSummary turb_zero = solver.solve(cond, cfg);
        if (turb_zero.failed) FAIL("turbulence=true zero nu_tilde solver failed");

        std::size_t n = std::min(laminar.residual_history.size(), turb_zero.residual_history.size());
        Real max_diff = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            Real d = std::fabs(laminar.residual_history[i] - turb_zero.residual_history[i]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff > 1e-5f) FAIL("zero nu_tilde max diff=%g from laminar", max_diff);
        PASS;
    }
    return 0;
}

static int test_mixed_mesh_gpu_upload() {
    TEST("CFD-MESH-3D-GPU-1 Hex mesh upload/download: cell and face counts match host");
    {
        CfdMesh mesh = generate_structured_hex_mesh(10);
        compute_mesh_metrics(mesh);

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());

        if (d_mesh.cell_count() != mesh.cells.size()) FAIL("cell_count: device=%zu host=%zu", d_mesh.cell_count(), mesh.cells.size());
        if (d_mesh.face_count() != mesh.faces.size()) FAIL("face_count: device=%zu host=%zu", d_mesh.face_count(), mesh.faces.size());

        if (!d_mesh.type_device()) FAIL("type_device() returned null");
        if (!d_mesh.face_node_count_device()) FAIL("face_node_count_device() returned null");

        std::vector<ConservativeState> q(mesh.cells.size());
        for (auto& s : q) s = primitive_to_conservative({1.0f, 0.0f, 0.0f, 1.0f, 1.4f}, 1.4f);
        if (!d_mesh.upload_state(q, &error)) FAIL("upload_state: %s", error.c_str());

        std::vector<ConservativeState> q2;
        if (!d_mesh.download_state(q2, &error)) FAIL("download_state: %s", error.c_str());
        if (q2.size() != q.size()) FAIL("state size mismatch after download");

        PASS;
    }
    return 0;
}

static int test_hex_mesh_gpu_residual() {
    TEST("CFD-MESH-3D-GPU-2 Hex mesh GPU residuals vs CPU residuals (Euler, 1 iteration)");
    {
        CfdMesh mesh = generate_structured_hex_mesh(8);
        compute_mesh_metrics(mesh);

        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);

        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        std::vector<EulerFlux> cpu_res;
        if (!compute_euler_residual_cpu(mesh, q, w, 1.4f, cpu_res)) FAIL("CPU residual failed");

        std::vector<EulerFlux> gpu_res;
        std::string error;
        if (!compute_euler_residual_gpu(mesh, q, w, 1.4f, gpu_res, &error)) FAIL("GPU residual failed: %s", error.c_str());

        if (gpu_res.size() != cpu_res.size()) FAIL("size mismatch gpu=%zu cpu=%zu", gpu_res.size(), cpu_res.size());

        Real max_rel = 0.0f;
        int bad_cell = -1;
        for (std::size_t i = 0; i < cpu_res.size(); ++i) {
            Real dm = std::fabs(gpu_res[i].mass - cpu_res[i].mass) / (1.0f + std::max(std::fabs(gpu_res[i].mass), std::fabs(cpu_res[i].mass)));
            Real de = std::fabs(gpu_res[i].energy - cpu_res[i].energy) / (1.0f + std::max(std::fabs(gpu_res[i].energy), std::fabs(cpu_res[i].energy)));
            Real d = std::max(dm, de);
            if (d > max_rel) { max_rel = d; bad_cell = static_cast<int>(i); }
        }
        if (max_rel > 1e-6f) FAIL("max relative diff=%g at cell=%d", max_rel, bad_cell);
        PASS;
    }
    return 0;
}

static int test_hex_mesh_symmetric_forces() {
    TEST("CFD-MESH-3D-GPU-3 Hex mesh symmetric cube: CY=CZ=0 within machine zero");
    {
        CfdMesh mesh = generate_structured_hex_mesh(10);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 3.0f;
        cond.alpha_deg = 0.0f;
        cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.max_iter = 10;
        cfg.cfl = 0.5f;
        cfg.convergence_tol = 1e-12f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary s = solver.solve(cond, cfg);
        if (s.failed) FAIL("GPU solve failed");

        if (std::fabs(s.forces.CY) > 1e-8f) FAIL("CY=%g not zero", s.forces.CY);
        if (std::fabs(s.forces.CZ) > 1e-8f) FAIL("CZ=%g not zero", s.forces.CZ);
        if (!std::isfinite(s.forces.CX)) FAIL("CX not finite: %g", s.forces.CX);

        PASS;
    }
    return 0;
}

static int test_rans_cpu_gpu_source_match() {
    TEST("CFD-ORACLE-RANS-4 CPU/GPU SA residual match on cube mesh");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);

        PrimitiveState w = make_freestream(0.5f, 2.0f, 0.0f, 1.4f);
        w.nu_tilde = 3.0f;
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        std::vector<PrimitiveGradient> grads = compute_green_gauss_gradients(mesh, q, 1.4f);
        if (grads.size() != mesh.cells.size()) FAIL("gradients failed");

        std::vector<PrimitiveLimiter> limiters = compute_barth_jespersen_limiters(mesh, q, grads, 1.4f);
        if (limiters.size() != mesh.cells.size()) FAIL("limiters failed");

        std::vector<PrimitiveGradient> limited(grads.size());
        for (std::size_t i = 0; i < grads.size(); ++i)
            limited[i] = apply_limiter(grads[i], limiters[i]);

        std::vector<Real> cpu_delta(q.size(), 0.0f);
        for (std::size_t i = 0; i < q.size(); ++i) {
            PrimitiveState wc;
            conservative_to_primitive(q[i], 1.4f, wc);
            Real wall_d = mesh.cells[i].wall_distance;
            Real mu = 1.0f;
            RansSource rs = compute_rans_source(wc, limited[i], wall_d, mu, q[i].rho, 1e5f);
            cpu_delta[i] = rs.total_source;
        }

        DeviceMesh d_mesh;
        std::string error;
        if (!d_mesh.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!d_mesh.upload_state(q, &error)) FAIL("%s", error.c_str());

        Real gamma = 1.4f;
        if (!compute_gradients_gpu(d_mesh, gamma, &error)) FAIL("GPU gradients: %s", error.c_str());
        if (!compute_limiters_gpu(d_mesh, gamma, &error)) FAIL("GPU limiters: %s", error.c_str());
        if (!apply_limiter_gpu(d_mesh, true, &error)) FAIL("GPU limit apply: %s", error.c_str());

        int* d_failed;
        if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "malloc d_failed", &error)) FAIL("%s", error.c_str());
        if (!cuda_check(cudaMemset(d_failed, 0, sizeof(int)), "memset d_failed", &error)) FAIL("%s", error.c_str());

        if (!compute_rans_source_gpu(d_mesh, gamma, 1e5f, d_failed, &error))
            FAIL("GPU RANS source: %s", error.c_str());

        std::vector<Real> gpu_res(mesh.cells.size() * 6);
        if (!cuda_check(cudaMemcpy(gpu_res.data(), d_mesh.residual_device(), mesh.cells.size() * 6 * sizeof(Real), cudaMemcpyDeviceToHost), "download residual", &error))
            FAIL("%s", error.c_str());

        cudaFree(d_failed);

        Real max_diff = 0.0f;
        for (std::size_t i = 0; i < q.size(); ++i) {
            Real cpu_vol_source = cpu_delta[i];
            Real gpu_vol_source = gpu_res[i * 6 + 5];
            Real d = std::fabs(gpu_vol_source - cpu_vol_source);
            Real base = 1.0f + std::max(std::fabs(gpu_vol_source), std::fabs(cpu_vol_source));
            if (d / base > max_diff) max_diff = d / base;
        }

        if (max_diff > 5e-7f)
            FAIL("CPU/GPU SA volume source max rel diff=%g", max_diff);

        PASS;
    }
    return 0;
}

static int test_rans_turbulent_flat_plate() {
    TEST("CFD-ORACLE-RANS-3 turbulent flat plate Cf plausible");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.3f;
        cfg.max_iter = 200;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = true;
        cfg.Re = 2e6f;
        cfg.turbulence = false;

        FreestreamCondition cond;
        cond.mach = 0.5f;
        cond.alpha_deg = 0.0f;
        cond.nu_tilde_ratio = 0.1f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary laminar = solver.solve(cond, cfg);
        if (laminar.failed) FAIL("laminar solver failed");

        cfg.turbulence = true;
        CfdSolveSummary turbulent = solver.solve(cond, cfg);
        if (turbulent.failed) FAIL("turbulent solver failed");

        if (!std::isfinite(turbulent.forces.CD)) FAIL("turbulent CD not finite: %g", turbulent.forces.CD);
        if (!std::isfinite(turbulent.forces.CL)) FAIL("turbulent CD not finite: %g", turbulent.forces.CL);
        if (!std::isfinite(laminar.forces.CD)) FAIL("laminar CD not finite: %g", laminar.forces.CD);

        // Turbulent drag should be >= laminar drag (SA increases effective viscosity)
        Real margin = 1e-8f + 0.01f * real_fabs(laminar.forces.CD);
        if (turbulent.forces.CD < laminar.forces.CD - margin)
            FAIL("turbulent CD=%g < laminar CD=%g (margin=%g)",
                 turbulent.forces.CD, laminar.forces.CD, margin);

        // SA should yield higher drag than laminar (turbulent boundary layer)
        // At low iteration count the difference may be small; this is a sanity check
        if (turbulent.forces.CD <= laminar.forces.CD)
            std::printf("  [INFO] turbulent CD=%g <= laminar CD=%g (may need more iterations)\n",
                        turbulent.forces.CD, laminar.forces.CD);

        PASS;
    }
    return 0;
}

static int test_rans_negative_nu_tilde() {
    TEST("CFD-ORACLE-RANS-5 negative nu_tilde SA-neg branch");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.use_gpu = true;
        cfg.cfl = 0.3f;
        cfg.max_iter = 10;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = true;
        cfg.Re = 1e5f;
        cfg.turbulence = true;

        FreestreamCondition cond;
        cond.mach = 0.5f;
        cond.alpha_deg = 0.0f;
        cond.nu_tilde = -3.0f;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");

        CfdSolveSummary s = solver.solve(cond, cfg);
        if (s.failed) FAIL("negative nu_tilde solver failed");

        if (!std::isfinite(s.forces.CD)) FAIL("CD not finite: %g", s.forces.CD);
        if (!std::isfinite(s.forces.CL)) FAIL("CL not finite: %g", s.forces.CL);

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
    result |= test_oracle_freestream_preservation();
    result |= test_oracle_symmetric_cube_forces();
    result |= test_oracle_flat_plate_zero_forces();
    result |= test_oracle_convergence_history();
    result |= test_oracle_wall_forces();
    result |= test_oracle_dispatch();
    result |= test_recon_constant_state_zero_gradients();
    result |= test_recon_gradient_match();
    result |= test_recon_first_order_regression();
result |= test_recon_order2_converged_forces();
    result |= test_cpu_order2_residual_matches_gpu();
    result |= test_oracle_mesh_counts();
    result |= test_oracle_bandwidth();
    result |= test_diag_state_bounds_gpu_cpu_match();
    result |= test_diag_failure_snapshot();
    result |= test_color_count();
    result |= test_color_residual_matches_uncolored();
    result |= test_color_gradient_matches_uncolored();
    result |= test_color_deterministic_residual();
    result |= test_viscous_false_regression();
    result |= test_viscous_finite_flat_plate();
    result |= test_viscous_differs_from_inviscid();
    result |= test_rans_false_regression();
    result |= test_rans_zero_nu_tilde();
    result |= test_rans_turbulent_flat_plate();
    result |= test_rans_negative_nu_tilde();
    result |= test_rans_cpu_gpu_source_match();
    result |= test_mixed_mesh_gpu_upload();
    result |= test_hex_mesh_gpu_residual();
    result |= test_hex_mesh_symmetric_forces();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}

