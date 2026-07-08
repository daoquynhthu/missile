#include "aero_cfd/real.hpp"
#include "aero_cfd/cfd_solver.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace AeroSim;
using namespace AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_state_bounds() {
    TEST("CFD-DIAG-1 state bounds report extrema and validity");
    {
        std::vector<ConservativeState> q;
        PrimitiveState a;
        a.rho = 1.0f;
        a.u = 2.0f;
        a.v = 0.0f;
        a.w = 0.0f;
        a.p = 1.0f;
        PrimitiveState b;
        b.rho = 0.5f;
        b.u = 0.0f;
        b.v = 1.0f;
        b.w = 0.0f;
        b.p = 0.25f;
        q.push_back(primitive_to_conservative(a, 1.4f));
        q.push_back(primitive_to_conservative(b, 1.4f));

        auto bounds = compute_state_bounds(q, 1.4f);
        if (!bounds.valid) FAIL("bounds invalid");
        if (std::fabs(bounds.min_rho - 0.5f) > 1e-6f) FAIL("min_rho=%g", bounds.min_rho);
        if (std::fabs(bounds.max_rho - 1.0f) > 1e-6f) FAIL("max_rho=%g", bounds.max_rho);
        if (std::fabs(bounds.min_p - 0.25f) > 1e-6f) FAIL("min_p=%g", bounds.min_p);
        if (std::fabs(bounds.max_p - 1.0f) > 1e-6f) FAIL("max_p=%g", bounds.max_p);
        if (bounds.max_mach <= bounds.min_mach) FAIL("mach range=[%g,%g]", bounds.min_mach, bounds.max_mach);
        PASS;
    }
    return 0;
}

static int test_solver_diagnostics() {
    TEST("CFD-DIAG-2 basic diagnostics do not change first-order result");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        for (auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::NoSlipWall) face.boundary = BoundaryKind::Farfield;
        }
        compute_mesh_metrics(mesh);

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig off_cfg;
        off_cfg.max_iter = 3;
        off_cfg.cfl = 0.1f;
        off_cfg.convergence_tol = 1e-12f;
        auto off = solver.solve({2.0f, 0.0f, 0.0f}, off_cfg);

        CfdConfig diag_cfg = off_cfg;
        diag_cfg.diagnostic_level = DiagnosticLevel::Basic;
        auto diag = solver.solve({2.0f, 0.0f, 0.0f}, diag_cfg);

        if (off.failed || diag.failed) FAIL("solver failed");
        if (off.residual_history.size() != diag.residual_history.size()) FAIL("residual history size changed");
        if (std::fabs(off.forces.CX - diag.forces.CX) > 1e-8f) FAIL("CX changed");
        if (diag.diagnostics.state_bounds_history.empty()) FAIL("missing state bounds");
        if (diag.diagnostics.dt_limiter_history.empty()) FAIL("missing dt limiter");
        if (diag.diagnostics.state_bounds_history.front().min_rho <= 0.0f) FAIL("min_rho invalid");
        if (diag.diagnostics.dt_limiter_history.front().cell < 0) FAIL("dt limiter cell missing");
        PASS;
    }

    TEST("CFD-DIAG-3 invalid injected state produces failure snapshot");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        PrimitiveState w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        q[2].rho = -1.0f;

        CfdConfig cfg;
        cfg.max_iter = 1;
        cfg.diagnostic_level = DiagnosticLevel::Basic;
        auto summary = solver.solve_from_state({2.0f, 0.0f, 0.0f}, cfg, q);
        if (!summary.failed) FAIL("solver did not fail");
        if (!summary.diagnostics.failure.valid) FAIL("missing failure snapshot");
        if (summary.diagnostics.failure.cell != 2) FAIL("failure cell=%d", summary.diagnostics.failure.cell);
        if (std::strstr(summary.diagnostics.failure.reason.c_str(), "invalid initial state") == nullptr) {
            FAIL("reason=%s", summary.diagnostics.failure.reason.c_str());
        }
        if (summary.diagnostics.failure.state.rho != -1.0f) FAIL("snapshot rho=%g", summary.diagnostics.failure.state.rho);
        PASS;
    }

    return 0;
}

static int test_vtk_output() {
    TEST("CFD-DIAG-4 VTK cell output writes mesh and primitive fields");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));

        const char* path = "cfd_diagnostics_test.vtk";
        std::remove(path);
        std::string error;
        if (!write_vtk_cells(path, mesh, q, 1.4f, &error)) FAIL("%s", error.c_str());

        std::ifstream in(path);
        if (!in) FAIL("failed to read vtk");
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        std::remove(path);

        if (text.find("DATASET UNSTRUCTURED_GRID") == std::string::npos) FAIL("missing dataset");
        if (text.find("CELL_DATA") == std::string::npos) FAIL("missing cell data");
        if (text.find("SCALARS rho float 1") == std::string::npos) FAIL("missing rho");
        if (text.find("SCALARS pressure float 1") == std::string::npos) FAIL("missing pressure");
        if (text.find("SCALARS mach float 1") == std::string::npos) FAIL("missing mach");
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_state_bounds();
    result |= test_solver_diagnostics();
    result |= test_vtk_output();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}

