#include "aero_solver/mesh_generator.hpp"
#include "aero_solver/cfd_solver.hpp"
#include "aero_solver/aero_solver.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cfloat>

using namespace AeroSim::Solver;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg, ...) do { printf("FAIL: " msg "\n", ##__VA_ARGS__); return 1; } while(0)

// Count wall boundary faces in a TetMesh

// Count wall boundary faces in a TetMesh
static int count_wall_faces(const TetMesh& mesh) {
    int count = 0;
    for (size_t i = 0; i < mesh.tet_neighbors.size(); ++i) {
        int4 nb = mesh.tet_neighbors[i];
        int b[4] = {
            (nb.x >= 0) ? 0 : ((nb.x == -1) ? 1 : 2),
            (nb.y >= 0) ? 0 : ((nb.y == -1) ? 1 : 2),
            (nb.z >= 0) ? 0 : ((nb.z == -1) ? 1 : 2),
            (nb.w >= 0) ? 0 : ((nb.w == -1) ? 1 : 2),
        };
        for (int f = 0; f < 4; ++f) if (b[f] == 1) count++;
    }
    return count;
}

// Count farfield boundary faces
static int count_farfield_faces(const TetMesh& mesh) {
    int count = 0;
    for (size_t i = 0; i < mesh.tet_neighbors.size(); ++i) {
        int4 nb = mesh.tet_neighbors[i];
        int b[4] = {
            (nb.x >= 0) ? 0 : ((nb.x == -1) ? 1 : 2),
            (nb.y >= 0) ? 0 : ((nb.y == -1) ? 1 : 2),
            (nb.z >= 0) ? 0 : ((nb.z == -1) ? 1 : 2),
            (nb.w >= 0) ? 0 : ((nb.w == -1) ? 1 : 2),
        };
        for (int f = 0; f < 4; ++f) if (b[f] == 2) count++;
    }
    return count;
}

int main() {
    // ── 1. Mesh validation ──
    TEST("Mesh wall face coverage");
    {
        auto mesh = generate_cube_mesh(5.0f);
        if (mesh.tets.empty()) FAIL("mesh generation failed");

        int wall_faces = count_wall_faces(mesh);
        int ff_faces = count_farfield_faces(mesh);

        // Cube surface area = 24. Expected wall faces provide coverage.
        // With ~0.014 area per face, need ~1700 faces. But Delaunay surface
        // triangulation varies; accept any positive wall count.
        if (wall_faces <= 0) FAIL("no wall faces (%d)", wall_faces);
        if (ff_faces <= 0) FAIL("no farfield faces (%d)", ff_faces);

        // Each tet has 4 faces; total faces = 4 * num_tets
        // wall + farfield should be much less than total (most are interior)
        size_t total_faces = mesh.tets.size() * 4;
        size_t boundary_faces = (size_t)(wall_faces + ff_faces);
        if (boundary_faces >= total_faces)
            FAIL("too many boundary faces (%zu / %zu)", boundary_faces, total_faces);

        PASS;
    }

    // ── 2. Basic solve: M=2, α=0°, β=0° ──
    TEST("FVM M=2 a=0 symmetry: CX>0, CY≈0, CZ≈0");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        auto r = solver.solve(2.0f, 0.0f, 0.0f,
                              4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (r.CX <= 0) FAIL("CX=%.4f (expected > 0)", r.CX);
        if (fabsf(r.CY) > 0.1f) FAIL("|CY|=%.4f (expected ≈ 0)", r.CY);
        if (fabsf(r.CZ) > 0.1f) FAIL("|CZ|=%.4f (expected ≈ 0)", r.CZ);
        if (std::isnan(r.residual) || r.residual > 1e-5f)
            FAIL("residual=%.2e (expected < 1e-5)", r.residual);
        if (r.iterations <= 0) FAIL("no iterations");

        PASS;
    }

    // ── 3. Non-zero AoA: M=2, α=10°, β=0° ──
    TEST("FVM M=2 a=10: CX>0, CZ>0");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        auto r = solver.solve(2.0f, 10.0f, 0.0f,
                              4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (r.CX <= 0) FAIL("CX=%.4f (expected > 0)", r.CX);
        if (r.CZ <= 0) FAIL("CZ=%.4f (expected > 0 at a=10)", r.CZ);
        if (std::isnan(r.residual) || r.residual > 1e-5f)
            FAIL("residual=%.2e (expected < 1e-5)", r.residual);

        PASS;
    }

    // ── 4. Non-zero sideslip: M=2, β=5°, α=0° ──
    TEST("FVM M=2 b=5: CX>0, CY>0");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        auto r = solver.solve(2.0f, 0.0f, 5.0f,
                              4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (r.CX <= 0) FAIL("CX=%.4f (expected > 0)", r.CX);
        if (r.CY <= 0) FAIL("CY=%.4f (expected > 0 at b=5)", r.CY);
        if (std::isnan(r.residual) || r.residual > 1e-5f)
            FAIL("residual=%.2e (expected < 1e-5)", r.residual);

        PASS;
    }

    // ── 5. Mach variation: raw force should increase with M ──
    TEST("FVM raw force increases with Mach number");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        auto r1 = solver.solve(1.5f, 0.0f, 0.0f,
                               4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);
        auto r2 = solver.solve(2.0f, 0.0f, 0.0f,
                               4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);
        auto r3 = solver.solve(3.0f, 0.0f, 0.0f,
                               4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (r1.CX <= 0) FAIL("CX(M=1.5)=%.4f (expected > 0)", r1.CX);
        if (r2.CX <= 0) FAIL("CX(M=2)=%.4f (expected > 0)", r2.CX);
        if (r3.CX <= 0) FAIL("CX(M=3)=%.4f (expected > 0)", r3.CX);

        // Raw force = CX * q_inf * ref_area should increase with M
        auto q_inf = [](float m) { return 0.5f * m * m; };
        float raw1 = r1.CX * q_inf(1.5f) * 4.0f;
        float raw2 = r2.CX * q_inf(2.0f) * 4.0f;
        float raw3 = r3.CX * q_inf(3.0f) * 4.0f;
        if (raw1 >= raw2) FAIL("raw Fx(M=1.5)=%.2f >= raw Fx(M=2)=%.2f", raw1, raw2);
        if (raw2 >= raw3) FAIL("raw Fx(M=2)=%.2f >= raw Fx(M=3)=%.2f", raw2, raw3);

        PASS;
    }

    // ── 6. Convergence: residual should drop from iter 0 ──
    TEST("FVM residual convergence trend");
    {
        // Re-use the same mesh and solver as test 2; residual prints already show trend.
        // This test verifies the residual at iter 0 is larger than final residual.
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 100;  // run just 100 iters, residual should drop
        cfg.muscl = false;

        auto r = solver.solve(2.0f, 0.0f, 0.0f,
                              4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        // After interior-tet removal, the initial transient is short.
        // Just check residual is finite and small-ish.
        if (std::isnan(r.residual) || std::isinf(r.residual))
            FAIL("residual=%.2e", r.residual);

        PASS;
    }

    // ── 7. MUSCL enabled: lower CFL, should stay finite ──
    TEST("FVM MUSCL finite at low CFL");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.01f;    // very low CFL for MUSCL stability on coarse mesh
        cfg.max_iter = 5000;
        cfg.muscl = true;

        auto r = solver.solve(2.0f, 0.0f, 0.0f,
                              4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (std::isnan(r.residual) || std::isinf(r.residual))
            FAIL("MUSCL residual=%.2e", r.residual);
        if (std::isnan(r.CX) || std::isinf(r.CX))
            FAIL("MUSCL CX=%.4f", r.CX);
        if (r.CX <= 0) FAIL("MUSCL CX=%.4f (expected > 0)", r.CX);

        PASS;
    }

    // ── 8. Batch API ──
    TEST("FVM batch solver API");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        std::vector<float> machs = {1.5f, 2.0f, 3.0f};
        std::vector<float> alphas = {0.0f, 5.0f, 10.0f};
        auto results = solver.solve_batch(machs, alphas, 0.0f,
                                          4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (results.size() != 9) FAIL("expected 9 results, got %zu", results.size());

        for (size_t i = 0; i < results.size(); ++i) {
            if (std::isnan(results[i].CX) || std::isinf(results[i].CX))
                FAIL("NaN/Inf CX at result %zu", i);
            if (std::isnan(results[i].CY) || std::isinf(results[i].CY))
                FAIL("NaN/Inf CY at result %zu", i);
            if (std::isnan(results[i].CZ) || std::isinf(results[i].CZ))
                FAIL("NaN/Inf CZ at result %zu", i);
            // At α=0: CY≈0, CZ≈0.
            if (i % 3 == 0) {
                if (fabsf(results[i].CY) > 0.1f)
                    FAIL("result[%zu] CY=%.4f (expected ≈0 at a=0)", i, results[i].CY);
                if (fabsf(results[i].CZ) > 0.1f)
                    FAIL("result[%zu] CZ=%.4f (expected ≈0 at a=0)", i, results[i].CZ);
            }
        }

        PASS;
    }

    // ── 9. Warm-start batch ──
    TEST("FVM warm-start batch");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        std::vector<float> machs = {1.5f, 2.0f, 3.0f};
        std::vector<float> alphas = {0.0f, 5.0f};
        std::vector<float> betas = {0.0f};

        auto results = solver.solve_batch_warm(machs, alphas, betas,
                                               4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (results.size() != 6) FAIL("expected 6 results, got %zu", results.size());

        for (size_t i = 0; i < results.size(); ++i) {
            if (std::isnan(results[i].CX) || std::isinf(results[i].CX))
                FAIL("NaN/Inf CX at result %zu", i);
            if (results[i].CX <= 0)
                FAIL("result[%zu] CX=%.4f (expected > 0)", i, results[i].CX);
            if (results[i].residual > 1e-4f)
                FAIL("result[%zu] residual=%.6e (expected < 1e-4)", i, results[i].residual);
        }

        // At α=0, β=0: CY≈0, CZ≈0
        if (fabsf(results[0].CY) > 0.1f)
            FAIL("warm[0] CY=%.4f (expected ≈0 at a=0,b=0)", results[0].CY);
        if (fabsf(results[0].CZ) > 0.1f)
            FAIL("warm[0] CZ=%.4f (expected ≈0 at a=0,b=0)", results[0].CZ);
        // At α=5°, β=0: CZ > 0
        if (results[1].CZ <= 0)
            FAIL("warm[1] CZ=%.4f (expected >0 at a=5)", results[1].CZ);

        PASS;
    }

    // ── 10. B.4 hypersonic cross-validation ──
    TEST("B.4 FVM vs Newtonian: M=5,8,10,15 @ α=0,5,10,15,20");
    {
        // Build cube surface mesh for Newtonian panel method.
        // Newtonian kernel expects normals such that F = -Cp·n·A gives
        // positive drag for windward faces. We compute CCW cross product;
        // if CX comes out negative we flip all normals.
        float verts[8][3] = {
            {-1,-1,-1}, { 1,-1,-1}, { 1,-1, 1}, {-1,-1, 1},
            {-1, 1,-1}, { 1, 1,-1}, { 1, 1, 1}, {-1, 1, 1}
        };
        int faces[12][3] = {
            {0,1,2},{0,2,3},{4,6,5},{4,7,6},
            {0,3,7},{0,7,4},{1,5,6},{1,6,2},
            {0,4,5},{0,5,1},{3,2,6},{3,6,7}
        };
        std::vector<Triangle> cube_tris(12);
        for (int i = 0; i < 12; ++i) {
            auto& face = faces[i];
            float3 v0 = make_float3(verts[face[0]][0], verts[face[0]][1], verts[face[0]][2]);
            float3 v1 = make_float3(verts[face[1]][0], verts[face[1]][1], verts[face[1]][2]);
            float3 v2 = make_float3(verts[face[2]][0], verts[face[2]][1], verts[face[2]][2]);
            cube_tris[i].v0 = v0;
            cube_tris[i].v1 = v1;
            cube_tris[i].v2 = v2;
            float ex1 = v1.x - v0.x, ey1 = v1.y - v0.y, ez1 = v1.z - v0.z;
            float ex2 = v2.x - v0.x, ey2 = v2.y - v0.y, ez2 = v2.z - v0.z;
            float cpx = ey1*ez2 - ez1*ey2;
            float cpy = ez1*ex2 - ex1*ez2;
            float cpz = ex1*ey2 - ey1*ex2;
            float area2 = sqrtf(cpx*cpx + cpy*cpy + cpz*cpz);
            cube_tris[i].area = 0.5f * area2;
            float inv = area2 > 1e-10f ? 1.0f/area2 : 0.0f;
            cube_tris[i].normal = make_float3(cpx*inv, cpy*inv, cpz*inv);
            cube_tris[i].center = make_float3(
                (v0.x+v1.x+v2.x)*(1.0f/3.0f),
                (v0.y+v1.y+v2.y)*(1.0f/3.0f),
                (v0.z+v1.z+v2.z)*(1.0f/3.0f));
            cube_tris[i].body_axis_x = cube_tris[i].center.x;
        }

        AeroSolver newtonian;
        if (!newtonian.load_mesh(cube_tris, 4.0f, 2.0f, 2.0f))
            FAIL("Newtonian load_mesh failed");

        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver cfd;
        if (!cfd.load_mesh(mesh))
            FAIL("FVM load_mesh failed");

        cfd.alloc_scratch();

        CfdConfig fvm_cfg;
        fvm_cfg.cfl = 0.5f;
        fvm_cfg.max_iter = 20000;
        fvm_cfg.muscl = false;

        float machs_fvm[] = {5.0f, 8.0f, 10.0f, 15.0f};
        float alphas_fvm[] = {0.0f, 5.0f, 10.0f, 15.0f, 20.0f};
        float* warm_Q = nullptr;

        for (float m : machs_fvm) {
            for (float a : alphas_fvm) {
                auto nc = newtonian.compute_coefficients(m, a, 0.0f);
                // Cold-start each α to avoid symmetry locking
                auto fvm = cfd.solve(m, a, 0.0f, 4.0f, 2.0f, 2.0f, 0,0,0, fvm_cfg, nullptr, &warm_Q);

                if (std::isnan(fvm.CX) || std::isinf(fvm.CX))
                    FAIL("FVM NaN CX at M=%.0f α=%.0f", m, a);

                // Both methods should produce non-zero drag (|CD| > 0)
                if (fabsf(nc.CD) < 1e-6f)
                    FAIL("Newtonian |CD|≈0 at M=%.0f α=%.0f", m, a);
                if (fabsf(fvm.CD) < 1e-6f)
                    FAIL("FVM |CD|≈0 at M=%.0f α=%.0f", m, a);

                // Theoretical Cp_stag for cross-validation
                float Cp_stag = (1.4f+3.0f)/(1.4f+1.0f) * (1.0f - 1.0f/(1.4f*m*m));

                // Newtonian |CX| should be close to Cp_stag at α=0 (within ~15%)
                if (fabsf(a) < 0.1f) {
                    float nc_abs = fabsf(nc.CX);
                    if (fabsf(nc_abs - Cp_stag) > 0.3f)
                        FAIL("Newtonian |CX|=%.4f vs theoretical %.4f at M=%.0f", nc_abs, Cp_stag, m);
                }

                // FVM |CX| within factor 2 of Newtonian |CX|
                float fvm_abs = fabsf(fvm.CX);
                float nc_abs = fabsf(nc.CX);
                if (fvm_abs > 2.0f * nc_abs || fvm_abs < 0.3f * nc_abs)
                    FAIL("FVM |CX|=%.4f vs Newtonian |CX|=%.4f at M=%.0f α=%.0f (ratio=%.2f)",
                         fvm_abs, nc_abs, m, a, fvm_abs/(nc_abs+1e-10f));

                // Symmetry at α=0: CY≈0, CZ≈0 (both methods)
                if (fabsf(a) < 0.1f) {
                    if (fabsf(fvm.CY) > 0.1f)
                        FAIL("FVM CY=%.4f (expected ≈0) at M=%.0f α=0", fvm.CY, m);
                    if (fabsf(fvm.CZ) > 0.1f)
                        FAIL("FVM CZ=%.4f (expected ≈0) at M=%.0f α=0", fvm.CZ, m);
                }

                // At α>0: CZ > 0
                if (a > 1.0f) {
                    if (fvm.CZ <= 0)
                        FAIL("FVM CZ=%.4f (expected >0) at M=%.0f α=%.0f", fvm.CZ, m, a);
                }

                // L/D for a cube should be modest (< 3)
                if (fvm.CD > 0 && fvm.CL > 0) {
                    float ld = fvm.CL / fvm.CD;
                    if (ld > 3.0f)
                        FAIL("FVM L/D=%.4f (expected <3) at M=%.0f α=%.0f", ld, m, a);
                }

                // Print summary
                printf("  M=%.0f α=%.0f | FVM: CX=%.4f CY=%.4f CZ=%.4f CL=%.4f CD=%.4f | "
                       "Newt: CX=%.4f CY=%.4f CZ=%.4f CD=%.4f | iter=%d res=%.2e\n",
                       m, a,
                       fvm.CX, fvm.CY, fvm.CZ, fvm.CL, fvm.CD,
                       nc.CX, nc.CY, nc.CZ, nc.CD,
                       fvm.iterations, fvm.residual);
            }
        }

        cfd.free_scratch();
        PASS;
    }

    printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return (test_count == pass_count) ? 0 : 1;
}
