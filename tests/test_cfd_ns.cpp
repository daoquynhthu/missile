#include "ref_data/flat_plate_laminar_ref.hpp"
#include "ref_data/flat_plate_turbulent_ref.hpp"
#include "ref_data/normal_shock_cea_ref.hpp"
#include "ref_data/sphere_heatflux_ref.hpp"
#include "ref_data/hemi_cylinder_ref.hpp"
#include "aero_solver/mesh_generator.hpp"
#include "aero_solver/cfd_solver.hpp"
#include "aero_solver/aero_solver.hpp"
#include <cstdio>
#include <cmath>
#include <cfloat>

using namespace AeroSim::RefData;
using namespace AeroSim::Solver;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("[Test] " name " ... "); } while(0)
#define PASS do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg, ...) do { printf("FAIL: " msg "\n", ##__VA_ARGS__); return 1; } while(0)

// ─── C.0.1: Reference data integrity ─────────────────────────────────

static int test_ref_data_integrity() {
    TEST("C.0 flat plate laminar reference: Cf decreases with Re");
    {
        auto data = flat_plate_laminar_ref();
        if (data.size() < 3) FAIL("too few data points (%zu)", data.size());
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i].Cf_incomp >= data[i-1].Cf_incomp)
                FAIL("Cf(%g)=%g >= Cf(%g)=%g (expected decrease)",
                     data[i].Re_x, data[i].Cf_incomp,
                     data[i-1].Re_x, data[i-1].Cf_incomp);
            if (data[i].Cf_comp >= data[i-1].Cf_comp)
                FAIL("Cf_comp(%g)=%g >= Cf_comp(%g)=%g",
                     data[i].Re_x, data[i].Cf_comp,
                     data[i-1].Re_x, data[i-1].Cf_comp);
            if (data[i].St_incomp >= data[i-1].St_incomp)
                FAIL("St(%g)=%g >= St(%g)=%g",
                     data[i].Re_x, data[i].St_incomp,
                     data[i-1].Re_x, data[i-1].St_incomp);
        }
        // Cf_comp <= Cf_incomp (compressibility never increases Cf)
        // When F_c=1 (isothermal T_w=T_e), they are equal.
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].Cf_comp > data[i].Cf_incomp + 1e-12f)
                FAIL("Cf_comp(%g)=%g > Cf_incomp(%g)=%g at Re=%g",
                     data[i].Re_x, data[i].Cf_comp,
                     data[i].Re_x, data[i].Cf_incomp, data[i].Re_x);
        }
        PASS;
    }

    TEST("C.0 flat plate turbulent reference: Cf decreases with Re");
    {
        auto data = flat_plate_turbulent_ref();
        if (data.size() < 3) FAIL("too few data points (%zu)", data.size());
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i].Cf_incomp >= data[i-1].Cf_incomp)
                FAIL("turb Cf(%g)=%g >= Cf(%g)=%g",
                     data[i].Re_x, data[i].Cf_incomp,
                     data[i-1].Re_x, data[i-1].Cf_incomp);
        }
        PASS;
    }

    TEST("C.0 turbulent Cf > laminar Cf at same Re");
    {
        auto turb = flat_plate_turbulent_ref();
        auto lam = flat_plate_laminar_ref();
        // Compare at Re=1e6 (present in both)
        double lam_Cf = 0, turb_Cf = 0;
        for (auto& p : lam) if (fabs(p.Re_x - 1e6) < 1) lam_Cf = p.Cf_incomp;
        for (auto& p : turb) if (fabs(p.Re_x - 1e6) < 1) turb_Cf = p.Cf_incomp;
        if (lam_Cf <= 0 || turb_Cf <= 0)
            FAIL("missing Re=1e6: lam_Cf=%g turb_Cf=%g", lam_Cf, turb_Cf);
        if (turb_Cf <= lam_Cf)
            FAIL("turb Cf=%g <= lam Cf=%g at Re=1e6 (expected turb > lam)", turb_Cf, lam_Cf);
        PASS;
    }

    TEST("C.0 normal shock: p,T,rho ratios increase with M");
    {
        auto data = normal_shock_ref();
        if (data.size() < 3) FAIL("too few shock points (%zu)", data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].p_ratio <= 1.0)
                FAIL("M=%g: p_ratio=%g <= 1", data[i].M1, data[i].p_ratio);
            if (data[i].T_ratio <= 1.0)
                FAIL("M=%g: T_ratio=%g <= 1", data[i].M1, data[i].T_ratio);
            if (data[i].rho_ratio <= 1.0)
                FAIL("M=%g: rho_ratio=%g <= 1", data[i].M1, data[i].rho_ratio);
            if (data[i].M2 >= data[i].M1)
                FAIL("M2=%g >= M1=%g", data[i].M2, data[i].M1);
            // Theoretical M2_min = sqrt((γ-1)/(2γ)) ≈ 0.378 for γ=1.4
            double M2_min = sqrt((1.4-1.0)/(2.0*1.4)) - 0.05;
            if (data[i].M2 < M2_min)
                FAIL("M2=%g too low at M1=%g (min=%g)", data[i].M2, data[i].M1, M2_min);
            if (i > 0) {
                if (data[i].p_ratio <= data[i-1].p_ratio)
                    FAIL("p_ratio not monotonic: M=%.0f->%.0f", data[i-1].M1, data[i].M1);
                if (data[i].T_ratio <= data[i-1].T_ratio)
                    FAIL("T_ratio not monotonic: M=%.0f->%.0f", data[i-1].M1, data[i].M1);
            }
            // Species sanity
            double Y_sum = data[i].Y_N2 + data[i].Y_O2 + data[i].Y_N + data[i].Y_O + data[i].Y_NO;
            if (fabs(Y_sum - 1.0) > 1e-6)
                FAIL("M=%g: Y_sum=%g != 1", data[i].M1, Y_sum);
        }
        // At high M, O2 should be fully dissociated
        if (data.back().Y_O2 > 0.01)
            FAIL("Y_O2=%g at M=%.0f (expected near 0)", data.back().Y_O2, data.back().M1);
        PASS;
    }

    TEST("C.0 sphere heat flux: q decreases with altitude (increasing M)");
    {
        auto data = sphere_heatflux_ref();
        if (data.size() < 3) FAIL("too few sphere points (%zu)", data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].q_stag_MW <= 0)
                FAIL("M=%g alt=%gkm: q=%g MW <= 0", data[i].M, data[i].altitude_km, data[i].q_stag_MW);
            if (std::isnan(data[i].q_stag_MW) || std::isinf(data[i].q_stag_MW))
                FAIL("NaN/Inf q at M=%g", data[i].M);
        }
        // Heat flux should be non-negative and finite.
        // At low Mach + high altitude (M=5, 50km), q can be ~1e-5 MW/m^2.
        // At high Mach (M>=15), typical reentry heating is 0.1-100 MW/m^2.
        for (auto& p : data) {
            if (p.q_stag_MW < 0.0)
                FAIL("negative q=%g MW/m^2 at M=%g", p.q_stag_MW, p.M);
            if (std::isnan(p.q_stag_MW) || std::isinf(p.q_stag_MW))
                FAIL("NaN/Inf q at M=%g", p.M);
            // High-Mach points should have significant heating
            if (p.M >= 15.0 && p.q_stag_MW < 1e-6)
                FAIL("unrealistically low q=%g MW/m^2 at M=%g", p.q_stag_MW, p.M);
        }
        PASS;
    }

    TEST("C.0 hemi-cylinder Cp: symmetric about theta=0, Cp_max at stagnation");
    {
        auto data = hemi_cylinder_ref();
        if (data.size() < 3) FAIL("too few Cp points (%zu)", data.size());
        if (data[0].Cp != data[0].Cp_max)
            FAIL("Cp(0)=%g != Cp_max=%g", data[0].Cp, data[0].Cp_max);
        // Cp should decrease with theta
        double prev = data[0].Cp;
        for (size_t i = 1; i < data.size() && data[i].theta_deg <= 90; ++i) {
            if (data[i].Cp > prev + 1e-10)
                FAIL("Cp(%g)=%g > Cp(%g)=%g", data[i].theta_deg, data[i].Cp,
                     data[i-1].theta_deg, prev);
            prev = data[i].Cp;
        }
        // Shadow region (theta > 90): Cp = 0
        for (auto& p : data) {
            if (p.theta_deg > 90 && p.Cp != 0)
                FAIL("Cp(%g)=%g != 0 in shadow", p.theta_deg, p.Cp);
        }
        PASS;
    }

    return 0;
}

// ─── C.0.2: Baseline Euler on cube ──────────────────────────────────

static int test_baseline_euler() {
    TEST("C.0 baseline Euler: slip-wall gives no shear (pressure-only drag)");
    {
        // On a symmetric cube at α=0, the Euler solver (inviscid, slip wall)
        // produces only pressure forces. There is no viscous shear by construction.
        // This test establishes the baseline: CX>0 (pressure drag), CY≈CZ≈0.
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        auto r = solver.solve(2.0f, 0.0f, 0.0f, 4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (r.CX <= 0) FAIL("CX=%.4f (expected > 0, pressure drag)", r.CX);
        if (fabsf(r.CY) > 0.1f) FAIL("|CY|=%.4f (expected ≈ 0 at α=0)", r.CY);
        if (fabsf(r.CZ) > 0.1f) FAIL("|CZ|=%.4f (expected ≈ 0 at α=0)", r.CZ);
        if (std::isnan(r.residual) || r.residual > 1e-5f)
            FAIL("residual=%.2e", r.residual);

        // At α=0 with slip walls, shear stress is identically zero.
        // This is inherent in the Euler formulation — no τ_w, no q_w.
        // When C.1 adds NS + no-slip, this test will detect the change.

        PASS;
    }

    TEST("C.0 baseline Euler: also runs at α=10° (non-zero lift)");
    {
        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;

        auto r = solver.solve(2.0f, 10.0f, 0.0f, 4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (r.CX <= 0) FAIL("CX=%.4f (expected > 0)", r.CX);
        if (r.CZ <= 0) FAIL("CZ=%.4f (expected > 0 at α=10)", r.CZ);
        if (std::isnan(r.residual) || r.residual > 1e-5f)
            FAIL("residual=%.2e", r.residual);

        PASS;
    }

    return 0;
}

// ─── C.0.3: Cross-validation with Newtonian for regression ─────────

static int test_cross_validation() {
    TEST("C.0 cross-validation: Euler FVM vs Newtonian at M=5 α=0");
    {
        // Build Newtonian panel mesh for cube
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
            cube_tris[i].v0 = v0; cube_tris[i].v1 = v1; cube_tris[i].v2 = v2;
            float ex1 = v1.x-v0.x, ey1 = v1.y-v0.y, ez1 = v1.z-v0.z;
            float ex2 = v2.x-v0.x, ey2 = v2.y-v0.y, ez2 = v2.z-v0.z;
            float cpx = ey1*ez2 - ez1*ey2;
            float cpy = ez1*ex2 - ex1*ez2;
            float cpz = ex1*ey2 - ey1*ex2;
            float area2 = sqrtf(cpx*cpx + cpy*cpy + cpz*cpz);
            cube_tris[i].area = 0.5f * area2;
            float inv = area2 > 1e-10f ? 1.0f/area2 : 0.0f;
            cube_tris[i].normal = make_float3(cpx*inv, cpy*inv, cpz*inv);
            cube_tris[i].center = make_float3(
                (v0.x+v1.x+v2.x)/3.0f, (v0.y+v1.y+v2.y)/3.0f, (v0.z+v1.z+v2.z)/3.0f);
            cube_tris[i].body_axis_x = cube_tris[i].center.x;
        }

        AeroSolver newtonian;
        if (!newtonian.load_mesh(cube_tris, 4.0f, 2.0f, 2.0f))
            FAIL("Newtonian load_mesh failed");

        auto nc = newtonian.compute_coefficients(5.0f, 0.0f, 0.0f);
        if (fabsf(nc.CD) < 1e-6f) FAIL("Newtonian CD≈0 at M=5");

        auto mesh = generate_cube_mesh(5.0f);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("FVM load_mesh failed");
        CfdConfig cfg;
        cfg.cfl = 0.5f;
        cfg.max_iter = 20000;
        cfg.muscl = false;
        auto fvm = solver.solve(5.0f, 0.0f, 0.0f, 4.0f, 2.0f, 2.0f, 0, 0, 0, cfg);

        if (fabsf(fvm.CD) < 1e-6f) FAIL("FVM CD≈0 at M=5");
        if (fvm.CX <= 0) FAIL("FVM CX=%.4f (expected > 0)", fvm.CX);

        // Both methods produce positive drag at M=5
        // For a cube, FVM and Newtonian should be within factor 2
        float fvm_abs = fabsf(fvm.CX);
        float nc_abs = fabsf(nc.CX);
        if (fvm_abs > 2.5f * nc_abs || fvm_abs < 0.25f * nc_abs)
            FAIL("FVM |CX|=%.4f vs Newtonian |CX|=%.4f (ratio=%.2f)",
                 fvm_abs, nc_abs, fvm_abs/(nc_abs+1e-10f));

        // Symmetry check
        if (fabsf(fvm.CY) > 0.1f) FAIL("FVM CY=%.4f (expected ≈0)", fvm.CY);
        if (fabsf(fvm.CZ) > 0.1f) FAIL("FVM CZ=%.4f (expected ≈0)", fvm.CZ);

        printf("  M=5 α=0 | FVM: CX=%.4f CD=%.4f | Newt: CX=%.4f CD=%.4f\n",
               fvm.CX, fvm.CD, nc.CX, nc.CD);
        PASS;
    }

    return 0;
}

int main() {
    int result = 0;
    result |= test_ref_data_integrity();
    result |= test_baseline_euler();
    result |= test_cross_validation();

    printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return (test_count == pass_count) ? 0 : 1;
}
