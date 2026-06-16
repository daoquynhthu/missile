#include "aero_cfd/cfd_mesh.hpp"

#include <cmath>
#include <cstdio>

using namespace AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_cube_mesh() {
    TEST("CFD-MESH-1 structured cube has valid metrics");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 13);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.nodes != 13*13*13) FAIL("nodes=%d", report.nodes);
        if (report.cells <= 0) FAIL("cells=%d", report.cells);
        if (report.slip_wall_faces <= 0) FAIL("slip_wall_faces=%d", report.slip_wall_faces);
        if (report.farfield_faces <= 0) FAIL("farfield_faces=%d", report.farfield_faces);
        if (report.min_volume <= 0.0f) FAIL("min_volume=%g", report.min_volume);
        PASS;
    }

    TEST("CFD-MESH-2 cube wall classification survives off-surface vertices");
    {
        auto mesh13 = generate_structured_cube_mesh(5.0f, 13);
        auto report13 = compute_mesh_metrics(mesh13);
        auto mesh17 = generate_structured_cube_mesh(5.0f, 17);
        auto report17 = compute_mesh_metrics(mesh17);
        if (report13.slip_wall_faces <= 0) FAIL("n=13 wall faces=%d", report13.slip_wall_faces);
        if (report17.slip_wall_faces <= 0) FAIL("n=17 wall faces=%d", report17.slip_wall_faces);
        if (report13.no_slip_wall_faces != 0) FAIL("unexpected no-slip wall faces=%d", report13.no_slip_wall_faces);
        PASS;
    }

    TEST("CFD-MESH-3 cube wall normal area is closed");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 13);
        float sx = 0.0f;
        float sy = 0.0f;
        float sz = 0.0f;
        for (const auto& face : mesh.faces) {
            if (face.boundary != BoundaryKind::SlipWall) continue;
            sx += face.nx * face.area;
            sy += face.ny * face.area;
            sz += face.nz * face.area;
        }
        if (std::fabs(sx) > 1e-5f) FAIL("sx=%g", sx);
        if (std::fabs(sy) > 1e-5f) FAIL("sy=%g", sy);
        if (std::fabs(sz) > 1e-5f) FAIL("sz=%g", sz);
        PASS;
    }

    return 0;
}

static int test_flat_plate_mesh() {
    TEST("CFD-MESH-4 flat plate wall area matches geometry");
    {
        float length = 0.5f;
        float width = 0.05f;
        auto mesh = generate_flat_plate_mesh(length, width, 0.1f, 1e-5f, 1.12f, 30, 3, 50);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.no_slip_wall_faces <= 0) FAIL("wall faces=%d", report.no_slip_wall_faces);
        float area = boundary_area(mesh, BoundaryKind::NoSlipWall);
        float expected = length * width;
        float rel = std::fabs(area - expected) / expected;
        if (rel > 1e-5f) FAIL("wall area=%g expected=%g rel=%g", area, expected, rel);
        if (report.min_wall_distance <= 0.0f) FAIL("min wall distance=%g", report.min_wall_distance);
        if (std::fabs(report.min_wall_distance - 2.5e-6f) > 5e-7f) {
            FAIL("min wall distance=%g", report.min_wall_distance);
        }
        PASS;
    }

    TEST("CFD-MESH-5 flat plate farfield exists and mesh is positive");
    {
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 10, 3, 12);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.farfield_faces <= 0) FAIL("farfield_faces=%d", report.farfield_faces);
        if (report.min_h <= 0.0f || report.max_h <= report.min_h) {
            FAIL("h range=[%g,%g]", report.min_h, report.max_h);
        }
        PASS;
    }

    return 0;
}

int main() {
    int result = 0;
    result |= test_cube_mesh();
    result |= test_flat_plate_mesh();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
