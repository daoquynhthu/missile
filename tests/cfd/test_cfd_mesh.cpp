#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/element_types.hpp"
#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/mesh_io_cgns.hpp"
#include "aero/cfd/mesh_validator.hpp"
#include "aero/cfd/real.hpp"

#include <cmath>
#include <cstdio>

using namespace aerosp;
using namespace aerosp::aero::cfd;

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
        Real sx = 0.0f;
        Real sy = 0.0f;
        Real sz = 0.0f;
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
        Real length = 0.5f;
        Real width = 0.05f;
        auto mesh = generate_flat_plate_mesh(length, width, 0.1f, 1e-5f, 1.12f, 30, 3, 50);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.no_slip_wall_faces <= 0) FAIL("wall faces=%d", report.no_slip_wall_faces);
        Real area = boundary_area(mesh, BoundaryKind::NoSlipWall);
        Real expected = length * width;
        Real rel = std::fabs(area - expected) / expected;
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

static int test_su2_round_trip() {
    TEST("CFD-MESH-IO-1 SU2 round-trip: hex mesh read/write/read matches");
    {
        CfdMesh original = generate_structured_hex_mesh(6);
        compute_mesh_metrics(original);

        const char* tmp_path = "test_mesh_su2_temp.su2";
        std::string err;
        if (!write_mesh_su2(original, tmp_path, &err)) FAIL("write: %s", err.c_str());

        CfdMesh reloaded;
        if (!read_mesh_su2(tmp_path, reloaded, &err)) FAIL("read: %s", err.c_str());
        compute_mesh_metrics(reloaded);

        if (reloaded.nodes.size() != original.nodes.size()) FAIL("node count: %zu vs %zu", reloaded.nodes.size(), original.nodes.size());
        if (reloaded.cells.size() != original.cells.size()) FAIL("cell count: %zu vs %zu", reloaded.cells.size(), original.cells.size());

        for (std::size_t i = 0; i < reloaded.cells.size(); ++i) {
            if (reloaded.cells[i].type != original.cells[i].type) FAIL("cell %zu type mismatch", i);
        }

        std::remove(tmp_path);
        PASS;
    }
    return 0;
}

static int test_su2_invalid_file() {
    TEST("CFD-MESH-IO-3 SU2 import with missing NELEM: read fails gracefully");
    {
        const char* tmp_path = "test_mesh_su2_bad.su2";
        std::FILE* f = std::fopen(tmp_path, "w");
        if (!f) FAIL("cannot create temp file");
        std::fprintf(f, "NDIME= 3\n");
        std::fprintf(f, "NPOIN= 0\n");
        std::fprintf(f, "NMARK= 0\n");
        std::fclose(f);

        CfdMesh mesh;
        std::string err;
        bool ok = read_mesh_su2(tmp_path, mesh, &err);
        if (ok) FAIL("expected read to fail, but it succeeded");

        std::remove(tmp_path);
        PASS;
    }
    return 0;
}

static int test_mesh_quality_detail() {
    TEST("CFD-MESH-IO-4 mesh quality detail on flat plate");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);
        MeshQualityReport r = compute_mesh_quality_detail(mesh);
        std::printf("\n  flat plate quality report:\n");
        std::printf("  cells=%d faces=%d\n", r.cells, r.faces);
        std::printf("  neg_jac=%d | aspect=[%.2f .. %.2f] avg=%.2f\n", r.negative_jacobian_count, r.min_aspect_ratio, r.max_aspect_ratio, r.avg_aspect_ratio);
        std::printf("  skew=[%.4f .. %.4f] avg=%.4f\n", r.min_skewness, r.max_skewness, r.avg_skewness);
        std::printf("  ortho=[%.1f .. %.1f] avg=%.1f deg\n", r.min_orthogonality, r.max_orthogonality, r.avg_orthogonality);
        std::printf("  closed_surf_err=%g\n", r.closed_surface_error);
        if (!r.valid) FAIL("detail: %s", r.message.c_str());
        if (r.negative_jacobian_count != 0) FAIL("neg jac=%d", r.negative_jacobian_count);
        if (r.min_volume <= 0.0f) FAIL("min vol=%g", r.min_volume);
        PASS;
    }

    TEST("CFD-MESH-IO-5 mesh quality on cube mesh (25^3)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 25);
        compute_mesh_metrics(mesh);
        MeshQualityReport r = compute_mesh_quality_detail(mesh);
        std::printf("\n  cube quality report:\n");
        std::printf("  cells=%d faces=%d\n", r.cells, r.faces);
        std::printf("  neg_jac=%d | aspect=[%.2f .. %.2f] avg=%.2f\n", r.negative_jacobian_count, r.min_aspect_ratio, r.max_aspect_ratio, r.avg_aspect_ratio);
        std::printf("  skew=[%.4f .. %.4f] avg=%.4f\n", r.min_skewness, r.max_skewness, r.avg_skewness);
        std::printf("  ortho=[%.1f .. %.1f] avg=%.1f deg\n", r.min_orthogonality, r.max_orthogonality, r.avg_orthogonality);
        std::printf("  vol=[%g .. %g]\n", r.min_volume, r.max_volume);
        std::printf("  closed_surf_err=%g\n", r.closed_surface_error);
        if (!r.valid) FAIL("detail: %s", r.message.c_str());
        if (r.negative_jacobian_count != 0) FAIL("neg jac=%d", r.negative_jacobian_count);
        if (r.min_volume <= 0.0f) FAIL("min vol=%g", r.min_volume);
        PASS;
    }

    TEST("CFD-MESH-IO-6 mesh quality on hex mesh");
    {
        CfdMesh mesh = generate_structured_hex_mesh(6);
        compute_mesh_metrics(mesh);
        MeshQualityReport r = compute_mesh_quality_detail(mesh);
        std::printf("\n  hex quality report:\n");
        std::printf("  cells=%d faces=%d\n", r.cells, r.faces);
        std::printf("  neg_jac=%d | aspect=[%.2f .. %.2f] avg=%.2f\n", r.negative_jacobian_count, r.min_aspect_ratio, r.max_aspect_ratio, r.avg_aspect_ratio);
        std::printf("  skew=[%.4f .. %.4f] avg=%.4f\n", r.min_skewness, r.max_skewness, r.avg_skewness);
        std::printf("  ortho=[%.1f .. %.1f] avg=%.1f deg\n", r.min_orthogonality, r.max_orthogonality, r.avg_orthogonality);
        std::printf("  closed_surf_err=%g\n", r.closed_surface_error);
        if (!r.valid) FAIL("detail: %s", r.message.c_str());
        if (r.negative_jacobian_count != 0) FAIL("neg jac=%d", r.negative_jacobian_count);
        if (r.min_volume <= 0.0f) FAIL("min vol=%g", r.min_volume);
        PASS;
    }

    return 0;
}

static int test_cgns_fallback() {
    TEST("CFD-MESH-IO-2 CGNS fallback when library unavailable");
    {
        CfdMesh mesh;
        std::string err;
        bool ok = read_mesh_cgns("nonexistent.cgns", mesh, &err);
        if (ok) FAIL("expected CGNS read to fail but it succeeded");
        if (err.empty()) FAIL("expected non-empty error message");
        std::printf("  CGNS fallback message: \"%s\"\n", err.c_str());
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_cube_mesh();
    result |= test_flat_plate_mesh();
    result |= test_su2_round_trip();
    result |= test_su2_invalid_file();
    result |= test_cgns_fallback();
    result |= test_mesh_quality_detail();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}


