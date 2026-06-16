#include "aero_cfd/reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_green_gauss() {
    TEST("CFD-RECON-1 Green-Gauss gradient is zero for constant primitive state");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        auto gradients = compute_green_gauss_gradients(mesh, q, 1.4f);
        if (gradients.size() != mesh.cells.size()) FAIL("gradient size=%zu cells=%zu", gradients.size(), mesh.cells.size());
        for (const auto& g : gradients) {
            if (std::fabs(g.drho_dx) > 1e-5f || std::fabs(g.drho_dy) > 1e-5f || std::fabs(g.drho_dz) > 1e-5f) {
                FAIL("rho gradient=[%g,%g,%g]", g.drho_dx, g.drho_dy, g.drho_dz);
            }
            if (std::fabs(g.dp_dx) > 1e-5f || std::fabs(g.dp_dy) > 1e-5f || std::fabs(g.dp_dz) > 1e-5f) {
                FAIL("p gradient=[%g,%g,%g]", g.dp_dx, g.dp_dy, g.dp_dz);
            }
        }
        PASS;
    }

    TEST("CFD-RECON-2 invalid state makes Green-Gauss fail closed");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        q[0].rho = -1.0f;
        auto gradients = compute_green_gauss_gradients(mesh, q, 1.4f);
        if (!gradients.empty()) FAIL("expected empty gradients");
        PASS;
    }
    return 0;
}

static int test_least_squares() {
    TEST("CFD-RECON-3 least-squares recovers linear pressure gradient");
    {
        CfdMesh mesh;
        mesh.cells.resize(4);
        mesh.cells[0].cx = 0.0f;
        mesh.cells[0].cy = 0.0f;
        mesh.cells[0].cz = 0.0f;
        mesh.cells[1].cx = 1.0f;
        mesh.cells[1].cy = 0.0f;
        mesh.cells[1].cz = 0.0f;
        mesh.cells[2].cx = 0.0f;
        mesh.cells[2].cy = 1.0f;
        mesh.cells[2].cz = 0.0f;
        mesh.cells[3].cx = 0.0f;
        mesh.cells[3].cy = 0.0f;
        mesh.cells[3].cz = 1.0f;

        for (int i = 1; i < 4; ++i) {
            CfdFace face;
            face.left_cell = 0;
            face.right_cell = i;
            face.boundary = BoundaryKind::Interior;
            mesh.faces.push_back(face);
        }

        std::vector<ConservativeState> q;
        for (const auto& cell : mesh.cells) {
            PrimitiveState w;
            w.rho = 1.0f;
            w.u = 0.0f;
            w.v = 0.0f;
            w.w = 0.0f;
            w.p = 1.0f + 0.2f*cell.cx - 0.1f*cell.cy + 0.3f*cell.cz;
            q.push_back(primitive_to_conservative(w, 1.4f));
        }

        auto gradients = compute_least_squares_gradients(mesh, q, 1.4f);
        if (gradients.size() != mesh.cells.size()) FAIL("gradient size=%zu", gradients.size());
        if (std::fabs(gradients[0].dp_dx - 0.2f) > 1e-6f) FAIL("dp_dx=%g", gradients[0].dp_dx);
        if (std::fabs(gradients[0].dp_dy + 0.1f) > 1e-6f) FAIL("dp_dy=%g", gradients[0].dp_dy);
        if (std::fabs(gradients[0].dp_dz - 0.3f) > 1e-6f) FAIL("dp_dz=%g", gradients[0].dp_dz);
        PASS;
    }
    return 0;
}

static int test_positive_guard() {
    TEST("CFD-RECON-4 positive guard limits density and pressure");
    {
        PrimitiveState center;
        center.rho = 1.0f;
        center.u = 0.0f;
        center.v = 0.0f;
        center.w = 0.0f;
        center.p = 1.0f;

        PrimitiveGradient g;
        g.drho_dx = -10.0f;
        g.dp_dx = -20.0f;
        float theta = 1.0f;
        auto out = reconstruct_primitive_positive(center, g, 0.1f, 0.0f, 0.0f, 0.2f, 0.2f, &theta);
        if (theta >= 1.0f || theta <= 0.0f) FAIL("theta=%g", theta);
        if (out.rho < 0.2f - 1e-6f) FAIL("rho=%g", out.rho);
        if (out.p < 0.2f - 1e-6f) FAIL("p=%g", out.p);
        PASS;
    }

    TEST("CFD-RECON-5 positive guard preserves safe reconstruction");
    {
        PrimitiveState center;
        center.rho = 1.0f;
        center.u = 0.0f;
        center.v = 0.0f;
        center.w = 0.0f;
        center.p = 1.0f;

        PrimitiveGradient g;
        g.drho_dx = 0.5f;
        g.dp_dx = -0.5f;
        float theta = 0.0f;
        auto guarded = reconstruct_primitive_positive(center, g, 0.1f, 0.0f, 0.0f, 0.2f, 0.2f, &theta);
        auto raw = reconstruct_primitive(center, g, 0.1f, 0.0f, 0.0f);
        if (std::fabs(theta - 1.0f) > 1e-6f) FAIL("theta=%g", theta);
        if (std::fabs(guarded.rho - raw.rho) > 1e-6f) FAIL("rho=%g raw=%g", guarded.rho, raw.rho);
        if (std::fabs(guarded.p - raw.p) > 1e-6f) FAIL("p=%g raw=%g", guarded.p, raw.p);
        PASS;
    }
    return 0;
}

static int test_limiter() {
    TEST("CFD-RECON-6 limiter is inactive for zero gradients");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        std::vector<PrimitiveGradient> gradients(mesh.cells.size());
        auto limiters = compute_barth_jespersen_limiters(mesh, q, gradients, 1.4f);
        if (limiters.size() != mesh.cells.size()) FAIL("limiter size=%zu", limiters.size());
        for (const auto& limiter : limiters) {
            if (std::fabs(limiter.rho - 1.0f) > 1e-6f) FAIL("rho limiter=%g", limiter.rho);
            if (std::fabs(limiter.p - 1.0f) > 1e-6f) FAIL("p limiter=%g", limiter.p);
        }
        PASS;
    }

    TEST("CFD-RECON-7 limiter suppresses new pressure extrema");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        std::vector<PrimitiveGradient> gradients(mesh.cells.size());
        for (auto& gradient : gradients) {
            gradient.dp_dx = 100.0f;
            gradient.dp_dy = 100.0f;
            gradient.dp_dz = 100.0f;
        }
        auto limiters = compute_barth_jespersen_limiters(mesh, q, gradients, 1.4f);
        if (limiters.size() != mesh.cells.size()) FAIL("limiter size=%zu", limiters.size());

        float min_p_limiter = 1.0f;
        for (const auto& limiter : limiters) min_p_limiter = std::min(min_p_limiter, limiter.p);
        if (min_p_limiter >= 1.0f) FAIL("min pressure limiter=%g", min_p_limiter);

        auto limited = apply_limiter(gradients[0], limiters[0]);
        if (std::fabs(limited.dp_dx) > std::fabs(gradients[0].dp_dx) + 1e-6f) FAIL("limited dp_dx=%g", limited.dp_dx);
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_green_gauss();
    result |= test_least_squares();
    result |= test_positive_guard();
    result |= test_limiter();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
