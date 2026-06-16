#include "aero_cfd/viscous.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_sutherland() {
    TEST("CFD-VISC-1 primitive temperature is p/rho");
    {
        PrimitiveState w;
        w.rho = 2.0f;
        w.p = 5.0f;
        if (std::fabs(primitive_temperature(w) - 2.5f) > 1e-6f) FAIL("T=%g", primitive_temperature(w));
        PASS;
    }

    TEST("CFD-VISC-2 Sutherland viscosity is normalized at reference temperature");
    {
        float mu = sutherland_viscosity(1.0f);
        if (std::fabs(mu - 1.0f) > 1e-6f) FAIL("mu=%g", mu);
        PASS;
    }

    TEST("CFD-VISC-3 Sutherland viscosity increases with temperature");
    {
        float cold = sutherland_viscosity(0.5f);
        float hot = sutherland_viscosity(2.0f);
        if (!(cold > 0.0f && hot > cold)) FAIL("cold=%g hot=%g", cold, hot);
        PASS;
    }
    return 0;
}

static int test_wall_states() {
    TEST("CFD-VISC-4 isothermal no-slip wall preserves pressure and wall temperature");
    {
        PrimitiveState interior;
        interior.rho = 1.0f;
        interior.u = 2.0f;
        interior.v = -0.5f;
        interior.w = 0.25f;
        interior.p = 0.8f;

        auto wall = no_slip_isothermal_wall_state(interior, 0.4f);
        if (std::fabs(wall.u) > 1e-8f || std::fabs(wall.v) > 1e-8f || std::fabs(wall.w) > 1e-8f) {
            FAIL("velocity=[%g,%g,%g]", wall.u, wall.v, wall.w);
        }
        if (std::fabs(wall.p - interior.p) > 1e-6f) FAIL("p=%g", wall.p);
        if (std::fabs(primitive_temperature(wall) - 0.4f) > 1e-6f) FAIL("T=%g", primitive_temperature(wall));
        PASS;
    }

    TEST("CFD-VISC-5 adiabatic no-slip wall keeps interior temperature");
    {
        PrimitiveState interior;
        interior.rho = 1.25f;
        interior.u = -3.0f;
        interior.v = 0.0f;
        interior.w = 0.0f;
        interior.p = 0.5f;

        auto wall = no_slip_adiabatic_wall_state(interior);
        if (std::fabs(wall.u) > 1e-8f || std::fabs(wall.v) > 1e-8f || std::fabs(wall.w) > 1e-8f) {
            FAIL("velocity=[%g,%g,%g]", wall.u, wall.v, wall.w);
        }
        if (std::fabs(primitive_temperature(wall) - primitive_temperature(interior)) > 1e-6f) {
            FAIL("Twall=%g Tin=%g", primitive_temperature(wall), primitive_temperature(interior));
        }
        PASS;
    }
    return 0;
}

static int test_viscous_gradients() {
    TEST("CFD-VISC-6 viscous gradient maps velocity and temperature derivatives");
    {
        PrimitiveState w;
        w.rho = 2.0f;
        w.u = 1.0f;
        w.v = -2.0f;
        w.w = 0.5f;
        w.p = 6.0f;

        PrimitiveGradient primitive_gradient;
        primitive_gradient.du_dx = 1.0f;
        primitive_gradient.dv_dy = 2.0f;
        primitive_gradient.dw_dz = 3.0f;
        primitive_gradient.drho_dx = 0.5f;
        primitive_gradient.dp_dx = 1.0f;
        primitive_gradient.dp_dy = 2.0f;
        primitive_gradient.dp_dz = 3.0f;

        auto gradient = viscous_gradient_from_primitive_gradient(w, primitive_gradient);
        if (std::fabs(gradient.du_dx - 1.0f) > 1e-6f) FAIL("du_dx=%g", gradient.du_dx);
        if (std::fabs(gradient.dv_dy - 2.0f) > 1e-6f) FAIL("dv_dy=%g", gradient.dv_dy);
        if (std::fabs(gradient.dw_dz - 3.0f) > 1e-6f) FAIL("dw_dz=%g", gradient.dw_dz);
        if (std::fabs(gradient.dT_dx + 0.25f) > 1e-6f) FAIL("dT_dx=%g", gradient.dT_dx);
        if (std::fabs(gradient.dT_dy - 1.0f) > 1e-6f) FAIL("dT_dy=%g", gradient.dT_dy);
        if (std::fabs(gradient.dT_dz - 1.5f) > 1e-6f) FAIL("dT_dz=%g", gradient.dT_dz);
        PASS;
    }

    TEST("CFD-VISC-7 constant state has zero viscous gradients");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        auto gradients = compute_viscous_gradients(mesh, q, 1.4f);
        if (gradients.size() != mesh.cells.size()) FAIL("gradient size=%zu", gradients.size());
        for (const auto& gradient : gradients) {
            if (std::fabs(gradient.du_dx) > 1e-6f || std::fabs(gradient.dv_dy) > 1e-6f ||
                std::fabs(gradient.dw_dz) > 1e-6f) {
                FAIL("velocity gradient=[%g,%g,%g]", gradient.du_dx, gradient.dv_dy, gradient.dw_dz);
            }
            if (std::fabs(gradient.dT_dx) > 1e-6f || std::fabs(gradient.dT_dy) > 1e-6f ||
                std::fabs(gradient.dT_dz) > 1e-6f) {
                FAIL("temperature gradient=[%g,%g,%g]", gradient.dT_dx, gradient.dT_dy, gradient.dT_dz);
            }
        }
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_sutherland();
    result |= test_wall_states();
    result |= test_viscous_gradients();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
