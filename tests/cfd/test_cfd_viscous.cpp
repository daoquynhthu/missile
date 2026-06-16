#include "aero_cfd/viscous.hpp"

#include <cmath>
#include <cstdio>

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

int main() {
    int result = 0;
    result |= test_sutherland();
    result |= test_wall_states();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
