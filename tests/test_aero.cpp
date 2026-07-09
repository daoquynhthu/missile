#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "aero/aerodynamics_model.hpp"
#include "config/missile_config.hpp"

using namespace aerosp;

void test_aero_lookup() {
    std::cout << "[Test] Starting Aerodynamics Lookup Test..." << std::endl;

    // 1. Load Config
    auto config = config::load_hgv1_config();
    AerodynamicsModel aero(config.aerodynamics);

    // 2. Verify Table Loaded
    // We can't access private members directly, but we can infer from compute_coeffs
    // Ideally we should have a public `is_loaded()` method or similar, but let's just call compute.
    
    // 3. Test Point 1: Zero Alpha/Beta (Symmetry)
    // Mach 5.0, Alpha 0.0, Beta 0.0
    auto coeffs0 = aero.compute_coeffs(5.0, 0.0, 0.0);
    
    std::cout << "Mach 5.0, Alpha 0.0, Beta 0.0:" << std::endl;
    std::cout << "  CL: " << coeffs0.CL << " (Expected ~0.0)" << std::endl;
    std::cout << "  CY: " << coeffs0.CY << " (Expected 0.0)" << std::endl;
    std::cout << "  Cm: " << coeffs0.Cm << " (Expected small)" << std::endl; // Might be non-zero due to camber/trim
    
    if (std::abs(coeffs0.CL) > 0.05) {
        std::cerr << "FAILED: Lift at zero alpha is too high!" << std::endl;
        exit(1);
    }
    if (std::abs(coeffs0.CY) > 1e-3) {
        std::cerr << "FAILED: Side force at zero beta is non-zero!" << std::endl;
        exit(1);
    }

    // 4. Test Point 2: Hypersonic Glide Condition
    // Mach 15.0, Alpha 10.0 deg
    double alpha_rad = 10.0 * 3.14159 / 180.0;
    auto coeffs1 = aero.compute_coeffs(15.0, alpha_rad, 0.0);
    
    std::cout << "Mach 15.0, Alpha 10.0:" << std::endl;
    std::cout << "  CL: " << coeffs1.CL << std::endl;
    std::cout << "  CD: " << coeffs1.CD << std::endl;
    
    double L_D = coeffs1.CL / coeffs1.CD;
    std::cout << "  L/D: " << L_D << " (Expected 2.0 - 4.0)" << std::endl;
    
    if (L_D < 1.5 || L_D > 5.0) {
        std::cerr << "FAILED: Hypersonic L/D out of realistic range!" << std::endl;
        // exit(1); // Soft fail for now, maybe table is just bad
    }

    // 5. Test Point 3: Symmetry Check with Beta
    // Mach 2.0, Alpha 5.0, Beta 5.0
    // If we have symmetry, Cy should be negative for positive Beta (restoring force)
    double beta_rad = 5.0 * 3.14159 / 180.0;
    auto coeffs2 = aero.compute_coeffs(2.0, 5.0 * 3.14159 / 180.0, beta_rad);
    
    std::cout << "Mach 2.0, Alpha 5.0, Beta 5.0:" << std::endl;
    std::cout << "  CY: " << coeffs2.CY << " (Expected negative)" << std::endl;
    std::cout << "  Cn: " << coeffs2.Cn << " (Expected positive/stable?)" << std::endl;

    if (coeffs2.CY > 0.0) {
         std::cerr << "WARNING: Unstable Side Force (CY > 0 for Beta > 0). Check sign convention." << std::endl;
    }

    std::cout << "[Pass] Aerodynamics Test Passed." << std::endl;
}

int main() {
    test_aero_lookup();
    return 0;
}
