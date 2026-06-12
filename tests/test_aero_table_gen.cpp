#include "aero_solver/aero_solver.hpp"
#include <iostream>
#include <vector>

int main() {
    AeroSim::Solver::AeroTableConfig cfg;
    cfg.ref_area = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span = 3.0f;
    cfg.com_x = 6.0f;

    std::vector<double> mach_grid  = {0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0};
    std::vector<double> alpha_grid = {-10.0, -5.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0,
                                      12.0, 15.0, 20.0, 25.0, 30.0, 40.0};
    std::vector<double> beta_grid  = {0.0};

    bool ok = AeroSim::Solver::generate_aero_table(
        "data/missile/hgv_model_optimized.stl",
        "data/missile/aerodynamics_table.csv",
        mach_grid, alpha_grid, beta_grid, cfg
    );

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
