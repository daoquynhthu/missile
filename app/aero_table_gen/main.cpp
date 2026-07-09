#include "aero/panel/aero_solver.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: AeroTableGen <stl_path> <csv_output> <mach_1,...> <alpha_1,...> <beta_1,...>\n";
        return 1;
    }

    std::string stl_path = argv[1];
    std::string csv_path = argv[2];

    auto parse_csv = [](const char* s) {
        std::vector<double> v;
        std::string cur;
        while (*s) {
            if (*s == ',') { v.push_back(std::stod(cur)); cur.clear(); }
            else cur += *s;
            ++s;
        }
        if (!cur.empty()) v.push_back(std::stod(cur));
        return v;
    };

    std::vector<double> mach_grid  = (argc > 3) ? parse_csv(argv[3]) : std::vector<double>{5.0, 10.0, 15.0};
    std::vector<double> alpha_grid = (argc > 4) ? parse_csv(argv[4]) : std::vector<double>{0.0, 5.0, 10.0};
    std::vector<double> beta_grid  = (argc > 5) ? parse_csv(argv[5]) : std::vector<double>{0.0};

    AeroSim::Solver::AeroTableConfig cfg;
    cfg.ref_area = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span = 3.0f;

    bool ok = AeroSim::Solver::generate_aero_table(
        stl_path, csv_path, mach_grid, alpha_grid, beta_grid, cfg);

    return ok ? 0 : 1;
}
