#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

// Include necessary headers from missile_lib
#include "aero_solver/aero_solver.hpp"
#include "utils.hpp"
#include "constants.hpp"

// We need a simple way to parse args
struct AeroArgs {
    std::string stl_path;
    double mach = 0.0;
    double alpha_deg = 0.0;
    double beta_deg = 0.0;
    double ref_area = 1.0;
    double ref_length = 1.0;
    double ref_span = 1.0;
};

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <stl_path> <mach> <alpha_deg> <beta_deg> [ref_area] [ref_length] [ref_span]" << std::endl;
    std::cerr << "Example: " << prog_name << " model.stl 15.0 5.0 0.0 15.0 10.0 3.0" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    AeroArgs args;
    args.stl_path = argv[1];
    args.mach = std::stod(argv[2]);
    args.alpha_deg = std::stod(argv[3]);
    args.beta_deg = std::stod(argv[4]);
    
    if (argc >= 6) args.ref_area = std::stod(argv[5]);
    if (argc >= 7) args.ref_length = std::stod(argv[6]);
    if (argc >= 8) args.ref_span = std::stod(argv[7]);

    // Load Solver
    AeroSim::Solver::AeroSolver solver;
    if (!solver.load_model(args.stl_path, args.ref_area, args.ref_length, args.ref_span)) {
        std::cerr << "[Error] Failed to load STL model: " << args.stl_path << std::endl;
        return 2;
    }

    // Compute Coefficients
    // solver.compute_coefficients takes (mach, alpha_deg, beta_deg)
    // It returns AeroCoefficients struct with computed CL, CD, L_D
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto coeffs = solver.compute_coefficients(args.mach, args.alpha_deg, args.beta_deg);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output JSON-like format for easy parsing by Python
    std::cout << "{"
              << "\"mach\": " << args.mach << ", "
              << "\"alpha\": " << args.alpha_deg << ", "
              << "\"beta\": " << args.beta_deg << ", "
              << "\"CX\": " << coeffs.CX << ", "
              << "\"CY\": " << coeffs.CY << ", "
              << "\"CZ\": " << coeffs.CZ << ", "
              << "\"CL\": " << coeffs.CL << ", "
              << "\"CD\": " << coeffs.CD << ", "
              << "\"L_D\": " << coeffs.L_D << ", "
              << "\"Cl\": " << coeffs.Cl << ", "
              << "\"Cm\": " << coeffs.Cm << ", "
              << "\"Cn\": " << coeffs.Cn << ", "
              << "\"computation_time_ms\": " << elapsed.count() * 1000.0
              << "}" << std::endl;

    return 0;
}
