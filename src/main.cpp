#include <iostream>
#include <iomanip>
#include "constants.hpp"

int main() {
    std::cout << "===============================================" << std::endl;
    std::cout << "  AeroHighPrecisionSim (AHPS) - Core Initialized" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "WGS-84 Semi-major axis (A): " << AeroSim::Earth::A << " m" << std::endl;
    std::cout << "Earth MU: " << AeroSim::Earth::MU << " m^3/s^2" << std::endl;
    std::cout << "J2 Coefficient: " << AeroSim::Earth::J2 << std::endl;
    
    std::cout << "\nStarting Core Engineering Development..." << std::endl;
    
    return 0;
}
