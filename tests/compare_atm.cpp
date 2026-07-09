#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "sim/atmosphere/atmosphere_model.hpp"
#include "infra/math/constants.hpp"

using namespace aerosp;

NRLMSISE00Input get_input(double alt_km, double lat, double lon, int doy, double sec, double f107, double ap) {
    NRLMSISE00Input input;
    input.year = 2026; 
    input.doy = doy;   
    input.sec = sec;     
    input.alt = alt_km; 
    input.lat = lat;
    input.lon = lon;
    input.lst = sec / 3600.0 + lon / 15.0; 
    input.f107A = f107;
    input.f107 = f107;
    input.ap = ap;
    for(int i=0; i<7; ++i) input.ap_vector[i] = ap;
    return input;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Atmosphere Model Comparison: USSA76 vs NRLMSISE-00" << std::endl;
    std::cout << "=================================================================" << std::endl;
    
    // Altitudes to test (km)
    std::vector<double> altitudes = {0, 10, 50, 100, 200, 400, 600};
    
    // Conditions: High Solar Activity vs Low Solar Activity
    double f107_high = 200.0;
    double f107_low = 70.0;
    double ap_quiet = 4.0;
    double ap_storm = 400.0; // Extreme storm
    
    std::cout << std::scientific << std::setprecision(3);
    
    std::cout << "\n--- 1. Altitude Profile (Low Solar Activity F10.7=70, Ap=4) ---" << std::endl;
    std::cout << "Alt(km) | USSA76 Rho(kg/m3) | NRL00 Rho(kg/m3) | Diff(%)" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    for (double h : altitudes) {
        // USSA76
        AtmosphereData ussa = AtmosphereModel::calculate_ussa76(h * 1000.0);
        
        // NRLMSISE-00
        NRLMSISE00Input input = get_input(h, 45.0, 0.0, 172, 43200.0, f107_low, ap_quiet);
        AtmosphereData nrl = AtmosphereModel::calculate(input);
        
        double diff = (nrl.density - ussa.density) / ussa.density * 100.0;
        
        std::cout << std::fixed << std::setprecision(0) << std::setw(7) << h << " | "
                  << std::scientific << std::setprecision(3) << ussa.density << " | "
                  << nrl.density << " | "
                  << std::fixed << std::setprecision(1) << std::setw(6) << diff << "%" << std::endl;
    }

    std::cout << "\n--- 2. Solar Activity Effect at 400km (Day 172, Noon, Lat 45) ---" << std::endl;
    double h = 400.0;
    AtmosphereData ussa = AtmosphereModel::calculate_ussa76(h * 1000.0);
    
    NRLMSISE00Input low = get_input(h, 45.0, 0.0, 172, 43200.0, f107_low, ap_quiet);
    AtmosphereData nrl_low = AtmosphereModel::calculate(low);
    
    NRLMSISE00Input high = get_input(h, 45.0, 0.0, 172, 43200.0, f107_high, ap_quiet);
    AtmosphereData nrl_high = AtmosphereModel::calculate(high);
    
    NRLMSISE00Input storm = get_input(h, 45.0, 0.0, 172, 43200.0, f107_high, ap_storm);
    AtmosphereData nrl_storm = AtmosphereModel::calculate(storm);
    
    std::cout << "Model          | Density (kg/m3) | Ratio to USSA76" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "USSA76         | " << ussa.density << " | 1.00" << std::endl;
    std::cout << "NRL00 (Low Act)| " << nrl_low.density << " | " << nrl_low.density/ussa.density << std::endl;
    std::cout << "NRL00 (High Act)| " << nrl_high.density << " | " << nrl_high.density/ussa.density << std::endl;
    std::cout << "NRL00 (Storm)  | " << nrl_storm.density << " | " << nrl_storm.density/ussa.density << std::endl;
    
    return 0;
}
