#pragma once
#include "propulsion_model.hpp"
#include "aerodynamics_model.hpp"
#include "gnc/autopilot.hpp"
#include <vector>
#include <string>

namespace AeroSim {
namespace MissileDesign {

    /**
     * @brief Missile Configuration Data Structure
     */
    struct HGV1Config {
        std::string name = "HGV-1 Dragonfire";
        
        // Mass Properties
        double total_mass = 15000.0;
        double payload_mass = 1000.0;
        
        // Propulsion
        AeroSim::SolidMotor::Config propulsion;
        
        // Aerodynamics
        AeroSim::AerodynamicsModel::Config aerodynamics;
        
        // Autopilot
        AeroSim::GNC::Autopilot::Config autopilot;

        // RCS
        // RCSModel::Config rcs; // If we want to move RCS here too
    };

    /**
     * @brief Load HGV-1 Configuration with high fidelity data
     */
    HGV1Config load_hgv1_config();

}
}
