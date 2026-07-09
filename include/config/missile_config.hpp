#pragma once
#include "sim/propulsion/propulsion_model.hpp"
#include "aero/aerodynamics_model.hpp"
#include "sim/control/autopilot.hpp"
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
        Eigen::Matrix3d inertia_tensor;
        Eigen::Vector3d center_of_mass;
        
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
