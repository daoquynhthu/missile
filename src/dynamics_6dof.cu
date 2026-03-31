#include "dynamics_6dof.hpp"
#include "constants.hpp"
#include <cmath>

namespace AeroSim {

__host__ __device__ State6DOF Dynamics6DOF::compute_derivatives(
    const State6DOF& state,
    const ForcesMoments& fm,
    const InertialProps& inertia,
    const Eigen::Vector3d& gravity_ecef,
    SimulationProfile profile
) {
    State6DOF dot;

    // 1. Position derivative (in ECEF)
    dot.pos_ecef = state.vel_ecef;

    // 2. Velocity derivative (in ECEF)
    Eigen::Vector3d coriolis(0, 0, 0);
    Eigen::Vector3d centrifugal(0, 0, 0);

    // Coriolis and centrifugal forces only for global ballistic
    if (profile == SimulationProfile::GLOBAL_BALLISTIC) {
        Eigen::Vector3d omega_e(0, 0, Earth::OMEGA());
        coriolis = -2.0 * omega_e.cross(state.vel_ecef);
        centrifugal = -omega_e.cross(omega_e.cross(state.pos_ecef));
    }
    
    // Rotate force from Body to ECEF
    Eigen::Vector3d force_ecef = state.quat_be * fm.force_body;
    
    dot.vel_ecef = (force_ecef / state.mass) + gravity_ecef + coriolis + centrifugal;

    // 3. Quaternion derivative (ECEF to Body)
    // q_dot = 0.5 * q_be * [0, omega_body]
    // Note: quaternion multiplication order depends on convention.
    // Standard kinematic equation: q_dot = 0.5 * q * omega (if omega is in body frame)
    // Eigen convention: q_new = q_old * delta_q
    Eigen::Quaterniond q_omega(0, state.omega_body.x(), state.omega_body.y(), state.omega_body.z());
    Eigen::Quaterniond q_dot;
    q_dot.coeffs() = 0.5 * (state.quat_be * q_omega).coeffs();
    dot.quat_be = q_dot;

    // 4. Angular velocity derivative (in Body frame)
    // I * omega_dot = moment_body - omega_body x (I * omega_body)
    Eigen::Vector3d h_body = inertia.inertia * state.omega_body;
    Eigen::Vector3d moment_net = fm.moment_body - state.omega_body.cross(h_body);
    
    // Simplified: assuming inertia is constant and diagonal for now or use inverse
    // In a more robust version, we would use a solver or pre-computed inverse.
    dot.omega_body = inertia.inertia.inverse() * moment_net;

    // 5. Mass derivative (placeholder for variable mass)
    dot.mass = fm.mass_flow_rate; // Mass decreases (rate is negative)

    // Debug Print (Temporary)
    // static int debug_cnt = 0;
    // debug_cnt++;
    
    // if (state.mass > 5000.0 && debug_cnt % 100 == 0) { 
    //    printf("DEBUG [Early]: F_body=[%.1f, %.1f, %.1f] M_body=[%.1f, %.1f, %.1f] Mass=%.1f\n", 
    //        fm.force_body.x(), fm.force_body.y(), fm.force_body.z(), 
    //        fm.moment_body.x(), fm.moment_body.y(), fm.moment_body.z(),
    //        state.mass);
    //    printf("      Acc_ECEF=[%.1f, %.1f, %.1f] Omega=[%.3f, %.3f, %.3f]\n",
    //        dot.vel_ecef.x(), dot.vel_ecef.y(), dot.vel_ecef.z(),
    //        state.omega_body.x(), state.omega_body.y(), state.omega_body.z());
    // }
    
    // if (state.mass > 1000.0 && state.mass < 3000.0 && debug_cnt % 1000 == 0) { 
    //    printf("DEBUG [Late]: F_body=[%.1f, %.1f, %.1f] Mass=%.1f Acc_ECEF=[%.1f, %.1f, %.1f]\n", 
    //        fm.force_body.x(), fm.force_body.y(), fm.force_body.z(), 
    //        state.mass, 
    //        dot.vel_ecef.x(), dot.vel_ecef.y(), dot.vel_ecef.z());
    // }

    return dot;
}

} // namespace AeroSim
