#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include "common.hpp"

namespace AeroSim {

/**
 * @brief 6-DOF state vector
 */
struct State6DOF {
    Eigen::Vector3d pos_ecef;      // Position in ECEF (m)
    Eigen::Vector3d vel_ecef;      // Velocity in ECEF (m/s)
    Eigen::Quaterniond quat_be;    // Quaternion from ECEF to Body
    Eigen::Vector3d omega_body;    // Angular velocity in Body frame (rad/s)
    double mass;                   // Mass (kg)

    // Operator overloading for RK4 integration
    CUDA_HOST_DEVICE State6DOF operator+(const State6DOF& other) const {
        State6DOF result;
        result.pos_ecef = pos_ecef + other.pos_ecef;
        result.vel_ecef = vel_ecef + other.vel_ecef;
        // Quaternion addition is component-wise for integration purposes
        result.quat_be.coeffs() = quat_be.coeffs() + other.quat_be.coeffs();
        result.omega_body = omega_body + other.omega_body;
        result.mass = mass + other.mass;
        return result;
    }

    CUDA_HOST_DEVICE State6DOF operator*(double scalar) const {
        State6DOF result;
        result.pos_ecef = pos_ecef * scalar;
        result.vel_ecef = vel_ecef * scalar;
        result.quat_be.coeffs() = quat_be.coeffs() * scalar;
        result.omega_body = omega_body * scalar;
        result.mass = mass * scalar;
        return result;
    }
    
    CUDA_HOST_DEVICE void normalize() {
        quat_be.normalize();
    }
};

/**
 * @brief Forces and moments in Body frame
 */
struct ForcesMoments {
    Eigen::Vector3d force_body;    // Force in Body frame (N)
    Eigen::Vector3d moment_body;   // Moment in Body frame (N*m)
    double mass_flow_rate;         // Mass flow rate (kg/s)
};

/**
 * @brief Inertial properties
 */
struct InertialProps {
    double mass;                   // Mass (kg)
    Eigen::Matrix3d inertia;       // Moment of inertia in Body frame (kg*m^2)
    Eigen::Vector3d com;           // Center of mass offset (m)
};

class Dynamics6DOF {
public:
    /**
     * @brief Compute the state derivative (f = x_dot)
     * @param state Current state
     * @param fm Forces and moments in Body frame
     * @param inertia Inertial properties
     * @param gravity_ecef Gravity acceleration in ECEF frame (m/s^2)
     */
    static CUDA_HOST_DEVICE State6DOF compute_derivatives(
        const State6DOF& state,
        const ForcesMoments& fm,
        const InertialProps& inertia,
        const Eigen::Vector3d& gravity_ecef
    );

    /**
     * @brief Integrate state using 4th order Runge-Kutta with dynamic force evaluation
     * @tparam SystemDynamics Functor type that computes derivative: State6DOF operator()(const State6DOF&, double t)
     * @param state Initial state
     * @param system Functor to compute derivatives (must include gravity, aero, etc.)
     * @param t0 Current time
     * @param dt Time step
     */
    template <typename SystemDynamics>
    static CUDA_HOST_DEVICE State6DOF integrate_rk4(
        const State6DOF& state,
        SystemDynamics& system,
        double t0,
        double dt
    ) {
        // k1 = f(t, y)
        State6DOF k1 = system(state, t0);
        
        // k2 = f(t + dt/2, y + k1 * dt/2)
        State6DOF s2 = state + k1 * (0.5 * dt);
        s2.normalize(); // Normalize quaternion after addition
        State6DOF k2 = system(s2, t0 + 0.5 * dt);
        
        // k3 = f(t + dt/2, y + k2 * dt/2)
        State6DOF s3 = state + k2 * (0.5 * dt);
        s3.normalize();
        State6DOF k3 = system(s3, t0 + 0.5 * dt);
        
        // k4 = f(t + dt, y + k3 * dt)
        State6DOF s4 = state + k3 * dt;
        s4.normalize();
        State6DOF k4 = system(s4, t0 + dt);
        
        // y_new = y + (k1 + 2k2 + 2k3 + k4) * dt / 6
        State6DOF next_state = state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        next_state.normalize();
        
        return next_state;
    }
};

} // namespace AeroSim
