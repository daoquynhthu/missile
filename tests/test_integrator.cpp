#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include "dynamics_6dof.hpp"

using namespace AeroSim;

// Simple Harmonic Oscillator System
// x'' = -omega^2 * x
// Let omega = 1.0
// x' = v
// v' = -x
struct HarmonicOscillator {
    State6DOF operator()(const State6DOF& state, double t) {
        State6DOF deriv;
        deriv.pos_ecef = state.vel_ecef; // dr/dt = v
        deriv.vel_ecef = -1.0 * state.pos_ecef; // dv/dt = -x
        
        // Zero derivatives for others
        deriv.quat_be.coeffs().setZero();
        deriv.omega_body.setZero();
        deriv.mass = 0.0;
        
        return deriv;
    }
};

bool test_harmonic_oscillator() {
    std::cout << "[Test] Harmonic Oscillator (Adaptive Integrator)..." << std::endl;
    
    State6DOF state;
    state.pos_ecef = Eigen::Vector3d(1.0, 0.0, 0.0); // Initial x = 1
    state.vel_ecef = Eigen::Vector3d(0.0, 0.0, 0.0); // Initial v = 0
    state.quat_be.setIdentity();
    state.omega_body.setZero();
    state.mass = 1.0;
    
    HarmonicOscillator system;
    double t = 0.0;
    double dt = 0.1; // Initial guess
    double t_end = 10.0; // Approx 1.5 periods (2*pi ~= 6.28)
    
    double tol = 1e-6;
    
    int steps = 0;
    while (t < t_end) {
        // Clamp dt to not exceed t_end
        if (t + dt > t_end) {
            dt = t_end - t;
        }
        
        state = Dynamics6DOF::integrate_adaptive(state, system, t, dt, tol);
        steps++;
    }
    
    // Analytical Solution at t_end
    // x(t) = cos(t)
    // v(t) = -sin(t)
    double expected_x = std::cos(t_end);
    double expected_v = -std::sin(t_end);
    
    double error_x = std::abs(state.pos_ecef.x() - expected_x);
    double error_v = std::abs(state.vel_ecef.x() - expected_v);
    
    std::cout << "Steps taken: " << steps << std::endl;
    std::cout << "Final Time: " << t << std::endl;
    std::cout << "Final X: " << state.pos_ecef.x() << " (Expected: " << expected_x << ")" << std::endl;
    std::cout << "Final V: " << state.vel_ecef.x() << " (Expected: " << expected_v << ")" << std::endl;
    std::cout << "Error X: " << error_x << std::endl;
    std::cout << "Error V: " << error_v << std::endl;
    
    bool passed = (error_x < 1e-5) && (error_v < 1e-5);
    if (passed) std::cout << "[PASS] Harmonic Oscillator Accuracy" << std::endl;
    else std::cout << "[FAIL] Harmonic Oscillator Accuracy" << std::endl;
    
    return passed;
}

bool test_step_adaptation() {
    std::cout << "\n[Test] Step Size Adaptation..." << std::endl;
    
    State6DOF state;
    state.pos_ecef = Eigen::Vector3d(1.0, 0.0, 0.0);
    state.vel_ecef = Eigen::Vector3d(0.0, 0.0, 0.0);
    state.quat_be.setIdentity();
    state.omega_body.setZero();
    state.mass = 1.0;
    
    HarmonicOscillator system;
    
    // High Tolerance -> Large Steps
    double t1 = 0.0;
    double dt1 = 0.1;
    double tol1 = 1e-3;
    Dynamics6DOF::integrate_adaptive(state, system, t1, dt1, tol1);
    double next_dt_loose = dt1;
    
    // Low Tolerance -> Small Steps
    double t2 = 0.0;
    double dt2 = 0.1;
    double tol2 = 1e-9;
    Dynamics6DOF::integrate_adaptive(state, system, t2, dt2, tol2);
    double next_dt_tight = dt2;
    
    std::cout << "Next DT (Tol 1e-3): " << next_dt_loose << std::endl;
    std::cout << "Next DT (Tol 1e-9): " << next_dt_tight << std::endl;
    
    bool passed = next_dt_tight < next_dt_loose;
    if (passed) std::cout << "[PASS] Step size reduced for tighter tolerance" << std::endl;
    else std::cout << "[FAIL] Step size did not adapt correctly" << std::endl;
    
    return passed;
}

int main() {
    bool p1 = test_harmonic_oscillator();
    bool p2 = test_step_adaptation();
    
    if (p1 && p2) return 0;
    return 1;
}
