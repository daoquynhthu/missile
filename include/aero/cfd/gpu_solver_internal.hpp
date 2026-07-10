#pragma once

#include "aero/cfd/real.hpp"
#include "aero/cfd/device_mesh.hpp"

#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt);
bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    bool viscous, Real* d_mu, Real Re);

bool compute_gradients_gpu(DeviceMesh& mesh, Real gamma, std::string* error = nullptr,
    int* d_failed = nullptr);

bool compute_limiters_gpu(DeviceMesh& mesh, Real gamma, std::string* error = nullptr,
    int* d_failed = nullptr);

bool compute_update_gpu(DeviceMesh& mesh, const Real* d_min_dt, Real gamma,
    Real* d_l2_sum, int* d_failed,
    int* d_failure_cell = nullptr, Real* d_failure_state = nullptr);

bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces);
bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces,
    bool viscous, Real prandtl, Real mu_ref, Real T_ref, Real sutherland_T,
    Real Re, Real wall_T);

bool compute_state_bounds_gpu(DeviceMesh& mesh, Real gamma, Real* d_bounds_slot);

bool compute_failure_snapshot_gpu(DeviceMesh& mesh, Real gamma,
    int* d_failure_cell, Real* d_failure_state);

bool compute_viscous_flux_gpu(DeviceMesh& mesh, Real gamma, Real prandtl,
    Real mu_ref, Real T_ref, Real sutherland_T, Real Re, Real wall_T,
    int turbulence, int* d_failed);

bool compute_rans_source_gpu(DeviceMesh& mesh, Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed, std::string* error = nullptr);

bool apply_rans_implicit_gpu(DeviceMesh& mesh, Real Re,
    const Real* d_min_dt, std::string* error = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
