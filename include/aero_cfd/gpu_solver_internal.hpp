#pragma once

#include "aero_cfd/device_mesh.hpp"

namespace AeroSim {
namespace Cfd {

bool compute_timestep_gpu(DeviceMesh& mesh, float gamma, float cfl, float* d_min_dt);

bool compute_gradients_gpu(DeviceMesh& mesh, float gamma, std::string* error = nullptr,
    int* d_failed = nullptr);

bool compute_limiters_gpu(DeviceMesh& mesh, float gamma, std::string* error = nullptr,
    int* d_failed = nullptr);

bool compute_update_gpu(DeviceMesh& mesh, const float* d_min_dt, float gamma,
    float* d_l2_sum, int* d_failed,
    int* d_failure_cell = nullptr, float* d_failure_state = nullptr);

bool compute_wall_forces_gpu(DeviceMesh& mesh, float gamma, float* d_forces);

bool compute_state_bounds_gpu(DeviceMesh& mesh, float gamma, float* d_bounds_slot);

bool compute_failure_snapshot_gpu(DeviceMesh& mesh, float gamma,
    int* d_failure_cell, float* d_failure_state);

} // namespace Cfd
} // namespace AeroSim
