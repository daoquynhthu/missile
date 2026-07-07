#pragma once

#include "aero_cfd/device_mesh.hpp"

namespace AeroSim {
namespace Cfd {

bool compute_timestep_gpu(DeviceMesh& mesh, float gamma, float cfl, float* d_min_dt);

bool compute_update_gpu(DeviceMesh& mesh, const float* d_min_dt, float gamma,
    float* d_l2_sum, int* d_failed);

bool compute_wall_forces_gpu(DeviceMesh& mesh, float gamma, float* d_forces);

} // namespace Cfd
} // namespace AeroSim
