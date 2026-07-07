#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

bool compute_euler_residual_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    float gamma,
    std::vector<EulerFlux>& residual);

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    float gamma,
    std::string* error = nullptr);

bool compute_euler_residual_gpu_timed(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    float gamma,
    float* elapsed_ms,
    std::string* error = nullptr);

std::size_t estimate_euler_residual_gpu_bytes(const CfdMesh& mesh);

bool compute_euler_residual_gpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    float gamma,
    std::vector<EulerFlux>& residual,
    std::string* error = nullptr);

} // namespace Cfd
} // namespace AeroSim
