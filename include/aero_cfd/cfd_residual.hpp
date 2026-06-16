#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/gpu_buffers.hpp"

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
    GpuCfdBuffers& buffers,
    const PrimitiveState& freestream,
    float gamma,
    std::string* error = nullptr);

bool compute_euler_residual_gpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    float gamma,
    std::vector<EulerFlux>& residual,
    std::string* error = nullptr);

} // namespace Cfd
} // namespace AeroSim
