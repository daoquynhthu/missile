#include "aero_cfd/real.hpp"
#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

bool compute_euler_residual_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual);

bool launch_euler_residual_kernel(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    int* d_failed,
    cudaEvent_t start_event = nullptr,
    std::string* error = nullptr,
    int reconstruction_order = 1);

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    int* d_failed,
    std::string* error = nullptr,
    int reconstruction_order = 1);

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    std::string* error = nullptr,
    int reconstruction_order = 1);

bool compute_euler_residual_gpu_timed(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    Real* elapsed_ms,
    std::string* error = nullptr,
    int reconstruction_order = 1);

std::size_t estimate_euler_residual_gpu_bytes(const CfdMesh& mesh);

bool compute_euler_residual_gpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    std::string* error = nullptr);

} // namespace Cfd
} // namespace AeroSim


