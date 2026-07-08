#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_state.hpp"

#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

struct PrimitiveGradient {
    float drho_dx = 0.0f;
    float drho_dy = 0.0f;
    float drho_dz = 0.0f;
    float du_dx = 0.0f;
    float du_dy = 0.0f;
    float du_dz = 0.0f;
    float dv_dx = 0.0f;
    float dv_dy = 0.0f;
    float dv_dz = 0.0f;
    float dw_dx = 0.0f;
    float dw_dy = 0.0f;
    float dw_dz = 0.0f;
    float dp_dx = 0.0f;
    float dp_dy = 0.0f;
    float dp_dz = 0.0f;
};

struct PrimitiveLimiter {
    float rho = 1.0f;
    float u = 1.0f;
    float v = 1.0f;
    float w = 1.0f;
    float p = 1.0f;
};

class DeviceMesh;

std::vector<PrimitiveGradient> compute_green_gauss_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    float gamma);

std::vector<PrimitiveGradient> compute_least_squares_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    float gamma);

std::vector<PrimitiveLimiter> compute_barth_jespersen_limiters(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    float gamma);

PrimitiveGradient apply_limiter(const PrimitiveGradient& gradient, const PrimitiveLimiter& limiter);

bool apply_limiter_gpu(DeviceMesh& mesh, std::string* error = nullptr);
bool apply_limiter_gpu(DeviceMesh& mesh, bool sync, std::string* error = nullptr);

PrimitiveState reconstruct_primitive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    float dx,
    float dy,
    float dz);

PrimitiveState reconstruct_primitive_positive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    float dx,
    float dy,
    float dz,
    float rho_floor,
    float p_floor,
    float* theta = nullptr);

} // namespace Cfd
} // namespace AeroSim
