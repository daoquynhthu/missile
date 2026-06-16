#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/reconstruction.hpp"
#include "aero_cfd/cfd_state.hpp"

#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

class GpuCfdBuffers {
public:
    GpuCfdBuffers() = default;
    ~GpuCfdBuffers();

    GpuCfdBuffers(const GpuCfdBuffers&) = delete;
    GpuCfdBuffers& operator=(const GpuCfdBuffers&) = delete;

    GpuCfdBuffers(GpuCfdBuffers&& other) noexcept;
    GpuCfdBuffers& operator=(GpuCfdBuffers&& other) noexcept;

    bool upload_mesh(const CfdMesh& mesh, std::string* error = nullptr);
    bool upload_state(const std::vector<ConservativeState>& q, std::string* error = nullptr);
    bool upload_gradients(const std::vector<PrimitiveGradient>& gradients, std::string* error = nullptr);
    bool upload_limiters(const std::vector<PrimitiveLimiter>& limiters, std::string* error = nullptr);
    bool clear_residual(std::string* error = nullptr);
    bool download_residual(std::vector<EulerFlux>& residual, std::string* error = nullptr) const;
    bool download_gradients(std::vector<PrimitiveGradient>& gradients, std::string* error = nullptr) const;

    void release();

    int cell_count() const { return cell_count_; }
    int face_count() const { return face_count_; }

    const CfdFace* faces_device() const { return d_faces_; }
    const ConservativeState* state_device() const { return d_state_; }
    EulerFlux* residual_device() const { return d_residual_; }
    PrimitiveGradient* gradients_device() const { return d_gradients_; }
    const PrimitiveLimiter* limiters_device() const { return d_limiters_; }

private:
    CfdFace* d_faces_ = nullptr;
    ConservativeState* d_state_ = nullptr;
    EulerFlux* d_residual_ = nullptr;
    PrimitiveGradient* d_gradients_ = nullptr;
    PrimitiveLimiter* d_limiters_ = nullptr;
    int cell_count_ = 0;
    int face_count_ = 0;
};

} // namespace Cfd
} // namespace AeroSim
