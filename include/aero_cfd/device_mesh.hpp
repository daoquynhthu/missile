#pragma once

#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/reconstruction.hpp"

#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

struct DeviceFaceData {
    float* nx = nullptr;
    float* ny = nullptr;
    float* nz = nullptr;
    float* area = nullptr;
    int* left_cell = nullptr;
    int* right_cell = nullptr;
    int* boundary = nullptr;
};

struct DeviceCellData {
    float* volume = nullptr;
    float* h_min = nullptr;
    float* cx = nullptr;
    float* cy = nullptr;
    float* cz = nullptr;
};

class DeviceMesh {
public:
    static constexpr int NVAR = 5;

    DeviceMesh() = default;
    ~DeviceMesh();

    DeviceMesh(const DeviceMesh&) = delete;
    DeviceMesh& operator=(const DeviceMesh&) = delete;

    DeviceMesh(DeviceMesh&& other) noexcept;
    DeviceMesh& operator=(DeviceMesh&& other) noexcept;

    bool upload_mesh(const CfdMesh& mesh, std::string* error = nullptr);
    bool upload_state(const std::vector<ConservativeState>& q, std::string* error = nullptr);
    bool upload_gradients(const std::vector<PrimitiveGradient>& gradients, std::string* error = nullptr);
    bool upload_limiters(const std::vector<PrimitiveLimiter>& limiters, std::string* error = nullptr);
    bool clear_residual(std::string* error = nullptr);

    bool download_state(std::vector<ConservativeState>& q, std::string* error = nullptr) const;
    bool download_residual(std::vector<EulerFlux>& residual, std::string* error = nullptr) const;
    bool download_gradients(std::vector<PrimitiveGradient>& gradients, std::string* error = nullptr) const;

    void release();

    int cell_count() const { return cell_count_; }
    int face_count() const { return face_count_; }

    DeviceFaceData face_data() const;
    DeviceCellData cell_data() const;

    float* state_device() const { return d_q_; }
    float* residual_device() const { return d_residual_; }
    float* gradients_device() const { return d_gradients_; }
    float* limiters_device() const { return d_limiters_; }

private:
    int cell_count_ = 0;
    int face_count_ = 0;

    float* d_q_ = nullptr;
    float* d_residual_ = nullptr;

    float* d_nx_ = nullptr;
    float* d_ny_ = nullptr;
    float* d_nz_ = nullptr;
    float* d_area_ = nullptr;
    int* d_left_cell_ = nullptr;
    int* d_right_cell_ = nullptr;
    int* d_boundary_ = nullptr;

    float* d_volume_ = nullptr;
    float* d_h_min_ = nullptr;
    float* d_cx_ = nullptr;
    float* d_cy_ = nullptr;
    float* d_cz_ = nullptr;

    float* d_gradients_ = nullptr;
    float* d_limiters_ = nullptr;
};

} // namespace Cfd
} // namespace AeroSim
