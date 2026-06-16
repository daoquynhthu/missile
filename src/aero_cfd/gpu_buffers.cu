#include "aero_cfd/gpu_buffers.hpp"
#include "aero_cfd/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <utility>

namespace AeroSim {
namespace Cfd {

GpuCfdBuffers::~GpuCfdBuffers() {
    release();
}

GpuCfdBuffers::GpuCfdBuffers(GpuCfdBuffers&& other) noexcept {
    *this = std::move(other);
}

GpuCfdBuffers& GpuCfdBuffers::operator=(GpuCfdBuffers&& other) noexcept {
    if (this == &other) return *this;
    release();
    d_faces_ = other.d_faces_;
    d_state_ = other.d_state_;
    d_residual_ = other.d_residual_;
    cell_count_ = other.cell_count_;
    face_count_ = other.face_count_;
    other.d_faces_ = nullptr;
    other.d_state_ = nullptr;
    other.d_residual_ = nullptr;
    other.cell_count_ = 0;
    other.face_count_ = 0;
    return *this;
}

bool GpuCfdBuffers::upload_mesh(const CfdMesh& mesh, std::string* error) {
    release();
    cell_count_ = static_cast<int>(mesh.cells.size());
    face_count_ = static_cast<int>(mesh.faces.size());
    if (cell_count_ <= 0 || face_count_ <= 0) {
        if (error) *error = "mesh is empty";
        release();
        return false;
    }

    std::size_t face_bytes = mesh.faces.size() * sizeof(CfdFace);
    std::size_t residual_bytes = mesh.cells.size() * sizeof(EulerFlux);
    if (!cuda_check(cudaMalloc(&d_faces_, face_bytes), "cudaMalloc faces", error)) {
        release();
        return false;
    }
    if (!cuda_check(cudaMalloc(&d_residual_, residual_bytes), "cudaMalloc residual", error)) {
        release();
        return false;
    }
    if (!cuda_check(cudaMemcpy(d_faces_, mesh.faces.data(), face_bytes, cudaMemcpyHostToDevice), "cudaMemcpy faces", error)) {
        release();
        return false;
    }
    return clear_residual(error);
}

bool GpuCfdBuffers::upload_state(const std::vector<ConservativeState>& q, std::string* error) {
    if (cell_count_ <= 0 || static_cast<int>(q.size()) != cell_count_) {
        if (error) *error = "state size does not match uploaded mesh";
        return false;
    }

    if (d_state_) {
        cudaFree(d_state_);
        d_state_ = nullptr;
    }
    std::size_t state_bytes = q.size() * sizeof(ConservativeState);
    if (!cuda_check(cudaMalloc(&d_state_, state_bytes), "cudaMalloc state", error)) return false;
    return cuda_check(cudaMemcpy(d_state_, q.data(), state_bytes, cudaMemcpyHostToDevice), "cudaMemcpy state", error);
}

bool GpuCfdBuffers::clear_residual(std::string* error) {
    if (!d_residual_ || cell_count_ <= 0) {
        if (error) *error = "residual buffer is not allocated";
        return false;
    }
    std::size_t residual_bytes = static_cast<std::size_t>(cell_count_) * sizeof(EulerFlux);
    return cuda_check(cudaMemset(d_residual_, 0, residual_bytes), "cudaMemset residual", error);
}

bool GpuCfdBuffers::download_residual(std::vector<EulerFlux>& residual, std::string* error) const {
    if (!d_residual_ || cell_count_ <= 0) {
        if (error) *error = "residual buffer is not allocated";
        return false;
    }
    residual.assign(static_cast<std::size_t>(cell_count_), EulerFlux{});
    std::size_t residual_bytes = residual.size() * sizeof(EulerFlux);
    return cuda_check(cudaMemcpy(residual.data(), d_residual_, residual_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy residual", error);
}

void GpuCfdBuffers::release() {
    cudaFree(d_faces_);
    cudaFree(d_state_);
    cudaFree(d_residual_);
    d_faces_ = nullptr;
    d_state_ = nullptr;
    d_residual_ = nullptr;
    cell_count_ = 0;
    face_count_ = 0;
}

} // namespace Cfd
} // namespace AeroSim
