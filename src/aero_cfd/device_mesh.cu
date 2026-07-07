#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <utility>

namespace AeroSim {
namespace Cfd {

DeviceMesh::~DeviceMesh() {
    release();
}

DeviceMesh::DeviceMesh(DeviceMesh&& other) noexcept {
    *this = std::move(other);
}

DeviceMesh& DeviceMesh::operator=(DeviceMesh&& other) noexcept {
    if (this == &other) return *this;
    release();
    cell_count_ = other.cell_count_;
    face_count_ = other.face_count_;
    d_q_ = other.d_q_;
    d_residual_ = other.d_residual_;
    d_nx_ = other.d_nx_;
    d_ny_ = other.d_ny_;
    d_nz_ = other.d_nz_;
    d_area_ = other.d_area_;
    d_left_cell_ = other.d_left_cell_;
    d_right_cell_ = other.d_right_cell_;
    d_boundary_ = other.d_boundary_;
    d_volume_ = other.d_volume_;
    d_h_min_ = other.d_h_min_;
    d_cx_ = other.d_cx_;
    d_cy_ = other.d_cy_;
    d_cz_ = other.d_cz_;
    d_gradients_ = other.d_gradients_;
    d_limiters_ = other.d_limiters_;
    other.cell_count_ = 0;
    other.face_count_ = 0;
    other.d_q_ = nullptr;
    other.d_residual_ = nullptr;
    other.d_nx_ = other.d_ny_ = other.d_nz_ = nullptr;
    other.d_area_ = nullptr;
    other.d_left_cell_ = other.d_right_cell_ = nullptr;
    other.d_boundary_ = nullptr;
    other.d_volume_ = other.d_h_min_ = nullptr;
    other.d_cx_ = other.d_cy_ = other.d_cz_ = nullptr;
    other.d_gradients_ = nullptr;
    other.d_limiters_ = nullptr;
    return *this;
}

DeviceFaceData DeviceMesh::face_data() const {
    DeviceFaceData fd;
    fd.nx = d_nx_;
    fd.ny = d_ny_;
    fd.nz = d_nz_;
    fd.area = d_area_;
    fd.left_cell = d_left_cell_;
    fd.right_cell = d_right_cell_;
    fd.boundary = d_boundary_;
    return fd;
}

DeviceCellData DeviceMesh::cell_data() const {
    DeviceCellData cd;
    cd.volume = d_volume_;
    cd.h_min = d_h_min_;
    cd.cx = d_cx_;
    cd.cy = d_cy_;
    cd.cz = d_cz_;
    return cd;
}

static bool cuda_free_and_null(void*& ptr) {
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        ptr = nullptr;
        return err == cudaSuccess;
    }
    return true;
}

void DeviceMesh::release() {
    cuda_free_and_null(reinterpret_cast<void*&>(d_q_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_residual_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_nx_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_ny_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_nz_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_area_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_left_cell_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_right_cell_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_boundary_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_volume_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_h_min_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_cx_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_cy_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_cz_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_gradients_));
    cuda_free_and_null(reinterpret_cast<void*&>(d_limiters_));
    cell_count_ = 0;
    face_count_ = 0;
}

bool DeviceMesh::upload_mesh(const CfdMesh& mesh, std::string* error) {
    release();
    cell_count_ = static_cast<int>(mesh.cells.size());
    face_count_ = static_cast<int>(mesh.faces.size());
    if (cell_count_ <= 0 || face_count_ <= 0) {
        if (error) *error = "mesh is empty";
        release();
        return false;
    }

    auto alloc = [&](auto*& ptr, std::size_t bytes, const char* name) -> bool {
        if (!cuda_check(cudaMalloc(&ptr, bytes), name, error)) { release(); return false; }
        return true;
    };

    auto copy = [&](auto* dst, const auto* src, std::size_t bytes, const char* name) -> bool {
        if (!cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), name, error)) { release(); return false; }
        return true;
    };

    std::size_t nf = static_cast<std::size_t>(face_count_);
    std::size_t nc = static_cast<std::size_t>(cell_count_);

    if (!alloc(d_nx_, nf * sizeof(float), "cudaMalloc face nx")) return false;
    if (!alloc(d_ny_, nf * sizeof(float), "cudaMalloc face ny")) return false;
    if (!alloc(d_nz_, nf * sizeof(float), "cudaMalloc face nz")) return false;
    if (!alloc(d_area_, nf * sizeof(float), "cudaMalloc face area")) return false;
    if (!alloc(d_left_cell_, nf * sizeof(int), "cudaMalloc face left_cell")) return false;
    if (!alloc(d_right_cell_, nf * sizeof(int), "cudaMalloc face right_cell")) return false;
    if (!alloc(d_boundary_, nf * sizeof(int), "cudaMalloc face boundary")) return false;

    std::vector<int> temp_boundary(nf);
    for (std::size_t i = 0; i < nf; ++i) {
        temp_boundary[i] = static_cast<int>(mesh.faces[i].boundary);
    }

    if (!alloc(d_volume_, nc * sizeof(float), "cudaMalloc cell volume")) return false;
    if (!alloc(d_h_min_, nc * sizeof(float), "cudaMalloc cell h_min")) return false;
    if (!alloc(d_cx_, nc * sizeof(float), "cudaMalloc cell cx")) return false;
    if (!alloc(d_cy_, nc * sizeof(float), "cudaMalloc cell cy")) return false;
    if (!alloc(d_cz_, nc * sizeof(float), "cudaMalloc cell cz")) return false;

    if (!alloc(d_q_, nc * NVAR * sizeof(float), "cudaMalloc state")) return false;
    if (!alloc(d_residual_, nc * NVAR * sizeof(float), "cudaMalloc residual")) return false;

    for (std::size_t i = 0; i < nf; ++i) {
        const auto& f = mesh.faces[i];
        if (!copy(&d_nx_[i], &f.nx, sizeof(float), "cudaMemcpy nx")) return false;
        if (!copy(&d_ny_[i], &f.ny, sizeof(float), "cudaMemcpy ny")) return false;
        if (!copy(&d_nz_[i], &f.nz, sizeof(float), "cudaMemcpy nz")) return false;
        if (!copy(&d_area_[i], &f.area, sizeof(float), "cudaMemcpy area")) return false;
        if (!copy(&d_left_cell_[i], &f.left_cell, sizeof(int), "cudaMemcpy left_cell")) return false;
        if (!copy(&d_right_cell_[i], &f.right_cell, sizeof(int), "cudaMemcpy right_cell")) return false;
    }
    if (!copy(d_boundary_, temp_boundary.data(), nf * sizeof(int), "cudaMemcpy boundary")) return false;

    for (std::size_t i = 0; i < nc; ++i) {
        const auto& c = mesh.cells[i];
        if (!copy(&d_volume_[i], &c.volume, sizeof(float), "cudaMemcpy volume")) return false;
        if (!copy(&d_h_min_[i], &c.h_min, sizeof(float), "cudaMemcpy h_min")) return false;
        if (!copy(&d_cx_[i], &c.cx, sizeof(float), "cudaMemcpy cx")) return false;
        if (!copy(&d_cy_[i], &c.cy, sizeof(float), "cudaMemcpy cy")) return false;
        if (!copy(&d_cz_[i], &c.cz, sizeof(float), "cudaMemcpy cz")) return false;
    }

    if (!cuda_check(cudaMemset(d_residual_, 0, nc * NVAR * sizeof(float)), "cudaMemset residual", error)) {
        release();
        return false;
    }

    return true;
}

bool DeviceMesh::upload_state(const std::vector<ConservativeState>& q, std::string* error) {
    if (cell_count_ <= 0 || static_cast<int>(q.size()) != cell_count_) {
        if (error) *error = "state size mismatch";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    std::vector<float> flat(nc * NVAR);
    for (std::size_t i = 0; i < nc; ++i) {
        flat[i * NVAR + 0] = q[i].rho;
        flat[i * NVAR + 1] = q[i].rho_u;
        flat[i * NVAR + 2] = q[i].rho_v;
        flat[i * NVAR + 3] = q[i].rho_w;
        flat[i * NVAR + 4] = q[i].rho_E;
    }
    return cuda_check(cudaMemcpy(d_q_, flat.data(), nc * NVAR * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy upload_state", error);
}

bool DeviceMesh::upload_gradients(const std::vector<PrimitiveGradient>& gradients, std::string* error) {
    if (cell_count_ <= 0 || static_cast<int>(gradients.size()) != cell_count_) {
        if (error) *error = "gradient size mismatch";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!d_gradients_) {
        if (!cuda_check(cudaMalloc(&d_gradients_, nc * 15 * sizeof(float)), "cudaMalloc gradients", error)) return false;
    }
    return cuda_check(cudaMemcpy(d_gradients_, gradients.data(), nc * sizeof(PrimitiveGradient), cudaMemcpyHostToDevice), "cudaMemcpy upload_gradients", error);
}

bool DeviceMesh::upload_limiters(const std::vector<PrimitiveLimiter>& limiters, std::string* error) {
    if (cell_count_ <= 0 || static_cast<int>(limiters.size()) != cell_count_) {
        if (error) *error = "limiter size mismatch";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!d_limiters_) {
        if (!cuda_check(cudaMalloc(&d_limiters_, nc * sizeof(PrimitiveLimiter)), "cudaMalloc limiters", error)) return false;
    }
    return cuda_check(cudaMemcpy(d_limiters_, limiters.data(), nc * sizeof(PrimitiveLimiter), cudaMemcpyHostToDevice), "cudaMemcpy upload_limiters", error);
}

bool DeviceMesh::clear_residual(std::string* error) {
    if (!d_residual_ || cell_count_ <= 0) {
        if (error) *error = "residual buffer not allocated";
        return false;
    }
    return cuda_check(cudaMemset(d_residual_, 0, static_cast<std::size_t>(cell_count_) * NVAR * sizeof(float)), "cudaMemset clear_residual", error);
}

bool DeviceMesh::download_state(std::vector<ConservativeState>& q, std::string* error) const {
    if (!d_q_ || cell_count_ <= 0) {
        if (error) *error = "state buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    std::vector<float> flat(nc * NVAR);
    if (!cuda_check(cudaMemcpy(flat.data(), d_q_, nc * NVAR * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy download_state", error)) return false;
    q.resize(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        q[i].rho = flat[i * NVAR + 0];
        q[i].rho_u = flat[i * NVAR + 1];
        q[i].rho_v = flat[i * NVAR + 2];
        q[i].rho_w = flat[i * NVAR + 3];
        q[i].rho_E = flat[i * NVAR + 4];
    }
    return true;
}

bool DeviceMesh::download_residual(std::vector<EulerFlux>& residual, std::string* error) const {
    if (!d_residual_ || cell_count_ <= 0) {
        if (error) *error = "residual buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    std::vector<float> flat(nc * NVAR);
    if (!cuda_check(cudaMemcpy(flat.data(), d_residual_, nc * NVAR * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy download_residual", error)) return false;
    residual.resize(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        residual[i].mass = flat[i * NVAR + 0];
        residual[i].mom_x = flat[i * NVAR + 1];
        residual[i].mom_y = flat[i * NVAR + 2];
        residual[i].mom_z = flat[i * NVAR + 3];
        residual[i].energy = flat[i * NVAR + 4];
    }
    return true;
}

bool DeviceMesh::download_gradients(std::vector<PrimitiveGradient>& gradients, std::string* error) const {
    if (!d_gradients_ || cell_count_ <= 0) {
        if (error) *error = "gradient buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    gradients.resize(nc);
    return cuda_check(cudaMemcpy(gradients.data(), d_gradients_, nc * sizeof(PrimitiveGradient), cudaMemcpyDeviceToHost), "cudaMemcpy download_gradients", error);
}

} // namespace Cfd
} // namespace AeroSim
