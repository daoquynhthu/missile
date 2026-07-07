#include "aero_cfd/device_mesh.hpp"
#include "aero_cfd/cuda_utils.hpp"

#include <cassert>
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
    d_face_cx_ = other.d_face_cx_;
    d_face_cy_ = other.d_face_cy_;
    d_face_cz_ = other.d_face_cz_;
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
    other.d_face_cx_ = other.d_face_cy_ = other.d_face_cz_ = nullptr;
    other.d_gradients_ = nullptr;
    other.d_limiters_ = nullptr;
    return *this;
}

DeviceFaceData DeviceMesh::face_data() {
    DeviceFaceData fd;
    fd.nx = d_nx_;
    fd.ny = d_ny_;
    fd.nz = d_nz_;
    fd.area = d_area_;
    fd.left_cell = d_left_cell_;
    fd.right_cell = d_right_cell_;
    fd.boundary = d_boundary_;
    fd.cx = d_face_cx_;
    fd.cy = d_face_cy_;
    fd.cz = d_face_cz_;
    return fd;
}

ConstDeviceFaceData DeviceMesh::face_data() const {
    ConstDeviceFaceData fd;
    fd.nx = d_nx_;
    fd.ny = d_ny_;
    fd.nz = d_nz_;
    fd.area = d_area_;
    fd.left_cell = d_left_cell_;
    fd.right_cell = d_right_cell_;
    fd.boundary = d_boundary_;
    fd.cx = d_face_cx_;
    fd.cy = d_face_cy_;
    fd.cz = d_face_cz_;
    return fd;
}

DeviceCellData DeviceMesh::cell_data() {
    DeviceCellData cd;
    cd.volume = d_volume_;
    cd.h_min = d_h_min_;
    cd.cx = d_cx_;
    cd.cy = d_cy_;
    cd.cz = d_cz_;
    return cd;
}

ConstDeviceCellData DeviceMesh::cell_data() const {
    ConstDeviceCellData cd;
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
#define FREE_AND_ASSERT(ptr) do { bool ok = cuda_free_and_null(reinterpret_cast<void*&>(ptr)); assert(ok); } while(0)
    FREE_AND_ASSERT(d_q_);
    FREE_AND_ASSERT(d_residual_);
    FREE_AND_ASSERT(d_nx_);
    FREE_AND_ASSERT(d_ny_);
    FREE_AND_ASSERT(d_nz_);
    FREE_AND_ASSERT(d_area_);
    FREE_AND_ASSERT(d_left_cell_);
    FREE_AND_ASSERT(d_right_cell_);
    FREE_AND_ASSERT(d_boundary_);
    FREE_AND_ASSERT(d_volume_);
    FREE_AND_ASSERT(d_h_min_);
    FREE_AND_ASSERT(d_cx_);
    FREE_AND_ASSERT(d_cy_);
    FREE_AND_ASSERT(d_cz_);
    FREE_AND_ASSERT(d_face_cx_);
    FREE_AND_ASSERT(d_face_cy_);
    FREE_AND_ASSERT(d_face_cz_);
    FREE_AND_ASSERT(d_gradients_);
    FREE_AND_ASSERT(d_limiters_);
#undef FREE_AND_ASSERT
    cell_count_ = 0;
    face_count_ = 0;
}

bool DeviceMesh::upload_mesh(const CfdMesh& mesh, std::string* error) {
    release();
    cell_count_ = mesh.cells.size();
    face_count_ = mesh.faces.size();
    if (cell_count_ == 0 || face_count_ == 0) {
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

    std::size_t nf = face_count_;
    std::size_t nc = cell_count_;

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

    if (!alloc(d_face_cx_, nf * sizeof(float), "cudaMalloc face cx")) return false;
    if (!alloc(d_face_cy_, nf * sizeof(float), "cudaMalloc face cy")) return false;
    if (!alloc(d_face_cz_, nf * sizeof(float), "cudaMalloc face cz")) return false;

    if (!alloc(d_volume_, nc * sizeof(float), "cudaMalloc cell volume")) return false;
    if (!alloc(d_h_min_, nc * sizeof(float), "cudaMalloc cell h_min")) return false;
    if (!alloc(d_cx_, nc * sizeof(float), "cudaMalloc cell cx")) return false;
    if (!alloc(d_cy_, nc * sizeof(float), "cudaMalloc cell cy")) return false;
    if (!alloc(d_cz_, nc * sizeof(float), "cudaMalloc cell cz")) return false;

    if (!alloc(d_q_, nc * NVAR * sizeof(float), "cudaMalloc state")) return false;
    if (!alloc(d_residual_, nc * NVAR * sizeof(float), "cudaMalloc residual")) return false;

    std::vector<float> h_nx(nf), h_ny(nf), h_nz(nf), h_area(nf), h_face_cx(nf), h_face_cy(nf), h_face_cz(nf);
    std::vector<int> h_left_cell(nf), h_right_cell(nf);
    for (std::size_t i = 0; i < nf; ++i) {
        const auto& f = mesh.faces[i];
        h_nx[i] = f.nx; h_ny[i] = f.ny; h_nz[i] = f.nz;
        h_area[i] = f.area;
        h_left_cell[i] = f.left_cell; h_right_cell[i] = f.right_cell;
        h_face_cx[i] = f.cx; h_face_cy[i] = f.cy; h_face_cz[i] = f.cz;
    }
    if (!copy(d_nx_, h_nx.data(), nf * sizeof(float), "cudaMemcpy nx")) return false;
    if (!copy(d_ny_, h_ny.data(), nf * sizeof(float), "cudaMemcpy ny")) return false;
    if (!copy(d_nz_, h_nz.data(), nf * sizeof(float), "cudaMemcpy nz")) return false;
    if (!copy(d_area_, h_area.data(), nf * sizeof(float), "cudaMemcpy area")) return false;
    if (!copy(d_left_cell_, h_left_cell.data(), nf * sizeof(int), "cudaMemcpy left_cell")) return false;
    if (!copy(d_right_cell_, h_right_cell.data(), nf * sizeof(int), "cudaMemcpy right_cell")) return false;
    if (!copy(d_boundary_, temp_boundary.data(), nf * sizeof(int), "cudaMemcpy boundary")) return false;
    if (!copy(d_face_cx_, h_face_cx.data(), nf * sizeof(float), "cudaMemcpy face cx")) return false;
    if (!copy(d_face_cy_, h_face_cy.data(), nf * sizeof(float), "cudaMemcpy face cy")) return false;
    if (!copy(d_face_cz_, h_face_cz.data(), nf * sizeof(float), "cudaMemcpy face cz")) return false;

    std::vector<float> h_volume(nc), h_h_min(nc), h_cx(nc), h_cy(nc), h_cz(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        const auto& c = mesh.cells[i];
        h_volume[i] = c.volume; h_h_min[i] = c.h_min;
        h_cx[i] = c.cx; h_cy[i] = c.cy; h_cz[i] = c.cz;
    }
    if (!copy(d_volume_, h_volume.data(), nc * sizeof(float), "cudaMemcpy volume")) return false;
    if (!copy(d_h_min_, h_h_min.data(), nc * sizeof(float), "cudaMemcpy h_min")) return false;
    if (!copy(d_cx_, h_cx.data(), nc * sizeof(float), "cudaMemcpy cx")) return false;
    if (!copy(d_cy_, h_cy.data(), nc * sizeof(float), "cudaMemcpy cy")) return false;
    if (!copy(d_cz_, h_cz.data(), nc * sizeof(float), "cudaMemcpy cz")) return false;

    if (!cuda_check(cudaMemset(d_residual_, 0, nc * NVAR * sizeof(float)), "cudaMemset residual", error)) {
        release();
        return false;
    }

    return true;
}

bool DeviceMesh::upload_state(const std::vector<ConservativeState>& q, std::string* error) {
    if (cell_count_ == 0 || static_cast<int>(q.size()) != cell_count_) {
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
    if (cell_count_ == 0 || static_cast<int>(gradients.size()) != cell_count_) {
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
    if (cell_count_ == 0 || static_cast<int>(limiters.size()) != cell_count_) {
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
    if (!d_residual_ || cell_count_ == 0) {
        if (error) *error = "residual buffer not allocated";
        return false;
    }
    return cuda_check(cudaMemset(d_residual_, 0, static_cast<std::size_t>(cell_count_) * NVAR * sizeof(float)), "cudaMemset clear_residual", error);
}

bool DeviceMesh::download_state(std::vector<ConservativeState>& q, std::string* error) const {
    if (!d_q_ || cell_count_ == 0) {
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
    if (!d_residual_ || cell_count_ == 0) {
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
    if (!d_gradients_ || cell_count_ == 0) {
        if (error) *error = "gradient buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    gradients.resize(nc);
    return cuda_check(cudaMemcpy(gradients.data(), d_gradients_, nc * sizeof(PrimitiveGradient), cudaMemcpyDeviceToHost), "cudaMemcpy download_gradients", error);
}

} // namespace Cfd
} // namespace AeroSim
