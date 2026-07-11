#include "aero/cfd/gpu_communicator.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

GpuCommunicator::~GpuCommunicator() {
    finalize();
}

GpuCommunicator::GpuCommunicator(GpuCommunicator&& other) noexcept {
    initialized_ = other.initialized_;
    rank_ = other.rank_;
    size_ = other.size_;
    other.initialized_ = false;
    other.rank_ = 0;
    other.size_ = 1;
}

GpuCommunicator& GpuCommunicator::operator=(GpuCommunicator&& other) noexcept {
    if (this == &other) return *this;
    finalize();
    initialized_ = other.initialized_;
    rank_ = other.rank_;
    size_ = other.size_;
    other.initialized_ = false;
    other.rank_ = 0;
    other.size_ = 1;
    return *this;
}

bool GpuCommunicator::initialize(int* argc, char*** argv, std::string* error) {
#ifdef MPI_ENABLED
    int provided = 0;
    int required = MPI_THREAD_FUNNELED;
    int rc = MPI_Init_thread(argc, argv, required, &provided);
    if (rc != MPI_SUCCESS) {
        if (error) *error = "MPI_Init_thread failed";
        return false;
    }
    if (provided < required) {
        std::fprintf(stderr, "[GpuCommunicator] Warning: MPI thread support %d < requested %d\n", provided, required);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    initialized_ = true;
    return true;
#else
    (void)argc;
    (void)argv;
    rank_ = 0;
    size_ = 1;
    initialized_ = true;
    if (error) *error = "MPI not enabled (compile with -DAEROSIM_MPI=ON)";
    return true;
#endif
}

void GpuCommunicator::finalize() {
#ifdef MPI_ENABLED
    if (initialized_ && size_ > 1) {
        MPI_Finalize();
    }
#endif
    initialized_ = false;
    rank_ = 0;
    size_ = 1;
}

bool GpuCommunicator::assign_device(int devices_per_node, std::string* error) {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess) {
        if (error) *error = "cudaGetDeviceCount failed";
        return false;
    }
    if (n_devices <= 0) {
        if (error) *error = "no CUDA devices found";
        return false;
    }

    int dev = rank_ % devices_per_node;
    const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    if (cvd) {
        dev = rank_ % n_devices;
    }

    if (dev >= n_devices) dev = 0;
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        if (error) *error = "cudaSetDevice(" + std::to_string(dev) + ") failed: " + cudaGetErrorString(err);
        return false;
    }
    return true;
}

bool GpuCommunicator::send_recv_exchange(const Real* send_buf, Real* recv_buf, int count,
    int peer_rank, int tag, std::string* error) {
#ifdef MPI_ENABLED
    int rc_send = MPI_SUCCESS;
    int rc_recv = MPI_SUCCESS;
    if (send_buf && count > 0) {
        rc_send = MPI_Send(const_cast<Real*>(send_buf), count, 
            MPI_DOUBLE_PRECISION, peer_rank, tag, MPI_COMM_WORLD);
        if (rc_send != MPI_SUCCESS) {
            if (error) *error = "MPI_Send to rank " + std::to_string(peer_rank) + " failed";
            return false;
        }
    }
    if (recv_buf && count > 0) {
        rc_recv = MPI_Recv(recv_buf, count, 
            MPI_DOUBLE_PRECISION, peer_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rc_recv != MPI_SUCCESS) {
            if (error) *error = "MPI_Recv from rank " + std::to_string(peer_rank) + " failed";
            return false;
        }
    }
    return true;
#else
    (void)send_buf;
    (void)recv_buf;
    (void)count;
    (void)peer_rank;
    (void)tag;
    if (error) *error = "MPI not enabled";
    return false;
#endif
}

bool GpuCommunicator::allreduce_min(Real& value, std::string* error) {
#ifdef MPI_ENABLED
    if (size_ <= 1) return true;
    Real result = 0;
    int rc = MPI_Allreduce(&value, &result, 1, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
        if (error) *error = "MPI_Allreduce MIN failed";
        return false;
    }
    value = result;
    return true;
#else
    (void)value;
    if (error) *error = "MPI not enabled";
    return false;
#endif
}

bool GpuCommunicator::allreduce_sum(Real& value, std::string* error) {
#ifdef MPI_ENABLED
    if (size_ <= 1) return true;
    Real result = 0;
    int rc = MPI_Allreduce(&value, &result, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
        if (error) *error = "MPI_Allreduce SUM failed";
        return false;
    }
    value = result;
    return true;
#else
    (void)value;
    if (error) *error = "MPI not enabled";
    return false;
#endif
}

bool GpuCommunicator::allreduce_sum(int& value, std::string* error) {
#ifdef MPI_ENABLED
    if (size_ <= 1) return true;
    int result = 0;
    int rc = MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
        if (error) *error = "MPI_Allreduce SUM int failed";
        return false;
    }
    value = result;
    return true;
#else
    (void)value;
    if (error) *error = "MPI not enabled";
    return false;
#endif
}

bool GpuCommunicator::allreduce_sum(Real* values, int count, std::string* error) {
#ifdef MPI_ENABLED
    if (size_ <= 1) return true;
    std::vector<Real> result(count);
    int rc = MPI_Allreduce(values, result.data(), count, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
        if (error) *error = "MPI_Allreduce SUM array failed";
        return false;
    }
    for (int i = 0; i < count; ++i) values[i] = result[i];
    return true;
#else
    (void)values;
    (void)count;
    if (error) *error = "MPI not enabled";
    return false;
#endif
}

bool GpuCommunicator::barrier(std::string* error) {
#ifdef MPI_ENABLED
    if (size_ <= 1) return true;
    int rc = MPI_Barrier(MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
        if (error) *error = "MPI_Barrier failed";
        return false;
    }
    return true;
#else
    if (error) *error = "MPI not enabled";
    return false;
#endif
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
