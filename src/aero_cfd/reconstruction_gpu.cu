#include "aero_cfd/reconstruction.hpp"
#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/gpu_buffers.hpp"

#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

namespace {

__device__ PrimitiveGradient d_apply_limiter(PrimitiveGradient gradient, PrimitiveLimiter limiter) {
    gradient.drho_dx *= limiter.rho;
    gradient.drho_dy *= limiter.rho;
    gradient.drho_dz *= limiter.rho;
    gradient.du_dx *= limiter.u;
    gradient.du_dy *= limiter.u;
    gradient.du_dz *= limiter.u;
    gradient.dv_dx *= limiter.v;
    gradient.dv_dy *= limiter.v;
    gradient.dv_dz *= limiter.v;
    gradient.dw_dx *= limiter.w;
    gradient.dw_dy *= limiter.w;
    gradient.dw_dz *= limiter.w;
    gradient.dp_dx *= limiter.p;
    gradient.dp_dy *= limiter.p;
    gradient.dp_dz *= limiter.p;
    return gradient;
}

__global__ void apply_limiter_kernel(PrimitiveGradient* gradients, const PrimitiveLimiter* limiters, int cell_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cell_count) return;
    gradients[idx] = d_apply_limiter(gradients[idx], limiters[idx]);
}

} // namespace

bool apply_limiter_gpu(GpuCfdBuffers& buffers, std::string* error) {
    if (buffers.cell_count() <= 0 || !buffers.gradients_device() || !buffers.limiters_device()) {
        if (error) *error = "GPU gradient/limiter buffers are not ready";
        return false;
    }

    int block = 128;
    int grid = (buffers.cell_count() + block - 1) / block;
    apply_limiter_kernel<<<grid, block>>>(buffers.gradients_device(), buffers.limiters_device(), buffers.cell_count());
    if (!cuda_check(cudaGetLastError(), "apply_limiter_kernel launch", error)) return false;
    return cuda_check(cudaDeviceSynchronize(), "apply_limiter_kernel synchronize", error);
}

} // namespace Cfd
} // namespace AeroSim
