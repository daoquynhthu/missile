#include "aero_cfd/reconstruction.hpp"
#include "aero_cfd/cuda_utils.hpp"
#include "aero_cfd/device_mesh.hpp"

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

bool apply_limiter_gpu(DeviceMesh& mesh, std::string* error) {
    if (mesh.cell_count() <= 0 || !mesh.gradients_device() || !mesh.limiters_device()) {
        if (error) *error = "GPU gradient/limiter buffers are not ready";
        return false;
    }

    int block = 128;
    int grid = (mesh.cell_count() + block - 1) / block;
    apply_limiter_kernel<<<grid, block>>>(
        reinterpret_cast<PrimitiveGradient*>(mesh.gradients_device()),
        reinterpret_cast<const PrimitiveLimiter*>(mesh.limiters_device()),
        mesh.cell_count());
    if (!cuda_check(cudaGetLastError(), "apply_limiter_kernel launch", error)) return false;
    return cuda_check(cudaDeviceSynchronize(), "apply_limiter_kernel synchronize", error);
}

} // namespace Cfd
} // namespace AeroSim
