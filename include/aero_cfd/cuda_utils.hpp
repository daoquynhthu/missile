#pragma once

#include <cuda_runtime.h>
#include <string>

namespace AeroSim {
namespace Cfd {

#define CUDA_KERNEL_CHECK(msg) cuda_check(cudaGetLastError(), msg)

bool cuda_check(cudaError_t error, const char* action, std::string* message = nullptr);

} // namespace Cfd
} // namespace AeroSim
