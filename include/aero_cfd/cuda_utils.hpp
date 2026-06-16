#pragma once

#include <cuda_runtime.h>
#include <string>

namespace AeroSim {
namespace Cfd {

bool cuda_check(cudaError_t error, const char* action, std::string* message = nullptr);

} // namespace Cfd
} // namespace AeroSim
