#include "aero/cfd/cuda_utils.hpp"

namespace AeroSim {
namespace Cfd {

bool cuda_check(cudaError_t error, const char* action, std::string* message) {
    if (error == cudaSuccess) return true;
    if (message) {
        *message = action ? action : "cuda call";
        *message += ": ";
        *message += cudaGetErrorString(error);
    }
    return false;
}

} // namespace Cfd
} // namespace AeroSim
