#include "aero/cfd/cuda_utils.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

bool cuda_check(cudaError_t error, const char* action, std::string* message) {
    if (error == cudaSuccess) return true;
    if (message) {
        *message = action ? action : "cuda call";
        *message += ": ";
        *message += cudaGetErrorString(error);
    }
    return false;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
