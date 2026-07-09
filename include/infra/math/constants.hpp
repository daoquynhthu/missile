#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cmath>
#include "infra/common.hpp"

namespace AeroSim {
    namespace Earth {
        inline CUDA_HOST_DEVICE constexpr double A() { return 6378137.0; }
        inline CUDA_HOST_DEVICE constexpr double F() { return 1.0 / 298.257223563; }
        inline CUDA_HOST_DEVICE constexpr double B() { return 6356752.314245; }
        inline CUDA_HOST_DEVICE constexpr double E2() { return 0.00669437999014; }
        inline CUDA_HOST_DEVICE constexpr double E_PRIME2() { return 0.00673949674228; }
        inline CUDA_HOST_DEVICE constexpr double MU() { return 3.986004418e14; }
        inline CUDA_HOST_DEVICE constexpr double OMEGA() { return 7.292115e-5; }
        inline CUDA_HOST_DEVICE constexpr double J2() { return 1.08262668355e-3; }
        inline CUDA_HOST_DEVICE constexpr double G() { return 9.80665; }
    }
    namespace Math {
        inline CUDA_HOST_DEVICE constexpr double PI() { return 3.14159265358979323846; }
        inline CUDA_HOST_DEVICE constexpr double DEG2RAD() { return PI() / 180.0; }
        inline CUDA_HOST_DEVICE constexpr double RAD2DEG() { return 180.0 / PI(); }
    }
    namespace Atmosphere {
        inline CUDA_HOST_DEVICE constexpr double R_GAS() { return 8.314462618; }
        inline CUDA_HOST_DEVICE constexpr double M_AIR() { return 0.0289644; }
    }
}

#endif
