#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.hpp"

namespace AeroSim {

    struct Vec3 {
        double x, y, z;

        CUDA_HOST_DEVICE Vec3() : x(0), y(0), z(0) {}
        CUDA_HOST_DEVICE Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

        CUDA_HOST_DEVICE Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
        CUDA_HOST_DEVICE Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
        CUDA_HOST_DEVICE Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
        CUDA_HOST_DEVICE Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
        
        CUDA_HOST_DEVICE double norm() const { return sqrt(x * x + y * y + z * z); }
        CUDA_HOST_DEVICE double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    };

}

#endif // CUDA_UTILS_CUH
