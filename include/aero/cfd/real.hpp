#pragma once

#include <cmath>
#include <cuda_runtime.h>

namespace AeroSim {

#ifdef __CUDACC__
#define AEROSIM_REAL_DEVICE static __device__
#define AEROSIM_REAL_HOST_DEVICE static __device__ __host__ inline
#else
#define AEROSIM_REAL_DEVICE static inline
#define AEROSIM_REAL_HOST_DEVICE static inline
#endif

#ifdef AEROSIM_REAL_DOUBLE
    using Real = double;

    AEROSIM_REAL_HOST_DEVICE Real real_sqrt(Real x) { return sqrt(x); }
    AEROSIM_REAL_HOST_DEVICE Real real_fabs(Real x) { return fabs(x); }
    AEROSIM_REAL_HOST_DEVICE Real real_fmin(Real x, Real y) { return fmin(x, y); }
    AEROSIM_REAL_HOST_DEVICE Real real_fmax(Real x, Real y) { return fmax(x, y); }
    AEROSIM_REAL_HOST_DEVICE bool real_isfinite(Real x) {
#ifdef __CUDA_ARCH__
        return static_cast<bool>(isfinite(x));
#else
        return static_cast<bool>(std::isfinite(x));
#endif
    }
    AEROSIM_REAL_HOST_DEVICE Real real_copysign(Real x, Real y) { return copysign(x, y); }
    AEROSIM_REAL_HOST_DEVICE Real real_cos(Real x) { return cos(x); }
    AEROSIM_REAL_HOST_DEVICE Real real_sin(Real x) { return sin(x); }

#ifdef __CUDACC__
    AEROSIM_REAL_DEVICE Real real_atomic_add(Real* addr, Real val) {
        return atomicAdd(addr, val);
    }

    AEROSIM_REAL_DEVICE Real real_atomic_min(Real* addr, Real val) {
        unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
        unsigned long long old = *addr_as_ull;
        unsigned long long assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_ull, assumed,
                __double_as_longlong(real_fmin(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    AEROSIM_REAL_DEVICE Real real_atomic_max(Real* addr, Real val) {
        unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
        unsigned long long old = *addr_as_ull;
        unsigned long long assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_ull, assumed,
                __double_as_longlong(real_fmax(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif
#else
    using Real = float;

    AEROSIM_REAL_HOST_DEVICE Real real_sqrt(Real x) { return sqrtf(x); }
    AEROSIM_REAL_HOST_DEVICE Real real_fabs(Real x) { return fabsf(x); }
    AEROSIM_REAL_HOST_DEVICE Real real_fmin(Real x, Real y) { return fminf(x, y); }
    AEROSIM_REAL_HOST_DEVICE Real real_fmax(Real x, Real y) { return fmaxf(x, y); }
    AEROSIM_REAL_HOST_DEVICE bool real_isfinite(Real x) {
#ifdef __CUDA_ARCH__
        return static_cast<bool>(__finitef(x));
#else
        return static_cast<bool>(std::isfinite(x));
#endif
    }
    AEROSIM_REAL_HOST_DEVICE Real real_copysign(Real x, Real y) { return copysignf(x, y); }
    AEROSIM_REAL_HOST_DEVICE Real real_cos(Real x) { return cosf(x); }
    AEROSIM_REAL_HOST_DEVICE Real real_sin(Real x) { return sinf(x); }

#ifdef __CUDACC__
    AEROSIM_REAL_DEVICE Real real_atomic_add(Real* addr, Real val) {
        return atomicAdd(addr, val);
    }

    AEROSIM_REAL_DEVICE Real real_atomic_min(Real* addr, Real val) {
        unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr);
        unsigned int old = *addr_as_int;
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_int, assumed,
                __float_as_int(real_fmin(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    AEROSIM_REAL_DEVICE Real real_atomic_max(Real* addr, Real val) {
        unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr);
        unsigned int old = *addr_as_int;
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_int, assumed,
                __float_as_int(real_fmax(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
#endif
#endif

} // namespace AeroSim
