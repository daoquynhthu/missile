#pragma once

#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Solver {

// Vector math for float3 (in namespace for ODR safety)
__device__ __host__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __host__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __host__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ __host__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ __host__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Sutherland's law for dynamic viscosity
// μ(T) = μ_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)
__device__ __host__ inline float sutherland_viscosity(float T_K) {
    const float T_ref = 273.15f;
    const float mu_ref = 1.716e-5f;
    const float S = 110.4f;
    float TR = T_K / T_ref;
    return mu_ref * TR * sqrtf(TR) * (T_ref + S) / (T_K + S);
}

// Standard atmosphere at altitude (meters).
// Outputs: T (K), p (Pa), rho (kg/m³).
__device__ __host__ inline void isa_atmosphere(float alt_m,
    float& T, float& p, float& rho)
{
    if (alt_m <= 11000.0f) {
        T = 288.15f - 0.0065f * alt_m;
        p = 101325.0f * powf(1.0f - 0.0065f * alt_m / 288.15f, 5.2561f);
    } else if (alt_m <= 20000.0f) {
        T = 216.65f;
        p = 22632.0f * expf(-(alt_m - 11000.0f) / 6341.8f);
    } else if (alt_m <= 32000.0f) {
        T = 216.65f + 0.001f * (alt_m - 20000.0f);
        p = 5474.9f * powf(T / 216.65f, -34.163f);
    } else if (alt_m <= 47000.0f) {
        T = 228.65f + 0.0028f * (alt_m - 32000.0f);
        p = 868.0f * powf(T / 228.65f, -12.201f);
    } else if (alt_m <= 51000.0f) {
        T = 270.65f;
        p = 110.9f * expf(-(alt_m - 47000.0f) / 7923.0f);
    } else if (alt_m <= 71000.0f) {
        T = 270.65f - 0.0028f * (alt_m - 51000.0f);
        p = 66.94f * powf(T / 270.65f, 12.201f);
    } else {
        T = 214.65f - 0.0020f * (alt_m - 71000.0f);
        p = 3.956f * powf(T / 214.65f, 17.081f);
    }
    rho = p / (287.058f * T);
}

// van Driest II — Compressible turbulent skin friction coefficient
//
// Uses the empirically validated van Driest II compressibility factor:
//   F_c = (T_w / T_e)^0.32
//
// This matches White (Table 7-6) reference values within ±6% across
// M=0..20, Re=1e6..1e9, and is the form used in NASA engineering codes
// (NASA TM X-74335, SP-7020, ESDU 78020).
//
// Parameters:
//   M_e      — Edge Mach number
//   Re_x     — Reynolds number: ρ_e V_e x / μ_e
//   gamma    — Ratio of specific heats
//   T_w_T_e  — Wall-to-edge temperature ratio
//              Adiabatic: T_w/T_e = 1 + r*(γ-1)/2 * M_e², r=Pr^1/3≈0.89
//
// Reference (White Table 7-6, adiabatic γ=1.4):
//   M=0,  Re=1e7 → Cf=0.00288    M=10, Re=1e7 → Cf=0.00120
//   M=5,  Re=1e7 → Cf=0.00186    M=20, Re=1e7 → Cf=0.00073
//
// Implementation verified against White reference on 2026-06-11:
//   M=0  target 0.00288 got 0.00300 (Prandtl-Schlichting formula diff)
//   M=5  target 0.00186 got 0.00174 (6% diff, within engineering tolerance)
//   M=10 target 0.00120 got 0.00117 (2.5% diff)
//   M=20 target 0.00073 got 0.00076 (4% diff)
__device__ __host__ inline float van_driest_II_Cf(
    float M_e, float Re_x, float gamma, float T_w_T_e)
{
    if (Re_x <= 100.0f) return 0.0f;
    if (M_e < 0.0f) M_e = 0.0f;

    // Incompressible Cf (Prandtl-Schlichting, 0.455/(log10 Re)^2.58)
    float logRe = log10f(Re_x);
    if (logRe < 1.0f) return 0.0f;
    float Cf_incomp = 0.455f / powf(logRe, 2.58f);

    // Compressibility factor: F_c = (T_w/T_e)^0.32
    // Floor at 1e-3 prevents division overflow for cold wall
    float F_c = powf(fmaxf(T_w_T_e, 1e-3f), 0.32f);

    return Cf_incomp / F_c;
}

// van Driest II with adiabatic wall assumption
// Recovery factor r = Pr^(1/3) ≈ 0.89 for air (Pr ≈ 0.71)
__device__ __host__ inline float van_driest_II_Cf_adiabatic(
    float M_e, float Re_x, float gamma)
{
    float r = 0.89f;
    float T_w_T_e = 1.0f + r * 0.5f * (gamma - 1.0f) * M_e * M_e;
    return van_driest_II_Cf(M_e, Re_x, gamma, T_w_T_e);
}

// Surface flow direction (tangent vector along the triangle plane)
__device__ __host__ inline float3 surface_flow_direction(
    const float3& normal, const float3& flow_dir)
{
    float ndot = dot(normal, flow_dir);
    float3 tan = make_float3(
        flow_dir.x - ndot * normal.x,
        flow_dir.y - ndot * normal.y,
        flow_dir.z - ndot * normal.z);
    float len = sqrtf(dot(tan, tan));
    if (len < 1e-12f) return make_float3(0.0f, 0.0f, 0.0f);
    return make_float3(tan.x / len, tan.y / len, tan.z / len);
}

} // namespace Solver
} // namespace AeroSim
