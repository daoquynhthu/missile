#pragma once

#include <cmath>

namespace AeroSim {
namespace Cfd {

struct ConservativeState {
    float rho = 0.0f;
    float rho_u = 0.0f;
    float rho_v = 0.0f;
    float rho_w = 0.0f;
    float rho_E = 0.0f;
};

struct PrimitiveState {
    float rho = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 0.0f;
    float p = 0.0f;
};

struct EulerFlux {
    float mass = 0.0f;
    float mom_x = 0.0f;
    float mom_y = 0.0f;
    float mom_z = 0.0f;
    float energy = 0.0f;
};

inline bool is_valid_primitive(const PrimitiveState& w) {
    return std::isfinite(w.rho) && std::isfinite(w.u) && std::isfinite(w.v) &&
           std::isfinite(w.w) && std::isfinite(w.p) &&
           w.rho > 0.0f && w.p > 0.0f;
}

inline ConservativeState primitive_to_conservative(const PrimitiveState& w, float gamma) {
    float kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    ConservativeState q;
    q.rho = w.rho;
    q.rho_u = w.rho * w.u;
    q.rho_v = w.rho * w.v;
    q.rho_w = w.rho * w.w;
    q.rho_E = w.p / (gamma - 1.0f) + w.rho * kinetic;
    return q;
}

inline bool conservative_to_primitive(const ConservativeState& q, float gamma, PrimitiveState& w) {
    if (!std::isfinite(q.rho) || q.rho <= 0.0f) return false;
    float inv_rho = 1.0f / q.rho;
    w.rho = q.rho;
    w.u = q.rho_u * inv_rho;
    w.v = q.rho_v * inv_rho;
    w.w = q.rho_w * inv_rho;
    float kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    w.p = (gamma - 1.0f) * (q.rho_E - q.rho * kinetic);
    return is_valid_primitive(w);
}

inline float speed_of_sound(const PrimitiveState& w, float gamma) {
    return std::sqrt(gamma * w.p / w.rho);
}

inline EulerFlux physical_flux(const PrimitiveState& w, float gamma, float nx, float ny, float nz) {
    float vn = w.u*nx + w.v*ny + w.w*nz;
    float kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    float rho_E = w.p / (gamma - 1.0f) + w.rho * kinetic;

    EulerFlux f;
    f.mass = w.rho * vn;
    f.mom_x = w.rho * w.u * vn + w.p * nx;
    f.mom_y = w.rho * w.v * vn + w.p * ny;
    f.mom_z = w.rho * w.w * vn + w.p * nz;
    f.energy = (rho_E + w.p) * vn;
    return f;
}

inline EulerFlux slip_wall_flux(const PrimitiveState& w, float nx, float ny, float nz) {
    EulerFlux f;
    f.mass = 0.0f;
    f.mom_x = w.p * nx;
    f.mom_y = w.p * ny;
    f.mom_z = w.p * nz;
    f.energy = 0.0f;
    return f;
}

EulerFlux hllc_flux(const PrimitiveState& left, const PrimitiveState& right, float gamma, float nx, float ny, float nz);

PrimitiveState make_freestream(float mach, float alpha_deg, float beta_deg, float gamma);

} // namespace Cfd
} // namespace AeroSim

