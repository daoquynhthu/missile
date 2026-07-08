#include "aero_cfd/real.hpp"
#pragma once

#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_result.hpp"
#include "aero_cfd/cfd_solver.hpp"
#include "aero_cfd/cfd_state.hpp"
#include "aero_cfd/device_mesh.hpp"

#include <string>
#include <vector>

namespace AeroSim {
namespace Cfd {

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error = nullptr);

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    Real* d_min_dt,
    Real* d_l2_sum,
    Real* d_forces,
    std::string* error = nullptr);

} // namespace Cfd
} // namespace AeroSim


