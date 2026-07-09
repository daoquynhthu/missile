#include "aero/cfd/real.hpp"
#pragma once

#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_result.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/device_mesh.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

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

} // namespace cfd
} // namespace aero
} // namespace aerosp


