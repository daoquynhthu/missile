#pragma once

#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"

#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

CfdSolveSummary solve_gpu_dispatch(
    const CfdMesh& mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
