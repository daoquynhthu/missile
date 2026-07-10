#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

bool read_mesh_cgns(const std::string& path, CfdMesh& mesh, std::string* err = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
