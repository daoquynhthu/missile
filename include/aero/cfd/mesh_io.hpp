#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

bool read_mesh_su2(const std::string& path, CfdMesh& mesh, std::string* err = nullptr);

bool write_mesh_su2(const CfdMesh& mesh, const std::string& path, std::string* err = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
