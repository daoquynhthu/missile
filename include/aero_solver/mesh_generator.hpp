#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Solver {

struct TetMesh {
    std::vector<float3> nodes;
    // tetrahedra: each int4 = {i, j, k, l} (vertex indices, 0-based)
    std::vector<int4> tets;
    // face neighbors: for each tet, 4 face neighbor indices.
    // -1 = wall boundary, -2 = farfield boundary, >=0 = interior neighbor tet index.
    // Faces are ordered: face 0 = opposite node 0 (nodes 1,2,3), etc.
    std::vector<int4> tet_neighbors;
};

// Generate a tetrahedral volume mesh from an STL surface mesh.
// surface_nodes: unique vertices of the STL (N unique vertices)
// surface_tris: triangles as (i,j,k) indices into surface_nodes
// outer_scale: multiplier on bounding box for farfield boundary (default 10x)
// target_tets: approximate target number of tetrahedra (default 50000)
TetMesh generate_tetrahedral_mesh(
    const std::vector<float3>& surface_nodes,
    const std::vector<int3>& surface_tris,
    float outer_scale = 10.0f,
    int target_tets = 50000);

// Generate a unit cube tetrahedral mesh (centered at origin, extent 2x2x2).
// outer_scale: farfield box half-extent = outer_scale (default 5x cube half-size).
TetMesh generate_cube_mesh(float outer_scale = 5.0f);

} // namespace Solver
} // namespace AeroSim
