#include "aero/cfd/real.hpp"
#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace aerosp {
namespace aero {
namespace cfd {

enum class BoundaryKind : int {
    Interior = 0,
    Farfield = 1,
    SlipWall = 2,
    NoSlipWall = 3,
    Symmetry = 4
};

struct CfdNode {
    Real x = 0.0f;
    Real y = 0.0f;
    Real z = 0.0f;
};

struct CfdCell {
    int node[4] = {-1, -1, -1, -1};
    int first_face = 0;
    int face_count = 0;
    Real volume = 0.0f;
    Real cx = 0.0f;
    Real cy = 0.0f;
    Real cz = 0.0f;
    Real h_min = 0.0f;
    Real wall_distance = 0.0f;
};

struct CfdFace {
    int left_cell = -1;
    int right_cell = -1;
    int node[3] = {-1, -1, -1};
    BoundaryKind boundary = BoundaryKind::Interior;
    Real area = 0.0f;
    Real nx = 0.0f;
    Real ny = 0.0f;
    Real nz = 0.0f;
    Real cx = 0.0f;
    Real cy = 0.0f;
    Real cz = 0.0f;
};

struct CfdMesh {
    std::vector<CfdNode> nodes;
    std::vector<CfdCell> cells;
    std::vector<CfdFace> faces;
};

struct MeshQualityReport {
    int nodes = 0;
    int cells = 0;
    int faces = 0;
    int interior_faces = 0;
    int farfield_faces = 0;
    int slip_wall_faces = 0;
    int no_slip_wall_faces = 0;
    int symmetry_faces = 0;
    Real min_volume = 0.0f;
    Real max_volume = 0.0f;
    Real min_h = 0.0f;
    Real max_h = 0.0f;
    Real min_wall_distance = 0.0f;
    bool valid = false;
    std::string message;
};

CfdMesh generate_structured_cube_mesh(Real outer_scale = 5.0f, int n_nodes_per_dim = 13);

CfdMesh generate_flat_plate_mesh(
    Real length = 0.5f,
    Real width = 0.05f,
    Real height = 0.1f,
    Real first_height = 1e-5f,
    Real growth_ratio = 1.12f,
    int nx = 30,
    int ny = 3,
    int nz = 50);

MeshQualityReport compute_mesh_metrics(CfdMesh& mesh);

bool validate_mesh(const CfdMesh& mesh, MeshQualityReport* report = nullptr);

Real boundary_area(const CfdMesh& mesh, BoundaryKind boundary);

} // namespace cfd
} // namespace aero
} // namespace aerosp



