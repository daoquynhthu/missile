#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace AeroSim {
namespace Cfd {

enum class BoundaryKind : int {
    Interior = 0,
    Farfield = 1,
    SlipWall = 2,
    NoSlipWall = 3,
    Symmetry = 4
};

struct CfdNode {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct CfdCell {
    int node[4] = {-1, -1, -1, -1};
    int first_face = 0;
    int face_count = 0;
    float volume = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float h_min = 0.0f;
    float wall_distance = 0.0f;
};

struct CfdFace {
    int left_cell = -1;
    int right_cell = -1;
    int node[3] = {-1, -1, -1};
    BoundaryKind boundary = BoundaryKind::Interior;
    float area = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
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
    float min_volume = 0.0f;
    float max_volume = 0.0f;
    float min_h = 0.0f;
    float max_h = 0.0f;
    float min_wall_distance = 0.0f;
    bool valid = false;
    std::string message;
};

CfdMesh generate_structured_cube_mesh(float outer_scale = 5.0f, int n_nodes_per_dim = 13);

CfdMesh generate_flat_plate_mesh(
    float length = 0.5f,
    float width = 0.05f,
    float height = 0.1f,
    float first_height = 1e-5f,
    float growth_ratio = 1.12f,
    int nx = 30,
    int ny = 3,
    int nz = 50);

MeshQualityReport compute_mesh_metrics(CfdMesh& mesh);

bool validate_mesh(const CfdMesh& mesh, MeshQualityReport* report = nullptr);

float boundary_area(const CfdMesh& mesh, BoundaryKind boundary);

} // namespace Cfd
} // namespace AeroSim

