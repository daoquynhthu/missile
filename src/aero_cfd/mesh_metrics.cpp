#include "aero_cfd/cfd_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace AeroSim {
namespace Cfd {

namespace {

struct Vec3 {
    float x;
    float y;
    float z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

Vec3 to_vec(const CfdNode& n) { return {n.x, n.y, n.z}; }

float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

float norm(Vec3 a) { return std::sqrt(dot(a, a)); }

struct FaceKey {
    int v[3];

    FaceKey(int a, int b, int c) {
        v[0] = a;
        v[1] = b;
        v[2] = c;
        if (v[0] > v[1]) std::swap(v[0], v[1]);
        if (v[1] > v[2]) std::swap(v[1], v[2]);
        if (v[0] > v[1]) std::swap(v[0], v[1]);
    }

    bool operator==(const FaceKey& other) const {
        return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
    }
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& key) const {
        std::uint64_t a = static_cast<std::uint64_t>(key.v[0]);
        std::uint64_t b = static_cast<std::uint64_t>(key.v[1]);
        std::uint64_t c = static_cast<std::uint64_t>(key.v[2]);
        return static_cast<std::size_t>(a ^ (b << 21) ^ (c << 42));
    }
};

struct PendingFace {
    int cell;
    int local_face;
};

std::array<int, 3> cell_face_nodes(const CfdCell& cell, int local_face) {
    switch (local_face) {
        case 0: return {cell.node[1], cell.node[2], cell.node[3]};
        case 1: return {cell.node[0], cell.node[3], cell.node[2]};
        case 2: return {cell.node[0], cell.node[1], cell.node[3]};
        default: return {cell.node[0], cell.node[2], cell.node[1]};
    }
}

int grid_index(int i, int j, int k, int n) {
    return i + j*n + k*n*n;
}

int hex_index(int i, int j, int k, int n_hex) {
    return i + j*n_hex + k*n_hex*n_hex;
}

void add_tet(CfdMesh& mesh, int a, int b, int c, int d) {
    CfdCell cell;
    cell.node[0] = a;
    cell.node[1] = b;
    cell.node[2] = c;
    cell.node[3] = d;
    mesh.cells.push_back(cell);
}

void add_hex_tets(CfdMesh& mesh, int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7) {
    add_tet(mesh, p0, p1, p2, p4);
    add_tet(mesh, p1, p2, p4, p5);
    add_tet(mesh, p2, p4, p5, p6);
    add_tet(mesh, p0, p2, p3, p4);
    add_tet(mesh, p2, p3, p4, p6);
    add_tet(mesh, p3, p4, p6, p7);
}

BoundaryKind classify_cube_boundary(Vec3 fc, Vec3 outward, float outer_scale, float delta, const std::vector<bool>& is_body, int n_hex) {
    const float outer_tol = delta * 0.25f;
    if (std::fabs(std::fabs(fc.x) - outer_scale) < outer_tol ||
        std::fabs(std::fabs(fc.y) - outer_scale) < outer_tol ||
        std::fabs(std::fabs(fc.z) - outer_scale) < outer_tol) {
        return BoundaryKind::Farfield;
    }

    int di = (outward.x > 0.25f) ? 1 : ((outward.x < -0.25f) ? -1 : 0);
    int dj = (outward.y > 0.25f) ? 1 : ((outward.y < -0.25f) ? -1 : 0);
    int dk = (outward.z > 0.25f) ? 1 : ((outward.z < -0.25f) ? -1 : 0);

    float x0 = -outer_scale;
    int hi = static_cast<int>(std::floor((fc.x - x0) / delta));
    int hj = static_cast<int>(std::floor((fc.y - x0) / delta));
    int hk = static_cast<int>(std::floor((fc.z - x0) / delta));
    hi = std::max(0, std::min(n_hex - 1, hi));
    hj = std::max(0, std::min(n_hex - 1, hj));
    hk = std::max(0, std::min(n_hex - 1, hk));

    int bi = hi + di;
    int bj = hj + dj;
    int bk = hk + dk;
    if (bi >= 0 && bi < n_hex && bj >= 0 && bj < n_hex && bk >= 0 && bk < n_hex) {
        if (is_body[hex_index(bi, bj, bk, n_hex)]) return BoundaryKind::SlipWall;
    }

    return BoundaryKind::Farfield;
}

void rebuild_faces(CfdMesh& mesh, bool classify_cube, float outer_scale, float delta, const std::vector<bool>& is_body, int n_hex) {
    mesh.faces.clear();
    for (auto& cell : mesh.cells) {
        cell.first_face = 0;
        cell.face_count = 0;
    }

    std::unordered_map<FaceKey, PendingFace, FaceKeyHash> face_map;
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        for (int lf = 0; lf < 4; ++lf) {
            auto nodes = cell_face_nodes(mesh.cells[ci], lf);
            FaceKey key(nodes[0], nodes[1], nodes[2]);
            auto it = face_map.find(key);
            if (it == face_map.end()) {
                face_map[key] = {ci, lf};
            } else {
                CfdFace face;
                face.left_cell = it->second.cell;
                face.right_cell = ci;
                face.boundary = BoundaryKind::Interior;
                auto left_nodes = cell_face_nodes(mesh.cells[face.left_cell], it->second.local_face);
                face.node[0] = left_nodes[0];
                face.node[1] = left_nodes[1];
                face.node[2] = left_nodes[2];
                mesh.faces.push_back(face);
                face_map.erase(it);
            }
        }
    }

    for (const auto& item : face_map) {
        CfdFace face;
        face.left_cell = item.second.cell;
        face.right_cell = -1;
        auto nodes = cell_face_nodes(mesh.cells[face.left_cell], item.second.local_face);
        face.node[0] = nodes[0];
        face.node[1] = nodes[1];
        face.node[2] = nodes[2];
        face.boundary = BoundaryKind::Farfield;
        mesh.faces.push_back(face);
    }

    compute_mesh_metrics(mesh);

    for (auto& face : mesh.faces) {
        if (face.boundary == BoundaryKind::Interior) continue;
        Vec3 fc{face.cx, face.cy, face.cz};
        Vec3 n{face.nx, face.ny, face.nz};
        if (classify_cube) {
            face.boundary = classify_cube_boundary(fc, n, outer_scale, delta, is_body, n_hex);
        } else if (std::fabs(fc.z) < 1e-7f) {
            face.boundary = BoundaryKind::NoSlipWall;
        } else {
            face.boundary = BoundaryKind::Farfield;
        }
    }

    compute_mesh_metrics(mesh);
}

} // namespace

CfdMesh generate_structured_cube_mesh(float outer_scale, int n_nodes_per_dim) {
    CfdMesh mesh;
    int n = n_nodes_per_dim;
    if (outer_scale <= 1.0f || n < 3) return mesh;

    float delta = 2.0f * outer_scale / static_cast<float>(n - 1);
    mesh.nodes.reserve(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                mesh.nodes.push_back({
                    -outer_scale + i * delta,
                    -outer_scale + j * delta,
                    -outer_scale + k * delta
                });
            }
        }
    }

    int n_hex = n - 1;
    std::vector<bool> is_body(static_cast<std::size_t>(n_hex) * n_hex * n_hex, false);
    for (int k = 0; k < n_hex; ++k) {
        for (int j = 0; j < n_hex; ++j) {
            for (int i = 0; i < n_hex; ++i) {
                float cx = -outer_scale + (i + 0.5f) * delta;
                float cy = -outer_scale + (j + 0.5f) * delta;
                float cz = -outer_scale + (k + 0.5f) * delta;
                bool body = std::fabs(cx) < 1.0f + 1e-6f &&
                            std::fabs(cy) < 1.0f + 1e-6f &&
                            std::fabs(cz) < 1.0f + 1e-6f;
                is_body[hex_index(i, j, k, n_hex)] = body;
            }
        }
    }

    for (int k = 0; k < n_hex; ++k) {
        for (int j = 0; j < n_hex; ++j) {
            for (int i = 0; i < n_hex; ++i) {
                if (is_body[hex_index(i, j, k, n_hex)]) continue;
                int p0 = grid_index(i,   j,   k, n);
                int p1 = grid_index(i+1, j,   k, n);
                int p2 = grid_index(i+1, j+1, k, n);
                int p3 = grid_index(i,   j+1, k, n);
                int p4 = grid_index(i,   j,   k+1, n);
                int p5 = grid_index(i+1, j,   k+1, n);
                int p6 = grid_index(i+1, j+1, k+1, n);
                int p7 = grid_index(i,   j+1, k+1, n);
                add_hex_tets(mesh, p0, p1, p2, p3, p4, p5, p6, p7);
            }
        }
    }

    rebuild_faces(mesh, true, outer_scale, delta, is_body, n_hex);
    return mesh;
}

CfdMesh generate_flat_plate_mesh(float length, float width, float height, float first_height, float growth_ratio, int nx, int ny, int nz) {
    CfdMesh mesh;
    if (length <= 0.0f || width <= 0.0f || height <= 0.0f || first_height <= 0.0f ||
        nx < 2 || ny < 2 || nz < 2) {
        return mesh;
    }

    std::vector<float> xs(nx);
    std::vector<float> ys(ny);
    std::vector<float> zs(nz);
    for (int i = 0; i < nx; ++i) xs[i] = length * static_cast<float>(i) / static_cast<float>(nx - 1);
    for (int j = 0; j < ny; ++j) ys[j] = -0.5f * width + width * static_cast<float>(j) / static_cast<float>(ny - 1);
    zs[0] = 0.0f;
    for (int k = 1; k < nz - 1; ++k) {
        if (growth_ratio > 1.001f) {
            zs[k] = first_height * (std::pow(growth_ratio, static_cast<float>(k)) - 1.0f) / (growth_ratio - 1.0f);
        } else {
            zs[k] = first_height * static_cast<float>(k);
        }
    }
    zs[nz - 1] = height;
    if (nz > 2 && zs[nz - 2] >= height) return CfdMesh{};

    mesh.nodes.reserve(static_cast<std::size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                mesh.nodes.push_back({xs[i], ys[j], zs[k]});
            }
        }
    }

    auto idx = [nx, ny](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                int p0 = idx(i,   j,   k);
                int p1 = idx(i+1, j,   k);
                int p2 = idx(i+1, j+1, k);
                int p3 = idx(i,   j+1, k);
                int p4 = idx(i,   j,   k+1);
                int p5 = idx(i+1, j,   k+1);
                int p6 = idx(i+1, j+1, k+1);
                int p7 = idx(i,   j+1, k+1);
                add_hex_tets(mesh, p0, p1, p2, p3, p4, p5, p6, p7);
            }
        }
    }

    std::vector<bool> empty;
    rebuild_faces(mesh, false, 0.0f, 0.0f, empty, 0);
    return mesh;
}

MeshQualityReport compute_mesh_metrics(CfdMesh& mesh) {
    MeshQualityReport report;
    report.nodes = static_cast<int>(mesh.nodes.size());
    report.cells = static_cast<int>(mesh.cells.size());
    report.faces = static_cast<int>(mesh.faces.size());
    report.min_volume = std::numeric_limits<float>::max();
    report.max_volume = 0.0f;
    report.min_h = std::numeric_limits<float>::max();
    report.max_h = 0.0f;
    report.min_wall_distance = std::numeric_limits<float>::max();

    for (auto& cell : mesh.cells) {
        Vec3 v[4] = {
            to_vec(mesh.nodes[cell.node[0]]),
            to_vec(mesh.nodes[cell.node[1]]),
            to_vec(mesh.nodes[cell.node[2]]),
            to_vec(mesh.nodes[cell.node[3]])
        };
        Vec3 e1 = v[1] - v[0];
        Vec3 e2 = v[2] - v[0];
        Vec3 e3 = v[3] - v[0];
        float det = dot(e1, cross(e2, e3));
        cell.volume = std::fabs(det) / 6.0f;
        cell.cx = 0.25f * (v[0].x + v[1].x + v[2].x + v[3].x);
        cell.cy = 0.25f * (v[0].y + v[1].y + v[2].y + v[3].y);
        cell.cz = 0.25f * (v[0].z + v[1].z + v[2].z + v[3].z);

        float max_area = 0.0f;
        for (int lf = 0; lf < 4; ++lf) {
            auto fn = cell_face_nodes(cell, lf);
            Vec3 a = to_vec(mesh.nodes[fn[0]]);
            Vec3 b = to_vec(mesh.nodes[fn[1]]);
            Vec3 c = to_vec(mesh.nodes[fn[2]]);
            float area = 0.5f * norm(cross(b - a, c - a));
            max_area = std::max(max_area, area);
        }
        cell.h_min = 3.0f * cell.volume / std::max(max_area, 1e-30f);
        cell.wall_distance = 0.0f;

        report.min_volume = std::min(report.min_volume, cell.volume);
        report.max_volume = std::max(report.max_volume, cell.volume);
        report.min_h = std::min(report.min_h, cell.h_min);
        report.max_h = std::max(report.max_h, cell.h_min);
    }

    std::vector<std::vector<int>> cell_faces(mesh.cells.size());
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        CfdFace& face = mesh.faces[fi];
        Vec3 a = to_vec(mesh.nodes[face.node[0]]);
        Vec3 b = to_vec(mesh.nodes[face.node[1]]);
        Vec3 c = to_vec(mesh.nodes[face.node[2]]);
        Vec3 nf = cross(b - a, c - a);
        float nlen = norm(nf);
        face.area = 0.5f * nlen;
        face.cx = (a.x + b.x + c.x) / 3.0f;
        face.cy = (a.y + b.y + c.y) / 3.0f;
        face.cz = (a.z + b.z + c.z) / 3.0f;
        if (nlen > 0.0f) {
            face.nx = nf.x / nlen;
            face.ny = nf.y / nlen;
            face.nz = nf.z / nlen;
        }
        Vec3 fc{face.cx, face.cy, face.cz};
        const CfdCell& left = mesh.cells[face.left_cell];
        Vec3 lc{left.cx, left.cy, left.cz};
        if (dot(fc - lc, {face.nx, face.ny, face.nz}) < 0.0f) {
            face.nx = -face.nx;
            face.ny = -face.ny;
            face.nz = -face.nz;
        }
        if (face.left_cell >= 0) cell_faces[face.left_cell].push_back(fi);
        if (face.right_cell >= 0) cell_faces[face.right_cell].push_back(fi);

        switch (face.boundary) {
            case BoundaryKind::Interior: report.interior_faces++; break;
            case BoundaryKind::Farfield: report.farfield_faces++; break;
            case BoundaryKind::SlipWall: report.slip_wall_faces++; break;
            case BoundaryKind::NoSlipWall: report.no_slip_wall_faces++; break;
            case BoundaryKind::Symmetry: report.symmetry_faces++; break;
        }
    }

    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        auto& cell = mesh.cells[ci];
        cell.first_face = cell_faces[ci].empty() ? 0 : cell_faces[ci][0];
        cell.face_count = static_cast<int>(cell_faces[ci].size());
        float min_wall = std::numeric_limits<float>::max();
        Vec3 cc{cell.cx, cell.cy, cell.cz};
        for (int fi : cell_faces[ci]) {
            const CfdFace& face = mesh.faces[fi];
            if (face.boundary != BoundaryKind::SlipWall && face.boundary != BoundaryKind::NoSlipWall) continue;
            Vec3 fc{face.cx, face.cy, face.cz};
            Vec3 n{face.nx, face.ny, face.nz};
            float distance = std::fabs(dot(cc - fc, n));
            min_wall = std::min(min_wall, distance);
        }
        if (min_wall < std::numeric_limits<float>::max()) {
            cell.wall_distance = min_wall;
            report.min_wall_distance = std::min(report.min_wall_distance, min_wall);
        }
    }

    if (report.cells == 0) {
        report.min_volume = 0.0f;
        report.min_h = 0.0f;
    }
    if (report.min_wall_distance == std::numeric_limits<float>::max()) report.min_wall_distance = 0.0f;

    report.valid = validate_mesh(mesh, &report);
    return report;
}

bool validate_mesh(const CfdMesh& mesh, MeshQualityReport* report) {
    if (mesh.nodes.empty() || mesh.cells.empty() || mesh.faces.empty()) {
        if (report) {
            report->valid = false;
            report->message = "mesh is empty";
        }
        return false;
    }

    for (const auto& cell : mesh.cells) {
        if (cell.volume <= 0.0f || !std::isfinite(cell.volume)) {
            if (report) {
                report->valid = false;
                report->message = "cell has non-positive volume";
            }
            return false;
        }
        if (cell.h_min <= 0.0f || !std::isfinite(cell.h_min)) {
            if (report) {
                report->valid = false;
                report->message = "cell has invalid h_min";
            }
            return false;
        }
    }

    for (const auto& face : mesh.faces) {
        if (face.area <= 0.0f || !std::isfinite(face.area)) {
            if (report) {
                report->valid = false;
                report->message = "face has invalid area";
            }
            return false;
        }
        if (face.boundary == BoundaryKind::Interior && face.right_cell < 0) {
            if (report) {
                report->valid = false;
                report->message = "interior face has no right cell";
            }
            return false;
        }
        if (face.boundary != BoundaryKind::Interior && face.right_cell >= 0) {
            if (report) {
                report->valid = false;
                report->message = "boundary face has right cell";
            }
            return false;
        }
    }

    if (report) {
        report->valid = true;
        report->message.clear();
    }
    return true;
}

float boundary_area(const CfdMesh& mesh, BoundaryKind boundary) {
    float area = 0.0f;
    for (const auto& face : mesh.faces) {
        if (face.boundary == boundary) area += face.area;
    }
    return area;
}

} // namespace Cfd
} // namespace AeroSim
