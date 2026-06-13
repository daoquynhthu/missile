#include "aero_solver/mesh_generator.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cfloat>

namespace AeroSim {
namespace Solver {

// ─── Geometric predicates ─────────────────────────────────────────────────

// Orientation test for 4 points in 3D. Returns >0 if d is above plane(a,b,c).
static double orient3d(const double* a, const double* b,
                       const double* c, const double* d) {
    double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
    double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
    double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
    return adx * (bdy * cdz - bdz * cdy)
         - ady * (bdx * cdz - bdz * cdx)
         + adz * (bdx * cdy - bdy * cdx);
}

// Insphere test for 5 points. Returns >0 if e is inside circumsphere of a,b,c,d.
// Uses 4x4 determinant: | (a-e) (a-e)^2 |  = det of matrix with rows [dx,dy,dz,dx^2+dy^2+dz^2]
//                       | (b-e) (b-e)^2 |    for each of a,b,c,d relative to e.
//                       | (c-e) (c-e)^2 |
//                       | (d-e) (d-e)^2 |
static double insphere(const double* a, const double* b,
                       const double* c, const double* d, const double* e) {
    double adx = a[0] - e[0], ady = a[1] - e[1], adz = a[2] - e[2], a2 = adx*adx + ady*ady + adz*adz;
    double bdx = b[0] - e[0], bdy = b[1] - e[1], bdz = b[2] - e[2], b2 = bdx*bdx + bdy*bdy + bdz*bdz;
    double cdx = c[0] - e[0], cdy = c[1] - e[1], cdz = c[2] - e[2], c2 = cdx*cdx + cdy*cdy + cdz*cdz;
    double ddx = d[0] - e[0], ddy = d[1] - e[1], ddz = d[2] - e[2], d2 = ddx*ddx + ddy*ddy + ddz*ddz;

    // Laplace expansion on first row: adx*C11 - ady*C12 + adz*C13 - a2*C14
    double C11 = bdy*(cdz*d2 - c2*ddz) - bdz*(cdy*d2 - c2*ddy) + b2*(cdy*ddz - cdz*ddy);
    double C12 = bdx*(cdz*d2 - c2*ddz) - bdz*(cdx*d2 - c2*ddx) + b2*(cdx*ddz - cdz*ddx);
    double C13 = bdx*(cdy*d2 - c2*ddy) - bdy*(cdx*d2 - c2*ddx) + b2*(cdx*ddy - cdy*ddx);
    double C14 = bdx*(cdy*ddz - cdz*ddy) - bdy*(cdx*ddz - cdz*ddx) + bdz*(cdx*ddy - cdy*ddx);

    return adx*C11 - ady*C12 + adz*C13 - a2*C14;
}

// ─── Face key: uniquely identify a triangular face by sorted vertex indices ──
struct FaceKey {
    int v[3];
    FaceKey(int a, int b, int c) {
        v[0] = a; v[1] = b; v[2] = c;
        if (v[0] > v[1]) std::swap(v[0], v[1]);
        if (v[1] > v[2]) std::swap(v[1], v[2]);
        if (v[0] > v[1]) std::swap(v[0], v[1]);
    }
    bool operator==(const FaceKey& o) const {
        return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2];
    }
};
struct FaceKeyHash {
    size_t operator()(const FaceKey& k) const {
        return static_cast<size_t>(k.v[0]) ^ (static_cast<size_t>(k.v[1]) << 10)
             ^ (static_cast<size_t>(k.v[2]) << 20);
    }
};

// Get the 3 vertex indices for a given face of a tetrahedron.
// Face f (0..3) is opposite vertex f.
static void get_tet_face(int4 tet, int f, int& a, int& b, int& c) {
    switch (f) {
        case 0: a = tet.y; b = tet.z; c = tet.w; break;
        case 1: a = tet.x; b = tet.z; c = tet.w; break;
        case 2: a = tet.x; b = tet.y; c = tet.w; break;
        case 3: a = tet.x; b = tet.y; c = tet.z; break;
    }
}

// ─── Core Bowyer-Watson Delaunay tetrahedralization ───────────────────────

struct DelaunayMesh {
    std::vector<double> pts; // flat array: [x0,y0,z0, x1,y1,z1, ...]
    std::vector<int4> tets;  // tetrahedra
    int np;                  // number of points

    DelaunayMesh() : np(0) {}

    int add_point(double x, double y, double z) {
        pts.push_back(x); pts.push_back(y); pts.push_back(z);
        return np++;
    }

    const double* point(int i) const { return &pts[i * 3]; }

    // Build the insphere-removed cavity and create new tets
    void insert_point(int pi) {
        const double* p = point(pi);
        std::vector<int> bad_tets;
        std::vector<bool> is_bad(tets.size(), false);

        // Find all bad tets whose circumsphere contains p.
        // The insphere sign depends on tet orientation:
        //   orient3d(tet) * insphere(tet, p) > 0  means p is inside.
        // We compute orientation from the tet vertices.
        for (size_t i = 0; i < tets.size(); ++i) {
            int4 t = tets[i];
            if (t.x < 0 || t.y < 0 || t.z < 0 || t.w < 0) continue;
            double s = insphere(point(t.x), point(t.y), point(t.z), point(t.w), p);
            double o = orient3d(point(t.x), point(t.y), point(t.z), point(t.w));
            if (o * s > 1e-12) {
                bad_tets.push_back(static_cast<int>(i));
                is_bad[i] = true;
            }
        }

        if (bad_tets.empty()) return;

        // Extract cavity boundary faces (faces shared by exactly one bad tet)
        std::unordered_map<FaceKey, int, FaceKeyHash> face_count;
        for (int ti : bad_tets) {
            int4 t = tets[ti];
            for (int f = 0; f < 4; ++f) {
                int a, b, c;
                get_tet_face(t, f, a, b, c);
                FaceKey fk(a, b, c);
                auto it = face_count.find(fk);
                if (it == face_count.end()) {
                    face_count[fk] = 1;
                } else {
                    it->second++;
                }
            }
        }

        struct BoundaryFace { int a, b, c; };
        std::vector<BoundaryFace> boundary;
        for (auto& fc : face_count) {
            if (fc.second == 1) {
                boundary.push_back({fc.first.v[0], fc.first.v[1], fc.first.v[2]});
            }
        }

        // Remove bad tets (mark as deleted)
        for (int ti : bad_tets) {
            tets[ti].x = -1;
        }

        // Create new tets from boundary faces + new point
        for (auto& bf : boundary) {
            tets.push_back(make_int4(bf.a, bf.b, bf.c, pi));
        }
    }

    // Remove deleted tets and compact the array
    void compact() {
        size_t write = 0;
        for (size_t i = 0; i < tets.size(); ++i) {
            if (tets[i].x >= 0) {
                if (write != i) tets[write] = tets[i];
                write++;
            }
        }
        tets.resize(write);
    }
};

// Build an initial Delaunay mesh from 8 box corner points.
// Decomposes the box into 5 tetrahedra (fan from main diagonal 0→6).
// All 5 tets have positive orientation (det > 0).
static void init_box_mesh(DelaunayMesh& dm,
    double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
{
    int c[8];
    c[0] = dm.add_point(xmin, ymin, zmin);
    c[1] = dm.add_point(xmax, ymin, zmin);
    c[2] = dm.add_point(xmax, ymax, zmin);
    c[3] = dm.add_point(xmin, ymax, zmin);
    c[4] = dm.add_point(xmin, ymin, zmax);
    c[5] = dm.add_point(xmax, ymin, zmax);
    c[6] = dm.add_point(xmax, ymax, zmax);
    c[7] = dm.add_point(xmin, ymax, zmax);

    // 5-tet fan along diagonal (origin → opposite corner)
    dm.tets.push_back(make_int4(c[0], c[1], c[3], c[6]));
    dm.tets.push_back(make_int4(c[0], c[3], c[7], c[6]));
    dm.tets.push_back(make_int4(c[0], c[7], c[4], c[6]));
    dm.tets.push_back(make_int4(c[0], c[4], c[5], c[6]));
    dm.tets.push_back(make_int4(c[0], c[5], c[1], c[6]));
}

// ─── Main mesh generation ─────────────────────────────────────────────────

TetMesh generate_tetrahedral_mesh(
    const std::vector<float3>& surface_nodes,
    const std::vector<int3>& surface_tris,
    float outer_scale,
    int target_tets)
{
    (void)target_tets;
    TetMesh mesh;

    if (surface_nodes.empty() || surface_tris.empty()) return mesh;

    // 1. Compute bounding box of surface, expand by outer_scale for farfield
    double xmin = surface_nodes[0].x, xmax = surface_nodes[0].x;
    double ymin = surface_nodes[0].y, ymax = surface_nodes[0].y;
    double zmin = surface_nodes[0].z, zmax = surface_nodes[0].z;
    for (auto& n : surface_nodes) {
        if (n.x < xmin) xmin = n.x; if (n.x > xmax) xmax = n.x;
        if (n.y < ymin) ymin = n.y; if (n.y > ymax) ymax = n.y;
        if (n.z < zmin) zmin = n.z; if (n.z > zmax) zmax = n.z;
    }
    double dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
    double pad = std::max({dx, dy, dz}) * (outer_scale - 1.0) * 0.5;

    // Outer bounding box (for farfield boundary)
    double oxmin = xmin - pad, oxmax = xmax + pad;
    double oymin = ymin - pad, oymax = ymax + pad;
    double ozmin = zmin - pad, ozmax = zmax + pad;

    DelaunayMesh dm;

    // 2. Build initial mesh from 8 outer box corners
    init_box_mesh(dm, oxmin, oxmax, oymin, oymax, ozmin, ozmax);
    int box_start = 0;

    // 3. Add surface points
    std::vector<int> surf_to_dm(surface_nodes.size());
    for (size_t i = 0; i < surface_nodes.size(); ++i) {
        surf_to_dm[i] = dm.add_point(surface_nodes[i].x, surface_nodes[i].y, surface_nodes[i].z);
    }

    // 4. Insert surface points via Bowyer-Watson
    for (int pi : surf_to_dm) {
        dm.insert_point(pi);
    }

    // 5. Insert interior grid points for volume coverage
    int interior_start = dm.np;
    double spacing = std::max({dx, dy, dz}) / 12.0;
    for (double x = xmin + spacing; x < xmax; x += spacing) {
        for (double y = ymin + spacing; y < ymax; y += spacing) {
            for (double z = zmin + spacing; z < zmax; z += spacing) {
                dm.add_point(x, y, z);
            }
        }
    }
    for (int i = interior_start; i < dm.np; ++i) {
        dm.insert_point(i);
    }

    // 6. Compact (remove deleted tets)
    dm.compact();

    // 7. Build output mesh nodes
    mesh.nodes.reserve(dm.np);
    for (int i = 0; i < dm.np; ++i) {
        const double* p = dm.point(i);
        mesh.nodes.push_back(make_float3(
            static_cast<float>(p[0]),
            static_cast<float>(p[1]),
            static_cast<float>(p[2])));
    }

    // 8. Remove tets inside the body (cube: |x|<1,|y|<1,|z|<1) and tets
    //    connected only to outer box corners (pure farfield region)
    std::vector<bool> keep(dm.tets.size(), true);
    for (size_t i = 0; i < dm.tets.size(); ++i) {
        int4 t = dm.tets[i];
        // Check if all 4 vertices are outer box corners (indices 0..7)
        bool all_box = (t.x >= box_start && t.x < box_start + 8 &&
                        t.y >= box_start && t.y < box_start + 8 &&
                        t.z >= box_start && t.z < box_start + 8 &&
                        t.w >= box_start && t.w < box_start + 8);
        if (all_box) {
            keep[i] = false;
            continue;
        }
        // Compute centroid and check if inside cube (|coord| < 1 + tol)
        auto point = [&](int vi) -> double3 {
            const double* p = dm.point(vi);
            return {p[0], p[1], p[2]};
        };
        int4 tv = dm.tets[i];
        double3 pa = point(tv.x), pb = point(tv.y), pc = point(tv.z), pd = point(tv.w);
        double cx = (pa.x + pb.x + pc.x + pd.x) / 4.0;
        double cy = (pa.y + pb.y + pc.y + pd.y) / 4.0;
        double cz = (pa.z + pb.z + pc.z + pd.z) / 4.0;
        const double tol = 1e-6;
        if (fabs(cx) < 1.0 + tol && fabs(cy) < 1.0 + tol && fabs(cz) < 1.0 + tol) {
            keep[i] = false; // inside body — discard
        }
    }

    int kept_count = 0;
    for (size_t i = 0; i < dm.tets.size(); ++i) {
        if (keep[i]) kept_count++;
    }
    mesh.tets.reserve(kept_count);
    for (size_t i = 0; i < dm.tets.size(); ++i) {
        if (keep[i]) {
            mesh.tets.push_back(dm.tets[i]);
        }
    }

    // 9. Compute face neighbor connectivity
    mesh.tet_neighbors.resize(mesh.tets.size(), make_int4(-1, -1, -1, -2));

    std::unordered_map<FaceKey, std::pair<int, int>, FaceKeyHash> face_to_tet;
    for (size_t ti = 0; ti < mesh.tets.size(); ++ti) {
        int4 t = mesh.tets[ti];
        for (int f = 0; f < 4; ++f) {
            int a, b, c;
            get_tet_face(t, f, a, b, c);
            FaceKey fk(a, b, c);
            auto it = face_to_tet.find(fk);
            if (it == face_to_tet.end()) {
                face_to_tet[fk] = {static_cast<int>(ti), f};
            } else {
                int other_ti = it->second.first;
                int other_f = it->second.second;
                int4* nb = &mesh.tet_neighbors[ti];
                int4* onb = &mesh.tet_neighbors[other_ti];
                switch (f) {
                    case 0: nb->x = other_ti; break;
                    case 1: nb->y = other_ti; break;
                    case 2: nb->z = other_ti; break;
                    case 3: nb->w = other_ti; break;
                }
                switch (other_f) {
                    case 0: onb->x = static_cast<int>(ti); break;
                    case 1: onb->y = static_cast<int>(ti); break;
                    case 2: onb->z = static_cast<int>(ti); break;
                    case 3: onb->w = static_cast<int>(ti); break;
                }
                face_to_tet.erase(it);
            }
        }
    }

    // Mark STL surface faces as wall (-1), remaining boundary faces as farfield (-2)
    // Use geometric detection: a face is a wall if all 3 vertices lie on the body surface
    // (within tolerance of cube coordinates ±1 for the cube test)
    const double wall_tol = 1e-4;

    for (auto& fk_pair : face_to_tet) {
        const FaceKey& fk = fk_pair.first;
        int ti = fk_pair.second.first;
        int face = fk_pair.second.second;

        // Get vertex coordinates for this face
        int4 tet = mesh.tets[ti];
        int verts[3];
        get_tet_face(tet, face, verts[0], verts[1], verts[2]);

        auto on_surface = [&](int vi) -> bool {
            float3 v = mesh.nodes[vi];
            // Cube surface: exactly one coordinate is ±1, the other two are within [-1,1]
            int cnt1 = 0;
            if (fabs(fabs(v.x) - 1.0f) < wall_tol) cnt1++;
            if (fabs(fabs(v.y) - 1.0f) < wall_tol) cnt1++;
            if (fabs(fabs(v.z) - 1.0f) < wall_tol) cnt1++;
            return cnt1 >= 1 && fabs(v.x) <= 1.0f + wall_tol && fabs(v.y) <= 1.0f + wall_tol && fabs(v.z) <= 1.0f + wall_tol;
        };

        bool is_wall = on_surface(verts[0]) && on_surface(verts[1]) && on_surface(verts[2]);

        int4* nb = &mesh.tet_neighbors[ti];
        int btype = is_wall ? -1 : -2;
        switch (face) {
            case 0: nb->x = btype; break;
            case 1: nb->y = btype; break;
            case 2: nb->z = btype; break;
            case 3: nb->w = btype; break;
        }
    }

    return mesh;
}

TetMesh generate_cube_mesh(float outer_scale) {
    float verts[8][3] = {
        {-1,-1,-1}, { 1,-1,-1}, { 1,-1, 1}, {-1,-1, 1},
        {-1, 1,-1}, { 1, 1,-1}, { 1, 1, 1}, {-1, 1, 1}
    };
    int faces[12][3] = {
        {0,1,2}, {0,2,3},
        {4,6,5}, {4,7,6},
        {0,3,7}, {0,7,4},
        {1,5,6}, {1,6,2},
        {0,4,5}, {0,5,1},
        {3,2,6}, {3,6,7}
    };
    std::vector<float3> nodes;
    std::vector<int3> tris;
    for (int i = 0; i < 8; ++i)
        nodes.push_back(make_float3(verts[i][0], verts[i][1], verts[i][2]));
    for (int i = 0; i < 12; ++i)
        tris.push_back(make_int3(faces[i][0], faces[i][1], faces[i][2]));
    return generate_tetrahedral_mesh(nodes, tris, outer_scale, 5000);
}

} // namespace Solver
} // namespace AeroSim
