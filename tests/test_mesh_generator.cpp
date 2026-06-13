#include "aero_solver/mesh_generator.hpp"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace AeroSim::Solver;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

int main() {
    // Test 1: create a cube surface from 12 triangles
    TEST("Cube tetrahedralization");
    {
        // A cube with vertices at (±1, ±1, ±1), triangulated
        float verts[8][3] = {
            {-1,-1,-1}, { 1,-1,-1}, { 1,-1, 1}, {-1,-1, 1},
            {-1, 1,-1}, { 1, 1,-1}, { 1, 1, 1}, {-1, 1, 1}
        };
        int faces[12][3] = {
            // bottom (z=-1), normal (0,0,-1)
            {0,1,2}, {0,2,3},
            // top (z=1), normal (0,0,1)
            {4,6,5}, {4,7,6},
            // front (y=-1), normal (0,-1,0)
            {0,3,7}, {0,7,4},
            // back (y=1), normal (0,1,0)
            {1,5,6}, {1,6,2},
            // left (x=-1), normal (-1,0,0)
            {0,4,5}, {0,5,1},
            // right (x=1), normal (1,0,0)
            {3,2,6}, {3,6,7}
        };

        std::vector<float3> nodes;
        std::vector<int3> tris;
        for (int i = 0; i < 8; ++i)
            nodes.push_back(make_float3(
                static_cast<float>(verts[i][0]),
                static_cast<float>(verts[i][1]),
                static_cast<float>(verts[i][2])));
        for (int i = 0; i < 12; ++i)
            tris.push_back(make_int3(faces[i][0], faces[i][1], faces[i][2]));

        auto mesh = generate_tetrahedral_mesh(nodes, tris, 5.0f, 5000);
        if (mesh.nodes.size() < 8) { FAIL("too few nodes"); return 1; }
        if (mesh.tets.empty()) { FAIL("no tets generated"); return 1; }
        if (mesh.tet_neighbors.size() != mesh.tets.size()) { FAIL("neighbor count mismatch"); return 1; }

        // Verify neighbor connectivity: at least some interior faces
        int interior_faces = 0;
        for (auto& nb : mesh.tet_neighbors) {
            if (nb.x >= 0) interior_faces++;
            if (nb.y >= 0) interior_faces++;
            if (nb.z >= 0) interior_faces++;
            if (nb.w >= 0) interior_faces++;
        }
        if (interior_faces == 0) { FAIL("no interior face connectivity"); return 1; }

        printf("nodes=%zu tets=%zu interior_faces=%d\n",
               mesh.nodes.size(), mesh.tets.size(), interior_faces);
        PASS;
    }

    // Summary
    printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return (test_count == pass_count) ? 0 : 1;
}
