#pragma once

#include <cstdint>

namespace aerosp {
namespace aero {
namespace cfd {

enum class ElementType : int8_t {
    TET4 = 0,
    HEX8 = 1,
    PENTA6 = 2,
    PYRAMID5 = 3
};

constexpr int ELEMENT_NODES[4] = {4, 8, 6, 5};
constexpr int ELEMENT_FACES[4] = {4, 6, 5, 5};

constexpr int FACE_NODES_PER_ELEMENT[4][6] = {
    {3, 3, 3, 3, 0, 0},
    {4, 4, 4, 4, 4, 4},
    {3, 3, 4, 4, 4, 0},
    {4, 3, 3, 3, 3, 0}
};

// Map: (element_type, local_face) -> nodes of that face
// TET4 face nodes (nodes are local indices into the cell's node array)
constexpr int TET4_FACE_NODES[4][3] = {
    {1, 2, 3},  // face 0
    {0, 3, 2},  // face 1
    {0, 1, 3},  // face 2
    {0, 2, 1}   // face 3
};

// HEX8 face nodes
constexpr int HEX8_FACE_NODES[6][4] = {
    {0, 1, 2, 3},  // bottom (-z)
    {4, 7, 6, 5},  // top    (+z)
    {0, 4, 5, 1},  // front  (-y)
    {2, 6, 7, 3},  // back   (+y)
    {0, 3, 7, 4},  // left   (-x)
    {1, 5, 6, 2}   // right  (+x)
};

// PENTA6 face nodes (prism)
constexpr int PENTA6_FACE_NODES[5][4] = {
    {0, 2, 1, -1},  // bottom tri (-z)
    {3, 4, 5, -1},  // top tri    (+z)
    {0, 1, 4, 3},   // quad face 1
    {1, 2, 5, 4},   // quad face 2
    {0, 3, 5, 2}    // quad face 3
};

// PYRAMID5 face nodes
constexpr int PYRAMID5_FACE_NODES[5][4] = {
    {0, 3, 2, 1},   // base quad (-z)
    {0, 1, 4, -1},  // tri face 1
    {1, 2, 4, -1},  // tri face 2
    {2, 3, 4, -1},  // tri face 3
    {0, 4, 3, -1}   // tri face 4
};

// Get face node indices for a given element type and local face
// Returns: pointer to 4 ints (unused entries are -1)
inline const int* get_face_nodes(ElementType type, int local_face) {
    switch (type) {
        case ElementType::TET4:
            if (local_face < 0 || local_face >= 4) return nullptr;
            return TET4_FACE_NODES[local_face];
        case ElementType::HEX8:
            if (local_face < 0 || local_face >= 6) return nullptr;
            return HEX8_FACE_NODES[local_face];
        case ElementType::PENTA6:
            if (local_face < 0 || local_face >= 5) return nullptr;
            return PENTA6_FACE_NODES[local_face];
        case ElementType::PYRAMID5:
            if (local_face < 0 || local_face >= 5) return nullptr;
            return PYRAMID5_FACE_NODES[local_face];
        default: return nullptr;
    }
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
