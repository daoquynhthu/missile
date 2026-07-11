#include "aero/cfd/partition.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

static int find_longest_axis(const CfdMesh& mesh) {
    if (mesh.nodes.empty()) return 0;
    Real min_x = mesh.nodes[0].x, max_x = mesh.nodes[0].x;
    Real min_y = mesh.nodes[0].y, max_y = mesh.nodes[0].y;
    Real min_z = mesh.nodes[0].z, max_z = mesh.nodes[0].z;
    for (const auto& n : mesh.nodes) {
        if (n.x < min_x) min_x = n.x;
        if (n.x > max_x) max_x = n.x;
        if (n.y < min_y) min_y = n.y;
        if (n.y > max_y) max_y = n.y;
        if (n.z < min_z) min_z = n.z;
        if (n.z > max_z) max_z = n.z;
    }
    Real dx = max_x - min_x;
    Real dy = max_y - min_y;
    Real dz = max_z - min_z;
    if (dx >= dy && dx >= dz) return 0;
    if (dy >= dx && dy >= dz) return 1;
    return 2;
}

bool partition_linear(const CfdMesh& mesh, int n_ranks, PartitionInfo& info, std::string* error) {
    if (n_ranks <= 0) {
        if (error) *error = "n_ranks must be >= 1";
        return false;
    }
    info.n_ranks = n_ranks;
    info.my_rank = 0;

    std::size_t nc = mesh.cells.size();
    info.partition_owner.resize(nc);

    if (n_ranks == 1) {
        for (std::size_t i = 0; i < nc; ++i) info.partition_owner[i] = 0;
        info.owned_cells.resize(nc);
        for (std::size_t i = 0; i < nc; ++i) info.owned_cells[i] = static_cast<int>(i);
        info.ghost_cells.clear();
        info.ghost_owner_rank.clear();
        info.send_cells.clear();
        info.send_rank.clear();
        return true;
    }

    int axis = find_longest_axis(mesh);
    std::vector<Real> cell_centers(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        if (axis == 0) cell_centers[i] = mesh.cells[i].cx;
        else if (axis == 1) cell_centers[i] = mesh.cells[i].cy;
        else cell_centers[i] = mesh.cells[i].cz;
    }

    std::vector<int> sorted_idx(nc);
    for (std::size_t i = 0; i < nc; ++i) sorted_idx[i] = static_cast<int>(i);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        return cell_centers[a] < cell_centers[b];
    });

    int cells_per_rank = static_cast<int>((nc + static_cast<std::size_t>(n_ranks) - 1) / static_cast<std::size_t>(n_ranks));
    for (std::size_t i = 0; i < nc; ++i) {
        int rank = static_cast<int>(i) / cells_per_rank;
        if (rank >= n_ranks) rank = n_ranks - 1;
        info.partition_owner[sorted_idx[i]] = rank;
    }

    info.owned_cells.clear();
    info.ghost_cells.clear();
    info.ghost_owner_rank.clear();
    info.send_cells.clear();
    info.send_rank.clear();

    for (std::size_t i = 0; i < nc; ++i) {
        if (info.partition_owner[i] == info.my_rank) {
            info.owned_cells.push_back(static_cast<int>(i));
        }
    }

    std::unordered_set<int> ghost_set;
    std::unordered_map<int, int> ghost_to_owner;
    for (const auto& face : mesh.faces) {
        int left = face.left_cell;
        int right = face.right_cell;
        if (right < 0) continue;
        int left_owner = info.partition_owner[left];
        int right_owner = info.partition_owner[right];
        if (left_owner != right_owner) {
            if (left_owner == info.my_rank && right_owner != info.my_rank) {
                ghost_set.insert(right);
                ghost_to_owner[right] = right_owner;
            }
            if (right_owner == info.my_rank && left_owner != info.my_rank) {
                ghost_set.insert(left);
                ghost_to_owner[left] = left_owner;
            }
            if (left_owner == info.my_rank) {
                info.send_cells.push_back(right);
                info.send_rank.push_back(right_owner);
            }
            if (right_owner == info.my_rank) {
                info.send_cells.push_back(left);
                info.send_rank.push_back(left_owner);
            }
        }
    }

    info.ghost_cells.assign(ghost_set.begin(), ghost_set.end());
    info.ghost_owner_rank.resize(info.ghost_cells.size());
    for (std::size_t i = 0; i < info.ghost_cells.size(); ++i) {
        info.ghost_owner_rank[i] = ghost_to_owner[info.ghost_cells[i]];
    }

    return true;
}

std::vector<int> build_ghost_from_partition(const CfdMesh& mesh, const PartitionInfo& info) {
    std::vector<int> ghost;
    if (info.n_ranks <= 1) return ghost;

    std::unordered_set<int> ghost_set;
    for (const auto& face : mesh.faces) {
        int left = face.left_cell;
        int right = face.right_cell;
        if (right < 0) continue;
        int left_owner = info.partition_owner[left];
        int right_owner = info.partition_owner[right];
        if (left_owner != right_owner) {
            if (left_owner == info.my_rank) ghost_set.insert(right);
            if (right_owner == info.my_rank) ghost_set.insert(left);
        }
    }
    ghost.assign(ghost_set.begin(), ghost_set.end());
    return ghost;
}

void GpuPartition::release() {
    cuda_free_safe(d_partition_owner);
    cuda_free_safe(d_ghost_indices);
    cuda_free_safe(d_ghost_owner);
    n_ghost = 0;
    my_rank = 0;
}

bool upload_partition_to_device(const PartitionInfo& info, GpuPartition& gpu_part, std::string* error) {
    gpu_part.release();
    std::size_t nc = info.partition_owner.size();
    std::size_t ng = info.ghost_cells.size();

    if (!cuda_check(cudaMalloc(&gpu_part.d_partition_owner, nc * sizeof(int)), "cudaMalloc d_partition_owner", error)) {
        gpu_part.release();
        return false;
    }
    if (!cuda_check(cudaMemcpy(gpu_part.d_partition_owner, info.partition_owner.data(), nc * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_partition_owner", error)) {
        gpu_part.release();
        return false;
    }

    if (ng > 0) {
        if (!cuda_check(cudaMalloc(&gpu_part.d_ghost_indices, ng * sizeof(int)), "cudaMalloc d_ghost_indices", error)) {
            gpu_part.release();
            return false;
        }
        if (!cuda_check(cudaMalloc(&gpu_part.d_ghost_owner, ng * sizeof(int)), "cudaMalloc d_ghost_owner", error)) {
            gpu_part.release();
            return false;
        }
        if (!cuda_check(cudaMemcpy(gpu_part.d_ghost_indices, info.ghost_cells.data(), ng * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_ghost_indices", error)) {
            gpu_part.release();
            return false;
        }
        if (!cuda_check(cudaMemcpy(gpu_part.d_ghost_owner, info.ghost_owner_rank.data(), ng * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_ghost_owner", error)) {
            gpu_part.release();
            return false;
        }
    }

    gpu_part.n_ghost = static_cast<int>(ng);
    gpu_part.my_rank = info.my_rank;
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
