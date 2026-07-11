#include "aero/cfd/gpu_topology.hpp"
#include <cstdio>
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

GpuTopology detect_gpu_topology() {
    GpuTopology topo;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[GpuTopology] cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return topo;
    }

    topo.devices.resize(count);
    topo.peer_access.resize(count, std::vector<bool>(count, false));
    topo.nvlink_count.resize(count, std::vector<int>(count, 0));

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;

        GpuDeviceInfo& dev = topo.devices[i];
        dev.device_id = i;
        dev.name = prop.name;
        dev.cc_major = prop.major;
        dev.cc_minor = prop.minor;
        dev.multi_processor_count = prop.multiProcessorCount;
        dev.total_global_mem = prop.totalGlobalMem;
        cudaDeviceGetAttribute(&dev.memory_clock_rate, cudaDevAttrMemoryClockRate, i);
        cudaDeviceGetAttribute(&dev.memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, i);

        for (int j = 0; j < count; ++j) {
            if (i == j) {
                topo.peer_access[i][j] = true;
                continue;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, i, j) == cudaSuccess) {
                topo.peer_access[i][j] = (can_access != 0);
            }
            int nv_links = 0;
#if CUDART_VERSION < 12000
            if (cudaDeviceGetP2PAttribute(&nv_links, i, j, cudaDevP2PAttrNumLinks) == cudaSuccess) {
                topo.nvlink_count[i][j] = nv_links;
            }
#else
            (void)nv_links;
#endif
        }
    }
    return topo;
}

int GpuTopology::best_peer(int i) const {
    int best = -1;
    int best_score = -1;
    for (int j = 0; j < device_count(); ++j) {
        if (i == j) continue;
        if (!peer_access[i][j]) continue;
        int score = nvlink_count[i][j] > 0 ? nvlink_count[i][j] + 100 : 1;
        if (score > best_score) {
            best_score = score;
            best = j;
        }
    }
    return best;
}

std::vector<int> GpuTopology::select_devices(int n) const {
    std::vector<int> selected;
    int total = device_count();
    if (total == 0 || n <= 0) return selected;

    if (n >= total) {
        for (int i = 0; i < total; ++i) selected.push_back(i);
        return selected;
    }

    std::vector<bool> used(total, false);
    for (int round = 0; round < n; ++round) {
        int pick = -1;
        int best_score = -1;
        for (int i = 0; i < total; ++i) {
            if (used[i]) continue;
            int score = 0;
            for (int j : selected) {
                if (peer_access[i][j]) score += nvlink_count[i][j] > 0 ? nvlink_count[i][j] * 10 : 1;
            }
            if (score > best_score) {
                best_score = score;
                pick = i;
            }
        }
        if (pick < 0) {
            for (int i = 0; i < total && static_cast<int>(selected.size()) < n; ++i) {
                if (!used[i]) { pick = i; break; }
            }
        }
        if (pick >= 0) {
            selected.push_back(pick);
            used[pick] = true;
        }
    }
    return selected;
}

std::string GpuTopology::bandwidth_report() const {
    std::string r;
    for (int i = 0; i < device_count(); ++i) {
        const auto& d = devices[i];
        char buf[256];
        double mem_gb = static_cast<double>(d.total_global_mem) / (1024.0 * 1024.0 * 1024.0);
        double bw = static_cast<double>(d.memory_clock_rate) / 1e6 * d.memory_bus_width / 8.0 * 2.0;
        std::snprintf(buf, sizeof(buf), "Device %d: %s (SM %d.%d, %d SMs, %.2f GB, %.1f GB/s)\n",
            i, d.name.c_str(), d.cc_major, d.cc_minor, d.multi_processor_count, mem_gb, bw);
        r += buf;
        for (int j = 0; j < device_count(); ++j) {
            if (i == j) continue;
            if (nvlink_count[i][j] > 0) {
                std::snprintf(buf, sizeof(buf), "  NVLink %d-%d: %d links\n", i, j, nvlink_count[i][j]);
                r += buf;
            } else if (peer_access[i][j]) {
                r += "  Peer access available\n";
            }
        }
    }
    return r;
}

std::string GpuTopology::to_string() const {
    return bandwidth_report();
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
