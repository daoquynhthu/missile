#include "aero/cfd/gpu_topology.hpp"
#include "aero/cfd/partition.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include <cstdio>
#include <cstdlib>

#define TEST(name) do { std::printf("  " name "... "); } while(0)
#define PASS() do { std::printf("PASS\n"); } while(0)
#define FAIL(msg) do { std::printf("FAIL: %s\n", msg); std::exit(1); } while(0)

using namespace aerosp;
using namespace aerosp::aero::cfd;

static void test_topology_detection() {
    TEST("topology detection runs without error");
    GpuTopology topo = detect_gpu_topology();
    if (topo.device_count() <= 0) {
        std::printf("SKIP (no GPU)\n");
        return;
    }
    if (topo.device_count() > 0 && topo.devices[0].cc_major == 0) {
        FAIL("device 0 has compute capability 0.0");
    }
    if (topo.peer_access.size() != static_cast<std::size_t>(topo.device_count())) {
        FAIL("peer_access matrix size mismatch");
    }
    std::printf("PASS (%d devices)\n", topo.device_count());
}

static void test_select_devices() {
    TEST("select_devices returns correct count");
    GpuTopology topo = detect_gpu_topology();
    if (topo.device_count() <= 0) {
        std::printf("SKIP (no GPU)\n");
        return;
    }
    auto sel = topo.select_devices(1);
    if (sel.size() != 1) FAIL("expected 1 device");
    if (sel[0] != 0) FAIL("expected device 0 as first selection");
    std::printf("PASS\n");
}

static void test_bandwidth_report() {
    TEST("bandwidth_report returns non-empty string");
    GpuTopology topo = detect_gpu_topology();
    if (topo.device_count() <= 0) {
        std::printf("SKIP (no GPU)\n");
        return;
    }
    std::string report = topo.bandwidth_report();
    if (report.empty()) FAIL("report is empty");
    if (report.find("Device 0") == std::string::npos) FAIL("report missing Device 0");
    std::printf("PASS\n");
}

static void test_partition_linear_single() {
    TEST("linear partition n_ranks=1: all cells owned by rank 0");
    CfdMesh mesh;
    mesh.cells.resize(10);
    for (int i = 0; i < 10; ++i) {
        mesh.cells[i].cx = static_cast<Real>(i);
        mesh.cells[i].cy = 0.0f;
        mesh.cells[i].cz = 0.0f;
        CfdNode n;
        n.x = static_cast<Real>(i); n.y = 0.0f; n.z = 0.0f;
        mesh.nodes.push_back(n);
        n.x = static_cast<Real>(i) + 1.0f; n.y = 0.0f; n.z = 0.0f;
        mesh.nodes.push_back(n);
        n.x = static_cast<Real>(i); n.y = 1.0f; n.z = 0.0f;
        mesh.nodes.push_back(n);
        n.x = static_cast<Real>(i) + 1.0f; n.y = 1.0f; n.z = 0.0f;
        mesh.nodes.push_back(n);
    }
    PartitionInfo info;
    if (!partition_linear(mesh, 1, info)) FAIL("partition_linear failed");
    if (info.n_ranks != 1) FAIL("expected 1 rank");
    if (info.owned_cells.size() != 10) FAIL("expected 10 owned cells");
    if (!info.ghost_cells.empty()) FAIL("expected no ghost cells for n_ranks=1");
    PASS();
}

static void test_partition_linear_multi() {
    TEST("linear partition n_ranks=2: total cells = sum(owned + ghost) per rank");
    CfdMesh mesh;
    mesh.cells.resize(100);
    for (int i = 0; i < 100; ++i) {
        mesh.cells[i].cx = static_cast<Real>(i) * 1.0f;
        mesh.cells[i].cy = 0.0f;
        mesh.cells[i].cz = 0.0f;
    }
    mesh.faces.resize(99);
    for (int i = 0; i < 99; ++i) {
        mesh.faces[i].left_cell = i;
        mesh.faces[i].right_cell = i + 1;
    }
    PartitionInfo info;
    if (!partition_linear(mesh, 2, info)) FAIL("partition_linear failed");
    if (info.n_ranks != 2) FAIL("expected 2 ranks");
    if (info.owned_cells.empty()) FAIL("expected some owned cells");
    PASS();
}

int main() {
    std::printf("[GPU Topology Test]\n");
    test_topology_detection();
    test_select_devices();
    test_bandwidth_report();
    std::printf("[Partition Test]\n");
    test_partition_linear_single();
    test_partition_linear_multi();
    std::printf("All PASS\n");
    return 0;
}
