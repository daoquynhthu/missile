#include "aero_cfd/cfd_residual.hpp"
#include "aero_cfd/reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static bool near(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::max(std::fabs(a), std::fabs(b)));
}

static int test_residual_equivalence() {
    TEST("CFD-GPU-1 Euler residual matches CPU reference for interior face");
    {
        CfdMesh mesh;
        mesh.cells.resize(2);
        CfdFace face;
        face.left_cell = 0;
        face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 0.75f;
        face.nx = 0.577350269f;
        face.ny = 0.577350269f;
        face.nz = 0.577350269f;
        mesh.faces.push_back(face);

        PrimitiveState left;
        left.rho = 1.0f;
        left.u = 2.0f;
        left.v = -0.2f;
        left.w = 0.1f;
        left.p = 0.9f;

        PrimitiveState right;
        right.rho = 0.8f;
        right.u = -0.4f;
        right.v = 0.3f;
        right.w = -0.1f;
        right.p = 0.7f;

        std::vector<ConservativeState> q;
        q.push_back(primitive_to_conservative(left, 1.4f));
        q.push_back(primitive_to_conservative(right, 1.4f));

        std::vector<EulerFlux> cpu;
        std::vector<EulerFlux> gpu;
        if (!compute_euler_residual_cpu(mesh, q, left, 1.4f, cpu)) FAIL("cpu residual failed");

        std::string error;
        if (!compute_euler_residual_gpu(mesh, q, left, 1.4f, gpu, &error)) {
            FAIL("%s", error.c_str());
        }
        if (gpu.size() != cpu.size()) FAIL("gpu size=%zu cpu size=%zu", gpu.size(), cpu.size());

        for (std::size_t i = 0; i < cpu.size(); ++i) {
            if (!near(gpu[i].mass, cpu[i].mass, 1e-6f)) FAIL("cell=%zu mass gpu=%g cpu=%g", i, gpu[i].mass, cpu[i].mass);
            if (!near(gpu[i].mom_x, cpu[i].mom_x, 1e-6f)) FAIL("cell=%zu mom_x gpu=%g cpu=%g", i, gpu[i].mom_x, cpu[i].mom_x);
            if (!near(gpu[i].mom_y, cpu[i].mom_y, 1e-6f)) FAIL("cell=%zu mom_y gpu=%g cpu=%g", i, gpu[i].mom_y, cpu[i].mom_y);
            if (!near(gpu[i].mom_z, cpu[i].mom_z, 1e-6f)) FAIL("cell=%zu mom_z gpu=%g cpu=%g", i, gpu[i].mom_z, cpu[i].mom_z);
            if (!near(gpu[i].energy, cpu[i].energy, 1e-6f)) FAIL("cell=%zu energy gpu=%g cpu=%g", i, gpu[i].energy, cpu[i].energy);
        }
        PASS;
    }
    return 0;
}

static int test_gpu_buffers() {
    TEST("CFD-GPU-2 GPU buffers own mesh state and residual transfer");
    {
        CfdMesh mesh;
        mesh.cells.resize(2);
        CfdFace face;
        face.left_cell = 0;
        face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 1.0f;
        face.nx = 1.0f;
        face.ny = 0.0f;
        face.nz = 0.0f;
        mesh.faces.push_back(face);

        PrimitiveState left;
        left.rho = 1.0f;
        left.u = 1.0f;
        left.v = 0.0f;
        left.w = 0.0f;
        left.p = 1.0f;
        PrimitiveState right = left;
        right.u = -0.5f;
        right.p = 0.75f;

        std::vector<ConservativeState> q;
        q.push_back(primitive_to_conservative(left, 1.4f));
        q.push_back(primitive_to_conservative(right, 1.4f));

        GpuCfdBuffers buffers;
        std::string error;
        if (!buffers.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (buffers.cell_count() != 2 || buffers.face_count() != 1) FAIL("counts cells=%d faces=%d", buffers.cell_count(), buffers.face_count());
        if (buffers.faces_device() == nullptr || buffers.residual_device() == nullptr) FAIL("missing device buffers");
        if (!buffers.upload_state(q, &error)) FAIL("%s", error.c_str());
        if (buffers.state_device() == nullptr) FAIL("missing device state");
        if (!compute_euler_residual_gpu(buffers, left, 1.4f, &error)) FAIL("%s", error.c_str());

        GpuCfdBuffers moved = std::move(buffers);
        std::vector<EulerFlux> residual;
        if (!moved.download_residual(residual, &error)) FAIL("%s", error.c_str());
        if (residual.size() != 2) FAIL("residual size=%zu", residual.size());
        if (std::fabs(residual[0].mass + residual[1].mass) > 1e-6f) {
            FAIL("mass conservation residual=[%g,%g]", residual[0].mass, residual[1].mass);
        }
        PASS;
    }
    return 0;
}

static int test_gpu_limiter() {
    TEST("CFD-GPU-3 GPU limiter application matches CPU reference");
    {
        CfdMesh mesh;
        mesh.cells.resize(3);
        CfdFace face;
        face.left_cell = 0;
        face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 1.0f;
        mesh.faces.push_back(face);

        std::vector<PrimitiveGradient> gradients(3);
        std::vector<PrimitiveLimiter> limiters(3);
        for (int i = 0; i < 3; ++i) {
            gradients[i].drho_dx = 1.0f + static_cast<float>(i);
            gradients[i].drho_dy = -2.0f;
            gradients[i].du_dx = 0.5f;
            gradients[i].dv_dy = -0.25f;
            gradients[i].dw_dz = 0.75f;
            gradients[i].dp_dx = -1.5f;
            gradients[i].dp_dy = 2.5f;
            limiters[i].rho = 0.25f;
            limiters[i].u = 0.5f;
            limiters[i].v = 0.75f;
            limiters[i].w = 1.0f;
            limiters[i].p = 0.1f * static_cast<float>(i + 1);
        }

        GpuCfdBuffers buffers;
        std::string error;
        if (!buffers.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!buffers.upload_gradients(gradients, &error)) FAIL("%s", error.c_str());
        if (!buffers.upload_limiters(limiters, &error)) FAIL("%s", error.c_str());
        if (!apply_limiter_gpu(buffers, &error)) FAIL("%s", error.c_str());

        std::vector<PrimitiveGradient> gpu;
        if (!buffers.download_gradients(gpu, &error)) FAIL("%s", error.c_str());
        if (gpu.size() != gradients.size()) FAIL("gradient size=%zu", gpu.size());
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            auto cpu = apply_limiter(gradients[i], limiters[i]);
            if (!near(gpu[i].drho_dx, cpu.drho_dx, 1e-6f)) FAIL("cell=%zu drho_dx gpu=%g cpu=%g", i, gpu[i].drho_dx, cpu.drho_dx);
            if (!near(gpu[i].du_dx, cpu.du_dx, 1e-6f)) FAIL("cell=%zu du_dx gpu=%g cpu=%g", i, gpu[i].du_dx, cpu.du_dx);
            if (!near(gpu[i].dv_dy, cpu.dv_dy, 1e-6f)) FAIL("cell=%zu dv_dy gpu=%g cpu=%g", i, gpu[i].dv_dy, cpu.dv_dy);
            if (!near(gpu[i].dw_dz, cpu.dw_dz, 1e-6f)) FAIL("cell=%zu dw_dz gpu=%g cpu=%g", i, gpu[i].dw_dz, cpu.dw_dz);
            if (!near(gpu[i].dp_dx, cpu.dp_dx, 1e-6f)) FAIL("cell=%zu dp_dx gpu=%g cpu=%g", i, gpu[i].dp_dx, cpu.dp_dx);
        }
        PASS;
    }
    return 0;
}

static int test_gpu_timing() {
    TEST("CFD-GPU-4 timed Euler residual returns non-negative kernel time");
    {
        CfdMesh mesh;
        mesh.cells.resize(2);
        CfdFace face;
        face.left_cell = 0;
        face.right_cell = 1;
        face.boundary = BoundaryKind::Interior;
        face.area = 0.75f;
        face.nx = 1.0f;
        mesh.faces.push_back(face);

        PrimitiveState w;
        w.rho = 1.0f;
        w.u = 0.5f;
        w.p = 1.0f;
        std::vector<ConservativeState> q(2, primitive_to_conservative(w, 1.4f));

        GpuCfdBuffers buffers;
        std::string error;
        if (!buffers.upload_mesh(mesh, &error)) FAIL("%s", error.c_str());
        if (!buffers.upload_state(q, &error)) FAIL("%s", error.c_str());
        float elapsed_ms = -1.0f;
        if (!compute_euler_residual_gpu_timed(buffers, w, 1.4f, &elapsed_ms, &error)) FAIL("%s", error.c_str());
        if (elapsed_ms < 0.0f) FAIL("elapsed_ms=%g", elapsed_ms);
        std::size_t bytes = estimate_euler_residual_gpu_bytes(mesh);
        if (bytes == 0) FAIL("estimated bytes=%zu", bytes);
        float bandwidth_gb_s = elapsed_ms > 0.0f ? (static_cast<float>(bytes) / (elapsed_ms * 1.0e6f)) : 0.0f;
        if (bandwidth_gb_s < 0.0f) FAIL("bandwidth=%g", bandwidth_gb_s);
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_residual_equivalence();
    result |= test_gpu_buffers();
    result |= test_gpu_limiter();
    result |= test_gpu_timing();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
