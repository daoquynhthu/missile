# GPU 架构稳定化计划

> 目标：在 Phase 5 viscous 之前消除 4 个架构瓶颈，使代码库能稳定扩展到集群 + 高精度计算。
>
> 依赖：Phase 1-4 已全部通过门禁，22/22 测试 PASS。

## 问题清单

| # | 问题 | 影响范围 | 严重度 |
|---|------|----------|--------|
| A | `atomicAdd` 非结合性残差归约（PH2-E-2） | 残差不可确定性复现，L2 平台 ~3e-4 | HIGH |
| B | `float` 硬编码无类型抽象 | DP 路径不可用，DNS 级精度不可达 | HIGH |
| C | 无 MPI halo 接口预留 | 集群扩展需要改写 device_mesh | MEDIUM |
| D | 默认流串行 kernel 启动 | 计算/通信无法重叠 | MEDIUM |

## 阶段划分

```
Phase 4-A ──→ Phase 4-B ──→ Phase 8 (MPI + DP)
  着色归约       Real 类型        halo 交换 + 多流
```

Phase 4-A 和 Phase 4-B 在 Phase 5 viscous 之前完成。Phase 8 项只做接口预留，不予实现。

---

## Phase 4-A：面着色确定性归约

### 动机

`euler_residual_kernel` 和 `gg_gradient_kernel` 使用 `atomicAdd` 累加 face 贡献到 cell residual/gradient。CUDA 不保证 `atomicAdd` 顺序，多次运行残差 byte-level 不同。此外 `atomicAdd` 非结合性导致收敛卡在 ~3e-4（PH2-E-2）。

### 方法：贪心面着色

将 mesh 面集合划分为 `n_colors` 个不相交子集，同一子集内的 face **不共享 left/right cell**。每个子集在独立 kernel launch 中执行，用**直写**（非 `atomicAdd`）累加 residual/gradient。

### 改动范围

#### 1. `include/aero_cfd/device_mesh.hpp`

```cpp
class DeviceMesh {
    // 新增成员
    int*    d_color_offsets_;     // 设备: [n_colors+1] int, launch 范围
    int     n_colors_;
    std::vector<int> host_color_offsets_;  // 主机 cache，用于 grid 计算
    // 新增方法
    int     color_count() const;
    const int* color_offsets_device() const;
    // face_count() 和 face_data() 现在返回已着色的索引入口
};
```

```cpp
struct DeviceFaceData {
    // 已有 face 数组（按颜色重排后会重装）
    float* d_nx; float* d_ny; float* d_nz;
    float* d_area;
    int*   d_left_cell;  int* d_right_cell;
    int*   d_boundary;
    float* d_cx; float* d_cy; float* d_cz;
};
```

face 数据物理存储不变（`d_left_cell` 等数组在 `upload()` 中按颜色重排）。color_offsets 记录每个颜色在数组中的 `[start, end)` 索引。

#### 2. `src/aero_cfd/device_mesh.cu`

在 `upload()` 末尾，在已有 face 数据上传后执行着色：

```
贪心着色算法 (pseudocode):

color_of_face[0..n_faces-1] = -1
cell_used_color[0..n_cells-1][bitset n_colors_max]

for f in 0..n_faces:
    left = face_left[f], right = face_right[f]
    used = cell_used_color[left] ∪ cell_used_color[right]
    color = smallest non-negative integer not in used
    color_of_face[f] = color
    cell_used_color[left].set(color)
    cell_used_color[right].set(color)

reorder_arrays(color_of_face)  // 稳定重排 face 数组，同色连续
build_color_offsets(n_colors, color_of_face)
```

约束：
- `n_colors_max = 64`（六面体网格典型 6-8，64 是安全上界）。超过则回退 `atomicAdd` 版（报 warning，不崩溃）。
- `n_colors_max` 定义为 `DeviceMesh` 静态常量，用于主机位集合的大小（`std::bitset<64>`）。
- 重排使用 stable partition（依次对每个颜色：将属于该色的 face 搬移到下一个连续区域），一次性完成，不申请额外大 staging buffer。

着色后的验证：
- 对每一颜色：校验所有 face 的 left/right cell 在该颜色内唯一（互斥性）。
- 如果验证失败，回退到未着色模式（`n_colors_ = 0`，通知调用方 fallback）。

#### 3. `src/aero_cfd/cfd_residual_gpu.cu`

`euler_residual_kernel` 签名增加 face 范围参数：

```cuda
__global__ void euler_residual_kernel(
    ...,                          // 现有参数不变
    int face_start, int face_end, // 新增
    float* d_residual);           // 非 atomic，直写
```

核心改动：所有 `atomicAdd(&d_residual[cell * NVAR + comp], flux_comp)` 替换为**非原子累加**。每个 cell 在每个颜色内最多被一个 face 线程写。

```cuda
// old:
atomicAdd(&d_residual[left * NVAR + 0], flux_rho);

// new:
d_residual[left * NVAR + 0] += flux_rho;
```

`launch_euler_residual_kernel` 外层循环：

```cpp
bool launch_euler_residual_kernel(..., int reconstruction_order, ...) {
    // 零残差 buffer（现有逻辑不变）
    clear_residual(d_mesh);

    int n_colors = d_mesh.color_count();
    if (n_colors == 0) {
        // 未着色回退：launch 旧 atomicAdd 版（全范围，单次）
        euler_residual_kernel_atomic<<<grid, block>>>(...);
        return check_launch("euler_residual_kernel_atomic");
    }

    for (int c = 0; c < n_colors; ++c) {
        int start = d_mesh.host_color_offsets()[c];
        int end   = d_mesh.host_color_offsets()[c+1];
        int nf_c  = end - start;
        int grid_c = (nf_c + 127) / 128;
        euler_residual_kernel<<<grid_c, block>>>(
            ..., start, end, d_mesh.residual());
        if (!cuda_check(cudaGetLastError(), format("color %d", c), error))
            return false;
    }
    return cuda_check(cudaDeviceSynchronize(), "euler residual sync", error);
}
```

#### 4. `src/aero_cfd/reconstruction_gpu.cu`

`gg_gradient_kernel` 同理：

```cuda
__global__ void gg_gradient_kernel(
    ...,                          // 现有参数
    int face_start, int face_end, // 新增
    float* d_gradients);          // 直写
```

`compute_gradients_gpu` 增加 color loop：

```cpp
bool compute_gradients_gpu(..., int* d_failed) {
    cudaMemset(d_gradients, 0, ...);

    int n_colors = mesh.color_count();
    if (n_colors == 0) {
        // atomicAdd 回退版
        gg_gradient_kernel_atomic<<<...>>>(..., d_failed);
    } else {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c+1];
            int nf_c  = end - start;
            gg_gradient_kernel<<<...>>>(..., start, end, d_gradients, d_failed);
        }
    }
    ...
}
```

`compute_limiters_gpu` 和 `bj_limiter_kernel` 不涉及原子残差累加，无需改造。

`apply_limiter_kernel` 不涉及 face 遍历，无需改造。

#### 5. 测试

| # | 测试 | 条件 | 门禁 |
|---|------|------|------|
| CFD-COLOR-1 | 小 cube 网格着色后 `n_colors > 0` 且 <= 64 | `reconstruction_order=1` | PASS |
| CFD-COLOR-2 | 同 CFD-ORACLE-EULER-2（对称 cube 力）在着色模式下运行 | `reconstruction_order=1` | PASS，力系数与 uncolored 版本 match |
| CFD-COLOR-3 | 着色模式 CF = uncolored CF，逐个 color 残余验证 | `reconstruction_order=2` | PASS（含 gradient 着色） |
| CFD-COLOR-4 | 两次连续运行 residual buffer **byte-level 相等** | deterministic = true | PASS（此为新的核心验证） |

#### 6. 门禁

- 所有现有 22 个测试 PASS（着色版回退到旧 atomicAdd 依然保证收敛）。
- 着色后 `n_colors <= 20`（六面体网格理论下界 2*3=6，可接受上界）。
- `CFD-COLOR-4` 验证 byte-level 确定性。
- `unsorted_face_data()` 随 `cudaFree` 释放，不泄露。

---

## Phase 4-B：`Real` 类型抽象

### 动机

当前所有物理量和中间计算硬编码为 `float`。DNS 级精度需要 `double` 路径。机械替换越晚做越贵。

### 方法

新增 `include/aero_cfd/real.hpp`，用宏切换 `Real` 类型。

#### `include/aero_cfd/real.hpp`

```cpp
#pragma once

#include <cmath>
#include <cuda_runtime.h>

namespace AeroSim {

#ifdef AEROSIM_REAL_DOUBLE
    using Real = double;

    __device__ __host__ inline Real real_sqrt(Real x) { return sqrt(x); }
    __device__ __host__ inline Real real_fabs(Real x) { return fabs(x); }
    __device__ __host__ inline Real real_fmin(Real x, Real y) { return fmin(x, y); }
    __device__ __host__ inline Real real_fmax(Real x, Real y) { return fmax(x, y); }
    __device__ __host__ inline bool real_isfinite(Real x) { return isfinite(x); }

    // CUDA atomic primitives for double
    __device__ inline Real real_atomic_add(Real* addr, Real val) {
        return atomicAdd(addr, val);  // sm_60+
    }
#else
    using Real = float;

    __device__ __host__ inline Real real_sqrt(Real x) { return sqrtf(x); }
    __device__ __host__ inline Real real_fabs(Real x) { return fabsf(x); }
    __device__ __host__ inline Real real_fmin(Real x, Real y) { return fminf(x, y); }
    __device__ __host__ inline Real real_fmax(Real x, Real y) { return fmaxf(x, y); }
    __device__ __host__ inline bool real_isfinite(Real x) { return __finitef(x); }

    __device__ inline Real real_atomic_add(Real* addr, Real val) {
        return atomicAdd(addr, val);
    }

    // CAS-based min/max for float
    __device__ inline Real real_atomic_min(Real* addr, Real val) {
        // 见现有 atomic_min_float 实现
    }
    __device__ inline Real real_atomic_max(Real* addr, Real val) {
        // 见现有 atomic_max_float 实现
    }
#endif

} // namespace AeroSim
```

`Real` 路径的 `atomicCAS` 无法直接复用到 `double` 或 `float`，需要 `reinterpret_cast` 到 `unsigned int*`/`unsigned long long*`。建议用条件编译在 `real_atomic_min`/`real_atomic_max` 内部处理。

CAS-based 原子操作统一收敛到 `real_atomic_min`/`real_atomic_max`。

### 改动范围

所有包含物理量计算的文件：

| 文件 | `float` → `Real` | CUDA 特异 |
|------|------------------|-----------|
| `include/aero_cfd/cfd_state.hpp` | `ConservativeState`, `PrimitiveState`, `PrimitiveGradient`, `PrimitiveLimiter` | 无 |
| `include/aero_cfd/cfd_mesh.hpp` | mesh 坐标 `cx/cy/cz` 用 `Real` | 无 |
| `include/aero_cfd/cfd_result.hpp` | `CfdForceResult`, `CfdSolveSummary` | 无 |
| `include/aero_cfd/cfd_config.hpp` | `cfl`, `gamma` 等 | 无 |
| `include/aero_cfd/device_mesh.hpp` | `DeviceFaceData` 中 `float*` → `Real*` | 是 |
| `include/aero_cfd/gpu_buffers.hpp` | `float*` → `Real*` | 是 |
| `include/aero_cfd/gpu_solver_internal.hpp` | 签名中 `float*` → `Real*` | 是 |
| `include/aero_cfd/diagnostics.hpp` | `StateBounds` 中 `float` | 无 |
| `src/aero_cfd/cfd_state.cpp` | 转换函数 | 无 |
| `src/aero_cfd/cfd_residual.cpp` | CPU 残差 | 无 |
| `src/aero_cfd/cfd_solver.cpp` | 求解器 | 无 |
| `src/aero_cfd/reconstruction.cpp` | CPU 梯度/limiter | 无 |
| `src/aero_cfd/diagnostics.cpp` | CPU 诊断 | 无 |
| `src/aero_cfd/viscous.cpp` | CPU 粘性 | 无 |
| `src/aero_cfd/device_mesh.cu` | `float*` → `Real*` | 是，包含 `cudaMemcpy` |
| `src/aero_cfd/cfd_residual_gpu.cu` | 所有 kernel 内部 `float` | 是 |
| `src/aero_cfd/reconstruction_gpu.cu` | 所有 kernel 内部 `float` | 是 |
| `src/aero_cfd/gpu_timestep.cu` | kernel 内部 | 是 |
| `src/aero_cfd/gpu_update.cu` | kernel 内部 | 是 |
| `src/aero_cfd/gpu_wall.cu` | kernel 内部 | 是 |
| `src/aero_cfd/gpu_diagnostics.cu` | kernel 内部 | 是 |
| `src/aero_cfd/gpu_solver.cu` | 求解器 | 是 |
| `tests/cfd/test_cfd_*.cpp` | 测试中 `float` 字面量 | 部分 |

`#include "aero_cfd/real.hpp"` 添加到每个更改的文件（或通过集中头文件）。

### 策略

1. 在一个独立分支执行全部替换。
2. 替换完成后，默认 `AEROSIM_REAL_DOUBLE=0`，`Real = float`，所有测试必须 PASS。
3. 确认 PASS 后，设置 `AEROSIM_REAL_DOUBLE=1`，运行非精度密集型测试（纯 mesh/configuration，不含计算），确保无编译错误。
4. 最终 PR 合并到 main。

估计改动量：~25 个文件，~1200 行 touch。大部分是 `s/float/Real/g` + 添加 `#include` + 原子操作包装。

### 门禁

- `AEROSIM_REAL_DOUBLE=0`：全部 22+ 测试 PASS，bit-wise 与当前 main 一致（不考虑着色影响）。
- `AEROSIM_REAL_DOUBLE=1`：编译成功，Mesh/Config 测试 PASS（浮点计算测试暂不要求通过，等待 Phase 8 激活）。

---

## 未来预留：MPI halo + 多流（Phase 8 前置）

当前不实现，只冻结接口契约。所有改动不会破坏现有单 GPU 代码。

### DeviceMesh 新增预留接口

```cpp
// include/aero_cfd/device_mesh.hpp
class DeviceMesh {
    // 新增（预留，当前 no-op）
    int*    d_halo_indices_ = nullptr;   // [n_halo_cells]
    Real*   d_halo_send_buf_ = nullptr;  // [n_halo_cells * NVAR]
    Real*   d_halo_recv_buf_ = nullptr;  // [n_halo_cells * NVAR]
    int     n_halo_cells_ = 0;

    // halo 数据为 nullptr → 单 GPU 模式，不做交换，性能零损失
    bool has_halo() const;
    bool allocate_halo(int n_halo_cells);
};
```

### gpu_solver.cu 预留结构

```cpp
// 多流模式（预留，当前仅默认流）
cudaStream_t stream_comp, stream_comm;
cudaStreamCreate(&stream_comp);
cudaStreamCreate(&stream_comm);

for (int iter = 0; iter < max_iter; ++iter) {
    if (mesh->has_halo()) {
        // 异步 halo 交换在 stream_comm
        exchange_halo_kernel<<<..., stream_comm>>>(mesh->halo_send_buf(), mesh->halo_recv_buf());
    }

    // 计算在 stream_comp（halo 交换在 stream_comm 上同时进行）
    timestep_kernel<<<..., stream_comp>>>(...);
    if (mesh->color_count() > 0) {
        for c in 0..n_colors:
            euler_residual_kernel<<<..., stream_comp>>>(...);
    } else {
        euler_residual_kernel_atomic<<<..., stream_comp>>>(...);
    }
    update_and_l2_kernel<<<..., stream_comp>>>(...);

    if (mesh->has_halo()) {
        cudaStreamSynchronize(stream_comm);  // 确保 halo 数据到达
    }
    check_status_kernel<<<1, 1, 0, stream_comp>>>(...);
}
```

### 门禁

- 不新增 `#include <mpi.h>`，不链接 MPI 库。
- `has_halo() == false` 时，单 GPU 性能零退化。
- 多流版本在 `MPI_ENABLED` 宏保护下编译。

---

## 执行顺序

```
Step 1: Phase 4-A 实现
  ├── device_mesh.hpp/cu: 贪心着色 + 重排
  ├── cfd_residual_gpu.cu: 着色版 euler_residual_kernel
  ├── reconstruction_gpu.cu: 着色版 gg_gradient_kernel
  ├── test_cfd_gpu.cpp: CFD-COLOR-1..4
  └── 验证: 现有 22 + 新 4 = 26 测试 PASS

Step 2: Phase 4-B 实现
  ├── include/aero_cfd/real.hpp: Real 类型 + 设备函数包装
  ├── 机械替换 (25 个文件)
  ├── 验证: AEROSIM_REAL_DOUBLE=0 全部 26+ 测试 PASS
  └── 验证: AEROSIM_REAL_DOUBLE=1 编译通过

Step 3: MPI 预留接口
  ├── device_mesh.hpp: allocate_halo() + 新增字段
  ├── gpu_solver.cu: 预留多流结构
  └── 验证: 单 GPU 零退化，测试不变

Step 4: 更新 PLAN.md + progress.md + ISSUES.md
  ├── PLAN.md: Phase 4-A/B 加入 Phase 4 子阶段
  ├── progress.md: 追加时间线
  └── ISSUES.md: PH2-E-2 → FIXED
```

## 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 着色时间随 face 数超线性增长 | 低 | 小网格无影响，大网格 ~秒级 | O(n_faces * avg_degree)，百万 face ~100ms |
| 着色后 n_colors 过大（>64） | 低 | 回退 atomicAdd | `n_colors_max=64`，达上限放弃着色 |
| face 重排破坏 mesh 处理假设 | 低 | 仅影响 face 遍历顺序，结果同 | CFD-COLOR-3 验证无 diff |
| `Real = double` 路径内存翻倍 | 中 | 大网格显存受限 | DP 仅在 Phase 8 按需开启 |

## 相关文档

- `PLAN.md` — 主执行计划
- `AERO_ACCURACY_UPGRADE.md` — 只读架构设计
- `ISSUES.md` — PH2-E-2, PH4-A-4, PH4-A-6
- `progress.md` — 进度日志
