# CPU Oracle 解耦与 FP64 黄金标准计划

> 关联: `docs/AERO_ACCURACY_UPGRADE.md`（架构设计）、`docs/ISSUES.md`（工程质量问题）、`docs/progress.md`（进度日志）

## 目标

1. **解耦**：把 GPU 调度代码从 `cfd_solver.cpp` 剥离，CPU 求解器变成纯 CPU 库，零 CUDA 依赖
2. **FP64 精度**：CPU 求解器独立编译为 `Real=double`，作为 GPU FP32 求解器的黄金标准
3. **补齐 2nd-order**：CPU 求解器增加二阶重建路径（梯度+限制器+面重构），与 GPU 能力对等
4. **清理工程质量问题**：同时修复 ISSUES.md 中与构建/代码结构相关的 9 个条目

## 架构变更

```
                      ┌──────────────────────────────────┐
                      │          missile_lib              │
                      │  (Real=float, .cu + .cpp)         │
                      │  包含: cfd_solver_gpu.cpp         │
                      │  不包含: cfd_solver.cpp (已迁出)   │
                      └──────────────────────────────────┘
                                
  新文件:
  ┌────────────────────────────────────────────────────────┐
  │ cfd_solver.cpp (纯 CPU 求解器, 零 GPU 头文件依赖)       │
  │   - solve_from_state()     ← 原函数，不变              │
  │   - compute_euler_residual_cpu() ← 新增 2nd-order 路径 │
  │   - 无 #include "gpu_solver.hpp"                       │
  │   - 无 DeviceMesh 引用                                 │
  └────────────────────────────────────────────────────────┘
                                
  ┌────────────────────────────────────────────────────────┐
  │ cfd_solver_gpu.cpp (GPU 调度器, 仅链接到 missile_lib)   │
  │   - solve_gpu_dispatch()   ← 从原 solve() 拆出的 GPU 分支│
  │   - cpu_oracle 比较逻辑                                 │
  │   - #include "gpu_solver.hpp"/"device_mesh.hpp"         │
  └────────────────────────────────────────────────────────┘

  新编译目标:
  ┌────────────────────────────────────────────────────────┐
  │ test_oracle_fp64.exe                                    │
  │   Real=double, 仅编译 CPU 求解器 .cpp 文件               │
  │   零 .cu 文件, 零 CUDA 设备链接                          │
  │   生成黄金标准 JSON                                     │
  └────────────────────────────────────────────────────────┘
```

## 关联 ISSUE 修复清单

| ID | 严重性 | 问题 | 修复方式 |
|-----|--------|------|----------|
| PH4-A-6 / PH4-A-14 | MEDIUM | CPU 无 2nd-order 路径 | Step 2 实现 |
| PERF-H1 | HIGH | 非 CUDA .cpp 经 nvcc 编译 | Step 4b 独立编译目标 |
| PH4-B-8 | MEDIUM | 同上（测试目标） | Step 4b |
| PERF-H2 | HIGH | 27 次设备链接 ~175s | Step 4b 纯 C++ 目标 |
| PERF-H6 | HIGH | CUDA 架构检测失效 | Step 4a 修复 |
| PERF-I1 | MEDIUM | 无 BUILD_TESTING 守卫 | Step 4c 添加 |
| AUDIT-FREE-M1 | MEDIUM | #include 在 #pragma once 前 | Step 5a 修复 |
| AUDIT-FREE-L5 | LOW | cfd_solver.cpp 重复代码 | Step 1 消除 |
| PH4-B-10 | INFO | CUDA_KERNEL_CHECK 未用 | Step 5b 删除 |

---

## 执行步骤

### Step 1: GPU 调度逻辑剥离

**目标**: 把 `solve()` 中 GPU 分支移到新文件，`cfd_solver.cpp` 变纯 CPU 文件。

**改动文件**:
- `src/aero/cfd/cfd_solver.cpp` — 删除 `gpu_solver.hpp` include，删除 GPU 分支 + cpu_oracle 代码块
- `src/aero/cfd/cfd_solver_gpu.cpp` — 新建，包含 GPU 调度函数
- `include/aero/cfd/cfd_solver.hpp` — `CfdSolver` 类定义不变，前向声明保留
- `src/aero/cfd/cfd_solver_gpu.hpp` — 新建，声明 `solve_gpu_dispatch()`

**细节**:
- GPU 分支从 `CfdSolver::solve()` 拆出为独立函数 `solve_gpu_dispatch(const CfdMesh&, const FreestreamCondition&, const CfdConfig&)`，返回 `CfdSolveSummary`
- `cfd_solver.cpp` 的 `solve()` 只保留 `make_initial_q()` + `solve_from_state()` CPU 路径
- `cfd_solver.cpp` 删除 `#include "aero/cfd/gpu_solver.hpp"`
- 原 `make_initial_q` lambda 的两个拷贝合并为一个（修 AUDIT-FREE-L5）
- 测试代码 `test_cfd_gpu.cpp` 改为调用 `solve_gpu_dispatch()` 代替 `solver.solve(use_gpu=true)`

**验证**:
- `missile_lib` 编译通过
- `test_cfd_gpu.exe` 全部测试通过（56 项）

### Step 2: CPU 2nd-order 重建路径

**目标**: `compute_euler_residual_cpu` 支持 `reconstruction_order==2`。

**改动文件**:
- `include/aero/cfd/cfd_residual.hpp` — 声明新增的重载
- `src/aero/cfd/cfd_residual.cpp` — 实现 2nd-order 残差路径

**细节**:
- 新增 `compute_euler_residual_cpu_order2()` 函数：
  1. 调用 `compute_green_gauss_gradients(mesh, q, gamma)` 获取梯度
  2. 调用 `compute_barth_jespersen_limiters(mesh, q, grads, gamma)` 获取限制器
  3. 逐面：对 left/right 调用 `reconstruct_primitive()` → `hllc_flux()` → 累加残差
  4. 与 `CfdSolver::solve_from_state()` 中已有的 order2 循环一致
- `CfdSolver::solve_from_state()` 保留现有 order2 内联实现或委托给新函数（二选一，推荐统一调用新函数以消除重复）

**验证**:
- CPU order1 残差与 GPU order1 残差匹配（已有测试）
- CPU order2 残差与 GPU order2 残差匹配（新增测试）
- 所有残差分量（含 `turbulence`）正确处理

### Step 3: GPU 测试适配

**目标**: 所有测试从 `solver.solve(use_gpu=true)` 改为调用 `solve_gpu_dispatch()`。

**改动文件**:
- `tests/cfd/test_cfd_gpu.cpp`

**细节**:
- 全局搜索 `config.use_gpu = true` 和 `solver.solve(cond, config)` 组合
- 替换为 `solve_gpu_dispatch(solver.mesh(), cond, config)`
- 保持所有通过条件不变

**验证**:
- 全部 56+ 测试通过

### Step 4: CMake 重构

#### 4a: 修复 CUDA 架构检测 (PERF-H6)

**改动**: `CMakeLists.txt:23-34`

```cmake
# 替换现有自动检测逻辑：
set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90;100" CACHE STRING
    "CUDA architectures to build for" FORCE)
```

#### 4b: 新增 FP64 CPU Oracle 编译目标

**改动**: `tests/CMakeLists.txt`

```cmake
set(CPU_ORACLE_SOURCES
    src/aero/cfd/cfd_solver.cpp
    src/aero/cfd/cfd_residual.cpp
    src/aero/cfd/reconstruction.cpp
    src/aero/cfd/viscous.cpp
    src/aero/cfd/rans.cpp
    src/aero/cfd/diagnostics.cpp
    src/aero/cfd/mesh_metrics.cpp
    src/aero/cfd/mesh_validator.cpp
)
add_executable(test_oracle_fp64
    tests/cfd/test_oracle_fp64.cpp
    ${CPU_ORACLE_SOURCES}
)
target_compile_definitions(test_oracle_fp64 PRIVATE AEROSP_REAL_DOUBLE)
target_include_directories(test_oracle_fp64 PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}     # 只为 real.hpp 中的 cuda_runtime.h
)
target_link_libraries(test_oracle_fp64 PRIVATE Eigen3::Eigen)
set_target_properties(test_oracle_fp64 PROPERTIES
    LANGUAGE CXX                    # 纯 C++，无 CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
```

#### 4c: BUILD_TESTING 守卫 (PERF-I1)

**改动**: `CMakeLists.txt`

```cmake
option(BUILD_TESTING "Build test executables" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### Step 5: 代码清洁

#### 5a: 修复 #include 顺序 (AUDIT-FREE-M1)

**改动文件**:
- `include/aero/cfd/cfd_solver.hpp`
- `include/aero/cfd/cfd_residual.hpp`
- `include/aero/cfd/gpu_solver.hpp`
- `include/aero/cfd/viscous.hpp`

把 `#include` 行移到 `#pragma once` 之后。

#### 5b: 删除未使用的 CUDA_KERNEL_CHECK (PH4-B-10)

**改动文件**:
- `include/aero/cfd/cuda_utils.hpp` — 删除宏定义
- `docs/AGENTS.md` — 删除对 CUDA_KERNEL_CHECK 的要求

### Step 6: FP64 黄金标准测试

**目标**: 生成可复现的 FP64 参考解。

**改动文件**:
- `tests/cfd/test_oracle_fp64.cpp` — 新建

**内容**:
```cpp
// 1. 基准案例: flat plate (M=0.2, viscous, turbulent)
// 2. 基准案例: cylinder (M=2.0, Euler)
// 3. 运行 CPU FP64 求解器 (50 迭代, 收敛到稳态)
// 4. 输出 CfdSolveSummary → JSON (力系数 + 残差历史)
// 5. 回归校验: 与内置期望值比较（或跳过如无 golden 文件）
```

### Step 7: 验证

```bash
# 1. missile_lib 编译
cmake -B build && cmake --build build --target missile_lib

# 2. GPU 测试全部通过
./build/bin/test_cfd_gpu.exe

# 3. FP64 oracle 编译
cmake --build build --target test_oracle_fp64

# 4. FP64 oracle 运行
./build/bin/test_oracle_fp64.exe
```

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **修改** | `src/aero/cfd/cfd_solver.cpp` | 删 GPU include + GPU 分支，合并 `make_initial_q` |
| **新建** | `src/aero/cfd/cfd_solver_gpu.cpp` | GPU 调度函数 |
| **新建** | `include/aero/cfd/cfd_solver_gpu.hpp` | GPU 调度函数声明 |
| **修改** | `include/aero/cfd/cfd_residual.hpp` | 声明 `compute_euler_residual_cpu_order2` |
| **修改** | `src/aero/cfd/cfd_residual.cpp` | 实现 2nd-order 残差路径 |
| **修改** | `tests/cfd/test_cfd_gpu.cpp` | 调用 `solve_gpu_dispatch()` |
| **修改** | `CMakeLists.txt` | 修复 PERF-H6, 添加 PERF-I1 守卫 |
| **修改** | `tests/CMakeLists.txt` | 新增 `test_oracle_fp64` 目标 |
| **新建** | `tests/cfd/test_oracle_fp64.cpp` | 黄金标准生成 |
| **修改** | `include/aero/cfd/cfd_solver.hpp` | 修复 #pragma once (AUDIT-FREE-M1) |
| **修改** | `include/aero/cfd/cfd_residual.hpp` | 同上 |
| **修改** | `include/aero/cfd/gpu_solver.hpp` | 同上 |
| **修改** | `include/aero/cfd/viscous.hpp` | 同上 |
| **修改** | `include/aero/cfd/cuda_utils.hpp` | 删除 CUDA_KERNEL_CHECK |

## 进度追踪

- [x] Step 1: GPU 调度剥离
- [ ] Step 2: CPU 2nd-order 路径
- [x] Step 3: GPU 测试适配
- [x] Step 4a: 修复 CUDA 架构检测
- [x] Step 4b: FP64 编译目标
- [x] Step 4c: BUILD_TESTING 守卫
- [x] Step 5a: #include 顺序修复
- [x] Step 5b: 删除 CUDA_KERNEL_CHECK
- [x] Step 6: 黄金标准测试
- [x] Step 7: 验证全部通过
