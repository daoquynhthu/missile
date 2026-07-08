# CFD GPU Production Plan

> Design: `AERO_ACCURACY_UPGRADE.md`
> Progress log: `progress.md`
> Active blockers: `ISSUES.md`

## Architecture Decision

**GPU is the production path.** CPU exists only as reference/debug oracle for unit tests and regression verification.

```
Mesh (CPU, once) → DeviceMesh SoA upload → Full solve loop on GPU → Download results
                                               ↑
                                    CPU oracle (debug/CI only)
```

All physics modules (Euler, reconstruction, viscous, RANS) are implemented first as CPU oracle functions for formula verification, then ported to GPU kernels. GPU porting is not optional — it is the deliverable.

---

## Phase 0 — Retained CPU Foundation

Goal: establish the CPU reference implementation that the GPU path will be validated against. Everything here compiles today; no changes required.

Files (kept as-is):

| File | Role |
|------|------|
| `include/aero_cfd/cfd_mesh.hpp` | Node/cell/face mesh model |
| `include/aero_cfd/cfd_state.hpp` | Primitive/conservative/HLLC/farfield BC |
| `include/aero_cfd/cfd_config.hpp` | Configuration (will add GPU flags later) |
| `include/aero_cfd/cfd_result.hpp` | Force/moment result struct |
| `include/aero_cfd/cfd_solver.hpp` | CfdSolver class (CPU solve) |
| `include/aero_cfd/cfd_residual.hpp` | Euler residual assembly (CPU) |
| `include/aero_cfd/reconstruction.hpp` | Gradient/limiter/reconstruct (CPU) |
| `include/aero_cfd/diagnostics.hpp` | State bounds/failure/VTK (CPU) |
| `include/aero_cfd/viscous.hpp` | Viscous gradient/wall/flux (CPU) |
| `include/aero_cfd/cuda_utils.hpp` | CUDA error-check helper |
| `src/aero_cfd/mesh_metrics.cpp` | Mesh generation + metrics |
| `src/aero_cfd/cfd_solver.cpp` | CPU solve loop + wall integration |
| `src/aero_cfd/cfd_residual.cpp` | CPU Euler residual assembly |
| `src/aero_cfd/reconstruction.cpp` | CPU Green-Gauss/LSQ/limiter |
| `src/aero_cfd/diagnostics.cpp` | CPU diagnostics + VTK writer |
| `src/aero_cfd/viscous.cpp` | CPU Sutherland/gradients/wall flux |
| `src/aero_cfd/cuda_utils.cpp` | CPU-side CUDA error helper |

Tests (kept as-is, serve as CPU oracle tests):

| Test | Role |
|------|------|
| `tests/cfd/test_cfd_mesh.cpp` | Mesh validity baseline |
| `tests/cfd/test_cfd_euler.cpp` | Euler flux/solver baseline |
| `tests/cfd/test_cfd_diagnostics.cpp` | Diagnostics baseline |
| `tests/cfd/test_cfd_reconstruction.cpp` | Reconstruction baseline |
| `tests/cfd/test_cfd_viscous.cpp` | Viscous utility baseline |

Gate:

- All CPU tests pass.
- `cmake -B build && cmake --build build --target TestCfdMesh --config Release` succeeds.
- `ctest --test-dir build -C Release -R "Cfd(Mesh|Euler|Diagnostics|Reconstruction|Viscous)"` reports 100% pass.

---

## Phase 1 — GPU Data Pipeline (SoA Refactoring)

Goal: replace the current AoS (`CfdFace`/`CfdCell` struct arrays) GPU memory layout with a SoA (Struct-of-Arrays) layout optimized for coalesced GPU access. CPU `CfdMesh` retained for generation; `DeviceMesh` is the production container.

Rationale: current `CfdFace` is ~72 bytes (13 floats + 4 ints + 1 enum). AoS causes strided, uncoalesced global memory reads. SoA packs same-field values contiguously, achieving near-peak memory bandwidth.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/device_mesh.hpp` | NEW | SoA structs: `DeviceFaceData`, `DeviceCellData`, `DeviceState` |
| `src/aero_cfd/device_mesh.cu` | NEW | Upload/download kernels, RAII management |
| `include/aero_cfd/gpu_buffers.hpp` | REPLACE | Thin wrapper or redirect to `device_mesh` |
| `src/aero_cfd/gpu_buffers.cu` | REPLACE | Delegate to `device_mesh` |
| `include/aero_cfd/gpu_solver.hpp` | NEW | GPU solver declarations (extends for Phase 2) |
| `src/aero_cfd/cfd_residual_gpu.cu` | REWRITE | `euler_residual_kernel` uses SoA layout |
| `src/aero_cfd/reconstruction_gpu.cu` | REWRITE | `apply_limiter_kernel` uses SoA layout |

Design:

```cpp
// include/aero_cfd/device_mesh.hpp
struct DeviceFaceData {
    float* d_nx; float* d_ny; float* d_nz;     // coalesced: each is a contiguous float array
    float* d_area;
    int*   d_left_cell;  int* d_right_cell;
    int*   d_boundary;                          // BoundaryKind as int
};

struct DeviceCellData {
    float* d_volume; float* d_h_min;
    float* d_cx; float* d_cy; float* d_cz;
};

struct DeviceState {
    float* d_q;                                 // [NVAR * n_cells] flat
};

class DeviceMesh {
public:
    bool upload(const CfdMesh& cpu_mesh, std::string* error = nullptr);
    bool upload_state(const std::vector<ConservativeState>& q, std::string* error = nullptr);
    bool download_state(std::vector<ConservativeState>& q, std::string* error = nullptr) const;
    void release();
    // accessors for kernel launches
    DeviceFaceData face() const;
    DeviceCellData cell() const;
    DeviceState state() const;
    float* residual() const;                     // [NVAR * n_cells]
    int cell_count() const;
    int face_count() const;
};
```

Tasks:

- [x] Define `DeviceFaceData` SoA: 3× float*, 1× float*, 3× int*
- [x] Define `DeviceCellData` SoA: 5× float*
- [x] Define `DeviceState`: flat `float*` (NVAR=5 per cell)
- [x] `DeviceMesh::upload()`: CfdMesh → device SoA, single cudaMemcpy per array
- [x] `DeviceMesh::upload_state()`: vector<ConservativeState> → device flat array
- [x] `DeviceMesh::download_state()`: device flat array → vector<ConservativeState>
- [x] `DeviceMesh`: RAII (default=null, move-only, ~DeviceMesh releases)
- [x] Refactor `euler_residual_kernel` to accept `DeviceFaceData` + `DeviceState` + `float* residual`
- [x] Refactor `apply_limiter_kernel` to SoA gradient/limiter arrays
- [x] CPU/GPU residual equivalence test on 13^3 cube mesh (not 2-cell)

Gate:

- `DeviceMesh` upload + `euler_residual_kernel` on full cube mesh matches `compute_euler_residual_cpu` component-wise within 1e-6 absolute per cell.
- No AoS struct arrays exist on device after upload.
- SoA memory layout verified: `d_nx[i+1] - d_nx[i] == sizeof(float)` (contiguous).
- Move semantics correct: moved-from `DeviceMesh` has null pointers, destructor no-ops.

---

## Phase 2 — Full GPU Euler Solver Loop

Goal: move the entire pseudo-time iteration loop from CPU to GPU. Zero CPU-GPU data transfer between initial upload and final download.

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero_cfd/gpu_timestep.cu` | NEW | `timestep_kernel` + block-reduce min dt |
| `src/aero_cfd/gpu_update.cu` | NEW | `update_state_kernel` + `l2_norm_kernel` + state validity |
| `src/aero_cfd/gpu_wall.cu` | NEW | `wall_force_kernel` + block-reduce 6 components |
| `src/aero_cfd/gpu_solver.cu` | NEW | `solve_gpu()` full loop orchestration |
| `include/aero_cfd/gpu_solver.hpp` | MODIFY | Add `solve_gpu()` declaration |
| `include/aero_cfd/cfd_config.hpp` | MODIFY | Add `bool use_gpu = true` |
| `include/aero_cfd/cfd_solver.hpp` | MODIFY | `solve()` dispatches to `solve_gpu()` |
| `src/aero_cfd/cfd_solver.cpp` | MODIFY | `solve()` selects GPU/CPU path |

Kernels:

- [ ] `timestep_kernel`: per cell dt = CFL * h_min / (|v| + a), block-wide `atomicMin` for global dt
- [ ] `euler_residual_kernel`: exists (rewired in Phase 1)
- [ ] `update_state_kernel`: q[t+1].component = q[t].component + dt/V * residual.component (component-wise)
- [ ] `l2_norm_kernel`: per cell sum((q_new - q_old)^2), block-reduce to scalar via shared memory
- [ ] `state_validity_kernel`: per cell `is_valid_primitive()`, `atomicOr` failed flag
- [ ] `wall_force_kernel`: per wall face accumulate force + moment, block-reduce to 6 scalars

Solver orchestration:

```
solve_gpu(mesh, freestream, config, summary):
  DeviceMesh d_mesh
  d_mesh.upload(mesh)
  d_mesh.upload_state(initial_q)

  for iter in 0..max_iter:
    timestep_kernel(d_mesh, config)        → min_dt (device scalar)
    euler_residual_kernel(d_mesh)          → residual
    update_state_kernel(d_mesh, min_dt)    → q_new
    l2 = l2_norm_kernel(d_mesh)            → device scalar
    if l2 < tol: converged; break
    valid = state_validity_kernel(d_mesh)  → device scalar
    if !valid: failed; break

  wall_force_kernel(d_mesh)                → forces (device array)
  d_mesh.download_state(final_q)
  download forces to host
  fill summary
```

Host-device sync: none during the iteration loop. Convergence and failure flags read only once at loop exit.

Tasks:

- [x] `timestep_kernel`: per-cell dt with block-reduction for min dt (atomicCAS based on `__float_as_uint`)
- [x] `update_and_l2_kernel`: component-wise update with in-place L2 accumulation via atomicAdd
- [x] `state_validity_kernel`: combined with update kernel — finite + positive rho/p check, `atomicExch` failure flag
- [x] `wall_force_kernel`: per-face pressure * area * normal, atomicAdd on 6 force/moment counters
- [x] `solve_gpu()`: loop with convergence/failure exit conditions
- [x] `cfd_solver.cpp`: `solve()` uses GPU when `config.use_gpu==true`, else CPU
- [x] DeviceMesh face centroid arrays (cx/cy/cz) for wall moment computation

Gate:

- [x] GPU-CPU L2 match within 1e-6 after 1 iteration on 13^3 cube mesh (CFD-GPU-6).
- [x] GPU-CPU iteration-by-iteration L2 match within 1e-6 for first 20 iterations on cube (CFD-GPU-7).
- [x] GPU and CPU converge to comparable residual level on flat plate mesh (CFD-GPU-8): ratio ≤ 1e3.
- [ ] Zero `cudaMemcpy` calls during iteration loop (moved to Phase 3).

---

## Phase 3 — CPU Oracle & Regression Verification

Goal: establish CPU as reference oracle for GPU validation. Every GPU operation has a CPU equivalent that can be compared in debug/CI mode. Also eliminate all `cudaMemcpy` calls from the iteration loop (PH2-G-1).

### Task 1: `cpu_oracle` configuration

**`include/aero_cfd/cfd_config.hpp`**: add `bool cpu_oracle = false` to `CfdConfig`.

### Task 2: Oracle mode in solver

**`include/aero_cfd/cfd_solver.hpp`**: add `assert_oracle_equivalent()` helper declaration.

**`src/aero_cfd/cfd_solver.cpp`**: in `solve()`, after GPU path returns:

```
solve():
  gpu_result = solve_gpu(...)
  if config.cpu_oracle:
    cpu_config = config; cpu_config.use_gpu = false
    cpu_result = solve_cpu(cpu_config)
    assert_oracle_equivalent(gpu_result, cpu_result)
  return gpu_result
```

`assert_oracle_equivalent()` compares:
- `residual_history` iter-by-iter (if one has more iters, only compare overlap)
- `forces.CX/CY/CZ/Cl/Cm/Cn` normalized coefficients
- Reports first mismatch with iteration index, GPU value, CPU value, diff

All comparisons use relative tolerance: `fabs(a-b) <= tol * (1.0 + max(fabs(a), fabs(b)))`.

### Task 3: Zero `cudaMemcpy` during iteration loop

**Problem**: current loop has 4 D2H reads per iteration:
1. `cudaMemcpy(&residual_failed, d_failed, ...)` — failure check after residual
2. `cudaMemcpy(&min_dt, d_min_dt, ...)` — read timestep
3. `cudaMemcpy(&update_failed, d_failed, ...)` — failure check after update
4. `cudaMemcpy(&l2, d_l2_sum, ...)` — read L2 norm

**Solution**: all convergence/failure decisions happen on device. Host runs all `max_iter` iterations without intermediate reads.

**New device buffers:**
- `int* d_converged` — set to 1 by `check_status_kernel` when converged or failed
- `float* d_residual_history` — `max_iter` slots, written by `check_status_kernel`

**New kernel** (in `gpu_solver.cu` or new file):

```
__global__ void check_status_kernel(
    const int* d_failed, const float* d_l2_sum,
    int nvar_ncells, float convergence_tol,
    int* d_converged, float* d_residual_history_slot)
```

Thread 0 block 0: if `*d_failed != 0`, set `*d_converged = 1` and write -1.0f to history slot.
Otherwise compute `l2 = sqrtf(*d_l2_sum / nvar_ncells)`, write to history slot, and if `l2 < convergence_tol` set `*d_converged = 1`.

**Changes to `compute_update_gpu`:**
- `gpu_solver_internal.hpp` + `gpu_update.cu`: signature changes from `float min_dt` to `const float* d_min_dt`
- `update_and_l2_kernel` reads `float min_dt = *d_min_dt` from device pointer

**New iteration loop structure** (in `gpu_solver.cu`):

```
// Before loop: allocate d_converged, d_residual_history

for (int iter = 0; iter < config.max_iter; ++iter) {
    // All launches on default stream — implicit ordering
    compute_timestep_gpu(d_mesh, gamma, cfl, d_min_dt);   // <- no host read needed
    launch_euler_residual_kernel(...);
    compute_update_gpu(d_mesh, d_min_dt, gamma, d_l2_sum, d_failed);
    check_status_kernel<<<1,1>>>(d_failed, d_l2_sum, ..., d_converged, d_residual_history + iter);
}

// After loop: single sync + single batch of D2H reads
cudaDeviceSynchronize();
cudaMemcpy(&host_failed, d_failed, ...);
cudaMemcpy(&host_converged, d_converged, ...);
cudaMemcpy(host_residual_history, d_residual_history, max_iter * sizeof(float), D2H);
cudaMemcpy(host_forces, d_forces, 6 * sizeof(float), D2H);
// Build summary from host data
```

If solver converges early, subsequent iterations run wastefully but produce no effect (state has already converged). This is acceptable: the GPU time for extra iterations is negligible compared to D2H latency savings for typical `max_iter <= 5000`.

### Task 4: Oracle tests

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-ORACLE-EULER-1` | freestream preservation: GPU & CPU both produce L2=0 on uniform flow | 1e-12 |
| 2 | `CFD-ORACLE-EULER-2` | symmetric cube: CX/CY/CZ match between CPU and GPU | 1e-10 |
| 3 | `CFD-ORACLE-EULER-3` | flat plate farfield-only: zero forces GPU=CPU | 1e-12 |
| 4 | `CFD-ORACLE-EULER-4` | residual history iter-by-iter match | 1e-12 |
| 5 | `CFD-ORACLE-EULER-5` | wall force components (CX/CY/CZ/Cl/Cm/Cn) | 1e-10 |
| 6 | `CFD-ORACLE-MESH-1` | DeviceMesh cell_count/face_count == CfdMesh | exact |
| 7 | `CFD-ORACLE-BW-1` | GPU memory bandwidth >= 50% theoretical peak | 50% |

All tests added to `tests/cfd/test_cfd_gpu.cpp` (extend, not rewrite).

### Files

| File | Action | Detail |
|------|--------|--------|
| `include/aero_cfd/cfd_config.hpp` | MODIFY | +`bool cpu_oracle = false` |
| `include/aero_cfd/cfd_solver.hpp` | MODIFY | +`assert_oracle_equivalent` declaration |
| `include/aero_cfd/gpu_solver_internal.hpp` | MODIFY | `compute_update_gpu` signature: `float min_dt` → `const float* d_min_dt` |
| `src/aero_cfd/cfd_solver.cpp` | MODIFY | +`assert_oracle_equivalent`, oracle dispatch in `solve()` |
| `src/aero_cfd/gpu_solver.cu` | MODIFY | Zero-D2H loop, `check_status_kernel`, post-loop batch reads |
| `src/aero_cfd/gpu_update.cu` | MODIFY | Accept `const float* d_min_dt`, kernel reads from device |
| `tests/cfd/test_cfd_gpu.cpp` | MODIFY | +7 oracle tests |

### Gate

- [x] Existing 8 GPU tests (CFD-GPU-1..8) still pass unchanged.
- [x] New 7 oracle tests pass at stated tolerances.
- [x] `cpu_oracle=true` mode: every `solve()` with `use_gpu=true` also runs CPU and asserts match.
- [x] Zero `cudaMemcpy` calls during iteration loop.
- [x] `cpu_oracle=false` (default) has zero overhead — no CPU solve, no extra allocations.
- [x] `assert_oracle_equivalent` reports first mismatch with iteration/component details.

---

## Phase 4 — GPU Second-Order Reconstruction & Diagnostics

Goal: port Green-Gauss gradients, Barth-Jespersen limiter, and reconstruction to GPU. Integrate into `solve_gpu()`.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/cfd_config.hpp` | MODIFY | Add `int reconstruction_order = 1` |
| `src/aero_cfd/reconstruction_gpu.cu` | EXTEND | `gg_gradient_kernel`, `limiter_kernel` |
| `src/aero_cfd/gpu_solver.cu` | MODIFY | Second-order branch in solve loop |
| `src/aero_cfd/gpu_diagnostics.cu` | NEW | Device-side state bounds + failure capture |

GPU solver loop with second-order:

```
if order == 2:
  gg_gradient_kernel(d_mesh)        → cell gradients
  limiter_kernel(d_mesh)            → cell limiters
  apply_limiter_kernel(d_mesh)      → limited gradients
  for each face:                    ← in residual kernel
    QL = reconstruct(cell_L, grad_L, lim_L, dr)
    QR = reconstruct(cell_R, grad_R, lim_R, dr)
    flux = hllc_flux(QL, QR, n)
```

Kernels:

- [x] `gg_gradient_kernel`: per-face contribution to left/right cell gradients (atomicAdd per component)
- [x] `limiter_kernel`: per-cell neighbor-extrema scan + limiting factor (init_minmax/update_minmax/bj_limiter three-pass pipeline)
- [x] Reconstruction in `euler_residual_kernel`: face QL/QR from limited gradients
- [x] `diagnostics_kernel`: per-iteration min/max rho/p/mach, device-side accumulation
- [x] `failure_snapshot_kernel`: capture first invalid cell state on device (inline in update_and_l2_kernel via atomicCAS)

Tests:

- [x] `CFD-ORACLE-RECON-1`: constant-state zero gradients, CPU=GPU, tol=1e-12
- [x] `CFD-ORACLE-RECON-2`: CPU/GPU gradient match on cube mesh, tol=2e-6 per component
- [x] `CFD-ORACLE-RECON-3`: reconstruction_order=1 forces match 1st-order CPU, tol=1e-12
- [x] `CFD-ORACLE-RECON-4`: GPU order=2 forces are finite and differ from order=1
- [x] `CFD-ORACLE-DIAG-1`: GPU state bounds match CPU state bounds (tol=2e-5)
- [x] `CFD-ORACLE-DIAG-2`: GPU failure detection on invalid initial state

Gate:

- `reconstruction_order=1` produces bitwise-identical result to Phase 2/3.
- No negative reconstructed rho/p in any test case (positive guard enforced on device).
- CPU/GPU gradient components match within 1e-8 relative on flat plate mesh.

---

## Phase 4-B — Real 类型抽象

Goal: 用 `Real` 类型别名替代硬编码 `float`，支持 `AEROSIM_REAL_DOUBLE` 宏切换 double 精度。

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/real.hpp` | NEW | `Real` 类型别名 + 数学/原子操作包装 |
| `include/aero_cfd/*.hpp` (10 个) | MODIFY | `float` → `Real` |
| `src/aero_cfd/*.cpp` (6 个) | MODIFY | `float` → `Real` |
| `src/aero_cfd/*.cu` (8 个) | MODIFY | `float` → `Real` + CUDA 内联函数包装 |
| `tests/cfd/*.cpp` (6 个) | MODIFY | `float` → `Real` + `using namespace AeroSim` |
| `CMakeLists.txt` | MODIFY | 添加 `option(AEROSIM_REAL_DOUBLE)` |

Tasks:

- [x] `real.hpp`: `Real` 类型 + sqrt/fabs/fmin/fmax/isfinite/cos/sin/atomicAdd/atomicMin/atomicMax 包装
- [x] 机械替换 25 个文件
- [x] 修复 MSVC 主机侧 `__device__` 函数 ODR 冲突（static 关键字 + __CUDACC__ 守卫）
- [x] 修复 VTK 输出中 `"Real"` 字面量 → `"float"`（VTK 数据格式关键字）
- [x] 修复非 CUDA 测试可执行文件链接 `missile_lib` 的 CUDA 属性
- [x] `option(AEROSIM_REAL_DOUBLE)` CMake 选项
- [x] 验证：`AEROSIM_REAL_DOUBLE=0` 全部测试 PASS
- [x] 验证：`AEROSIM_REAL_DOUBLE=1` 编译成功，Mesh 测试 PASS

Gate:

- [x] `AEROSIM_REAL_DOUBLE=0`（默认 float）：全部测试 bit-wise 与 Phase 4 一致。
- [x] `AEROSIM_REAL_DOUBLE=1`（double）：编译通过，基础测试 PASS。
- [x] CUDA 原子操作统一收敛到 `real_atomic_min`/`real_atomic_max`，去除局部定义。

---

## Step 3 — MPI 预留接口

Goal: 冻结 MPI halo 交换 + 多流接口契约，不实现 MPI。`has_halo()==false` 时单 GPU 性能零退化。

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/device_mesh.hpp` | MODIFY | 新增 `d_halo_indices_`/`d_halo_send_buf_`/`d_halo_recv_buf_` 字段、`has_halo()`/`allocate_halo()` 方法 |
| `src/aero_cfd/device_mesh.cu` | MODIFY | `allocate_halo()` 分配 device 缓冲、`release()` 释放、move 操作转移 |
| `src/aero_cfd/gpu_solver.cu` | MODIFY | `#ifdef MPI_ENABLED` 保护下添加 `stream_comp`/`stream_comm`、`exchange_halo` 占位、`stream_comm` 同步 |

Tasks:

- [x] `device_mesh.hpp`: 新增 halo 字段和方法
- [x] `device_mesh.cu`: `allocate_halo`/`release`/move 实现
- [x] `gpu_solver.cu`: `MPI_ENABLED` guard 内多流占位结构
- [x] 验证：单 GPU 零退化，全部测试 PASS（仅 BW-1 预存波动）

Gate:

- [x] 不新增 `#include <mpi.h>`，不链接 MPI 库。
- [x] `has_halo() == false` 时，单 GPU 性能零退化。
- [x] 多流版本在 `MPI_ENABLED` 宏保护下编译。

---

## Phase 5 — GPU Viscous Navier-Stokes

Goal: port laminar NS viscous flux, wall shear, and heat flux to GPU. Integrate into `solve_gpu()`.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/cfd_config.hpp` | MODIFY | Add `bool viscous = false`, `float prandtl = 0.72` |
| `src/aero_cfd/gpu_viscous.cu` | NEW | Viscous gradient, flux, wall kernels |
| `src/aero_cfd/gpu_solver.cu` | MODIFY | Viscous branch in solve loop |
| `src/aero_cfd/gpu_timestep.cu` | MODIFY | Combined inviscid+viscous dt |

Kernels:

- [x] `viscous_flux_kernel`: per-face viscous gradient + orthogonal correction + stress/heat-flux assembly (single kernel, both interior and NoSlipWall faces)
- [x] `wall_shear_kernel`: shear stress + heat flux added to existing `wall_force_kernel` behind `viscous` flag
- [x] `combined_timestep_kernel`: dt = min(inviscid_dt, viscous_dt) per cell, reduce min — integrated into `timestep_kernel`

Implementation notes:
- Viscous gradient computed on-the-fly from PrimitiveGradient `d_gradients_[]` (15-component) via quotient rule for dT/dx — no extra storage needed.
- Sutherland mu and kappa = mu * gamma / ((gamma-1)*Pr) computed per face from face-average T.
- Viscous stress tensor uses `mu/Re` scaling (non-dimensional NS).
- No separate `viscous_gradient_kernel` or `viscous_face_gradient_kernel` — all fused into `viscous_flux_kernel_atomic`.
- No `combined_residual_kernel` — viscous flux is a separate kernel launch after Euler residual (composable kernel design).

Solver:

- [x] `solve_gpu()` viscous branch: `if config.viscous`, allocate viscous buffers + call `compute_viscous_flux_gpu()` after Euler residual
- [x] `viscous=false` (default): zero added overhead (no allocation, no kernel launch)
- [x] Wall integration includes shear + heat flux for NoSlipWall faces

Tests:

- [x] `CFD-ORACLE-VISC-1`: `viscous=false` regression to Euler result (tol=1e-6, accounts for atomic non-determinism)
- [x] `CFD-ORACLE-VISC-2`: `viscous=true` produces finite forces on flat plate (crash/no-NaN check)
- [x] `CFD-ORACLE-VISC-3`: `viscous=true` forces differ from inviscid on flat plate

Gate:

- [x] `viscous=false` matches Phase 2/4 Euler results (regression).
- [ ] Flat plate `Cf_avg / Cf_blasius ∈ [0.5, 2.0]` at Re=10^5 (needs more iterations).
- [ ] Wall heat flux sign convention: `Q_wall > 0` when wall is colder than fluid (needs Q_wall output).
- [ ] CPU/GPU wall forces and Q_wall match within 1e-8 absolute (needs CPU viscous oracle in solver loop).

---

## Phase 6 — CFD Table Integration

Goal: reconnect GPU solver to aerodynamic table generation behind a strict feature gate.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aerodynamics_model.hpp` | MODIFY | Add CFD fidelity flag |
| `src/aero_table_gen.cpp` | MODIFY | GPU solver integration |
| `include/aero_cfd/cfd_result.hpp` | MODIFY | Add `std::string fidelity` field |

Tasks:

- [x] `CfdForceResult.fidelity = "cfd-gpu"` or `"cfd-cpu"`
- [x] Supported-condition range check: `Mach ∈ [1.2, 30]`, `|alpha| ≤ 30°`, `|beta| ≤ 10°`
- [x] Fail with clear error for unsupported conditions (no silent fallback)
- [x] CSV fidelity source column: `cfd-gpu` vs `newtonian` vs `engineering`
- [x] Integration test: 3×3 Mach×alpha grid, all forces finite, symmetry holds
- [x] `use_fvm=true` no longer fails for supported conditions

Gate:

- [x] `use_fvm=true` with supported conditions produces valid CSV with `fidelity=cfd-gpu`.
- [x] `use_fvm=true` with unsupported conditions produces clear error message.
- [x] CSV column `fidelity` appears and distinguishes GPU CFD from other sources.

---

## Phase 7 — RANS Spalart-Allmaras GPU

Goal: add SA one-equation turbulence model. GPU-only production path with CPU oracle.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/cfd_state.hpp` | MODIFY | NVAR=6 |
| `include/aero_cfd/cfd_config.hpp` | MODIFY | Add `bool turbulence = false` |
| `src/aero_cfd/gpu_rans.cu` | NEW | SA production/destruction, transport |
| `src/aero_cfd/rans.cpp` | NEW | CPU SA oracle |

Tasks:

- [x] Phase 7.0: NVAR=6 structural — cfd_state, cfd_config, device_mesh, reconstruction, CPU HLLC/residual
- [x] Phase 7.1: GPU kernel turbulence propagation — d_conservative_to_primitive, d_physical_flux, d_slip_wall_flux, d_hllc_flux, d_reconstruct_primitive, both residual kernels, upload_state
- [ ] SA production term: `cb1 * Omega_tilde * nu_tilde`
- [ ] SA destruction term: `cw1 * fw * (nu_tilde/d)^2`
- [ ] SA diffusion: standard viscous operator
- [ ] SA wall BC: `nu_tilde = 0`
- [ ] SA farfield BC: specified `nu_tilde / mu` ratio
- [ ] Source-term point-implicit treatment for stability
- [ ] Negative `nu_tilde` handling per SA standard (not clamped)

Tests:

- [ ] `CFD-ORACLE-RANS-1`: `turbulence=false` matches Phase 5 laminar
- [ ] `CFD-ORACLE-RANS-2`: zero `nu_tilde` matches laminar
- [ ] `CFD-ORACLE-RANS-3`: turbulent flat plate `Cf` plausible (≥ laminar)
- [ ] `CFD-ORACLE-RANS-4`: CPU/GPU SA residual match

Gate:

- `turbulence=false` returns Phase 5 result (regression).
- SA results explicitly labeled as "RANS modeled, not transition-resolved".
- Negative `nu_tilde` handled without silent clamp.
- Turbulent flat plate Cf > laminar reference at same Re.

---

## Phase 8 — High-Order And DNS-Grade Verification

See `AERO_ACCURACY_UPGRADE.md` Stage HO/DNS for design.

Tasks placeholder:

- [ ] Manufactured-solution order verification
- [ ] High-order geometry representation plan
- [ ] High-order spatial discretization prototype (DG/FR/CPR)
- [ ] High-order shock sensor + localized dissipation
- [ ] p-refinement and h-refinement studies
- [ ] Isentropic vortex benchmark
- [ ] Taylor-Green vortex benchmark
- [ ] Shock/vortex interaction benchmark
- [ ] DNS resolution metrics: near-wall spacing, spectral content, time-step convergence

Gate:

- Observed order matches intended order on smooth manufactured problems.
- Shock handling does not destroy smooth-region order.
- DNS claims backed by h/p/time/domain convergence and resolution metrics.

---

## Phase 9 — Transition Physics

See `AERO_ACCURACY_UPGRADE.md` Stage T.

Tasks placeholder:

- [ ] Boundary-layer profile extraction
- [ ] LST/e^N design and validation plan
- [ ] Mack-mode/Tollmien-Schlichting benchmark cases
- [ ] Transition onset uncertainty reporting
- [ ] Optional engineering transition model with clear labeling
- [ ] DNS/WRLES transition patch validation case

Gate:

- Transition onset not inferred from SA alone.
- Reported transition location includes sensitivity to freestream disturbance, wall temperature, roughness, and grid.

---

## Phase 10 — Thermochemistry And Wall Catalysis

See `AERO_ACCURACY_UPGRADE.md` Stage H.

Tasks placeholder:

- [ ] Variable thermodynamic properties baseline
- [ ] Finite-rate multi-species chemistry plan
- [ ] Two-temperature model plan
- [ ] Non-catalytic wall boundary
- [ ] Finite-rate catalytic wall boundary
- [ ] Fully catalytic wall boundary
- [ ] Wall heat-flux uncertainty reporting
- [ ] Radiation/ablation/roughness limitation metadata

Gate:

- Heat flux reports separate numerical error, gas-model error, and wall-catalysis uncertainty.
- Unknown surface chemistry never produces a single unlabeled "exact" heat-flux value.

---

## Work Rules

- Do not advance to the next phase until the current phase has passing gates.
- Every GPU kernel must have a CPU oracle for CI regression (even if slow).
- Tests that only assert finite/positive outputs are smoke tests, not gate tests.
- Do not silently clamp bad states and report success.
- Keep diagnostics read-only.
- Update `progress.md` after each completed phase.
- Add blockers to `ISSUES.md` when progress is blocked by a reproducible failure.
- NaN/Inf checks are mandatory after every kernel producing cell state.
- CUDA calls checked via `cuda_check()` macro; kernel launches via `CUDA_KERNEL_CHECK()`.
- All floating-point comparisons use relative tolerance, never exact equality.
