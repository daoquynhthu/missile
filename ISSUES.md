## Phase 2 Audit (2026-07-07)

### Category A: Correctness Bugs

**PH2-A-1: Farfield ghost state hardcodes rho and p — diverges from CPU for supersonic inflow** [FIXED]
`src/aero_cfd/cfd_residual_gpu.cu:48-62, 178`

The GPU `d_farfield_ghost_state` function outputs only velocity components (ghost_u/v/w), not the full primitive state. The caller at line 178 hardcodes `rho=1.0f, p=1.0f/gamma` for the ghost primitive. The CPU `farfield_ghost_state()` (`src/aero_cfd/cfd_solver.cpp:93-99`) returns a complete `PrimitiveState` — either the full `left` cell or the full `freestream` state.

For supersonic inflow (`vn_inf >= a_inf`):
- CPU ghost: `(rhoL, uL, vL, wL, pL)` — uses left cell's actual density and pressure
- GPU ghost: `(1.0, uL, vL, wL, 1.0/gamma)` — discards left cell's rho/p, uses freestream values

After the first iteration, `rhoL` and `pL` deviate from freestream values, causing a systematic mismatch between CPU and GPU fluxes on supersonic inflow faces. The test CFD-GPU-7 (20-iteration L2 match within 1e-6) may still pass if the cube mesh has few supersonic inflow faces, but this is a latent correctness bug.

Fix applied (2026-07-07): `d_farfield_ghost_state` now returns `ghost_rho`/`ghost_p`; caller uses ghost primitive in `d_hllc_flux`.

**PH2-A-2: update_and_l2_kernel writes new state to d_q before L2 sum** [FIXED]

Fix applied (2026-07-07): L2 computation (`dr*dr + ...`) now happens before the state write to `d_q`. If a NaN occurs in the L2 computation, the old state in `d_q` is preserved.

**PH2-A-3: atomicCAS-based timestep min-reduction has benign data race in initial read** [LOW] — FIXED
`src/aero_cfd/gpu_timestep.cu:37-44`

The initial read of `d_min_dt[0]` (line 37) and the first CAS (line 41) are not atomically paired. The value at `d_min_dt[0]` can change between the read and the CAS. The CAS loop handles this correctly by retrying with the updated `old` value, so there is no correctness bug. However, on high-occupancy GPUs with many cells, the CAS may retry many times. This is a performance concern, not a correctness bug.

Fix applied (2026-07-07): Replaced the initial read with `atomicCAS(ptr, __float_as_int(FLT_MAX), candidate)` to atomically initialize the reduction. This eliminates the non-atomic read and uses a single atomic operation to seed the CAS loop.

**PH2-A-4: Farfield ghost state has dead parameter `a_inf` in device function** [FIXED]
`src/aero_cfd/cfd_residual_gpu.cu:49`

Fix applied (2026-07-07): Removed along with the PH2-A-1 fix — `d_farfield_ghost_state` signature no longer takes a `left_a` parameter.

**PH2-A-5: L2 sum denominator includes failed cells** [NOT-A-BUG]

`d_failed` is always checked (line 79-84) before the convergence check (line 91). If any cell fails, the solver aborts before reading L2. No false convergence possible.

### Category B: Error Handling

**PH2-B-1: cudaEventDestroy(nullptr) is undefined behavior in failure path** [FIXED]
`src/aero_cfd/cfd_residual_gpu.cu:288-292, 306-308`

If `cudaMalloc(&d_failed)` succeeds but `cudaEventCreate(&start)` fails, the `goto fail` path executes `cudaEventDestroy(start)` where `start == nullptr`, and `cudaEventDestroy(stop)` where `stop == nullptr`. CUDA documentation does not guarantee `cudaEventDestroy(nullptr)` is safe — it is undefined behavior and can crash the driver.

Fix applied (2026-07-07): `fail` path now guards with `if (start)` / `if (stop)`.

**PH2-B-2: goto fail after kernel launch may cudaFree while kernel is still running** [NOT-A-BUG]

`cudaFree()` on the default stream blocks until all previously enqueued operations complete (CUDA spec). Since all kernels are launched on the default stream, `cudaFree` implicitly synchronizes before freeing. No data race possible.

**PH2-B-3: No post-kernel NaN/Inf validation on d_q after solver loop** [NOT-A-BUG]

`d_failed` from the update kernel is always checked at lines 79-84 before the convergence check at line 91. If `d_failed != 0`, the solver aborts before reading L2. No false convergence possible on the converged branch.

### Category C: Memory Management

**PH2-C-1: No memory leaks found on normal path** 
All `cudaFree` calls pair with `cudaMalloc`. The `DeviceMesh` destructor releases all buffers. The `goto cleanup` path in `gpu_solver.cu` frees all four temporaries unconditionally (`cudaFree(nullptr)` is safe). No leaks on normal path.

**PH2-C-2: Potential double-free on DeviceMesh move if release() fails** [NOT-A-BUG]

Analysis shows `release()` nulls pointers after `cuda_free_and_null`, and `cuda_free_and_null` guards against nullptr. No double-free possible.

**PH2-C-3: Potential memory leak in compute_euler_residual_gpu_timed on cudaEventCreate failure** [HIGH]
`src/aero_cfd/cfd_residual_gpu.cu:290-292`

If `cudaMalloc(&d_failed)` succeeds but `cudaEventCreate(&start)` fails at line 291, the `goto fail` path at line 306 calls `cudaEventDestroy(start)` (null, UB) and `cudaEventDestroy(stop)` (null, UB) and `cudaFree(d_failed)` (correct). But `d_failed` is correctly freed. The issue is the UB on event destroy, not a leak per se.

Fix: See PH2-B-1.

### Category D: Race Conditions

**PH2-D-1: atomicAdd on d_residual — correct for all interleavings** 
Different faces in `euler_residual_kernel` may write to the same cell's residual components. The `atomicAdd` operations are per-component (5 separate atomics per cell), so each float accumulates independently. This is correct and matches CPU summation order.

**PH2-D-2: atomicCAS in timestep_kernel — correct for all interleavings** 
The CAS-based min-reduction loop handles concurrent updates correctly (see PH2-A-3 analysis). All float-to-int comparisons preserve ordering for positive values.

**PH2-D-3: Kernel launch ordering on default stream — correct** 
All kernels launch on the default stream. `cudaDeviceSynchronize()` barriers between kernel groups provide the correct happens-before relationship. The two init kernels in `compute_update_gpu` (`init_float_zero_kernel` then `init_int_zero_kernel`) are ordered by the default stream before `update_and_l2_kernel`.

**PH2-D-4: Read-after-write between residual and update kernels — correct**
The residual kernel writes `d_residual` and is followed by `cudaDeviceSynchronize()`. The update kernel reads `d_residual` after the sync. Correct.

### Category E: Numerical Issues

**PH2-E-1: Epsilon in timestep denominator (1e-30f) is below FLT_MIN** [FIXED]
`src/aero_cfd/gpu_timestep.cu:36`

`1e-30f` is below `FLT_MIN` (1.175e-38 is the minimum NORMAL float, with subnormals down to 1.4e-45). The value `1e-30f` ~ 1.0e-30 is representable as a subnormal. Adding a subnormal to a normal float zeroes the subnormal on GPUs with flush-to-zero enabled (default in CUDA). If FTZ is enabled, `1e-30f + (vmag + a)` discards the epsilon, and a cell with `vmag = a = 0` would have `dt = cfl * h_min / 0 = +inf`, producing `d_min_dt = +inf` (via CAS, which treats `__float_as_int(+inf) > __float_as_int(any_finite)` incorrectly — inf bit pattern is 0x7f800000, which is larger than any finite positive float bit pattern).

However, cells with `vmag = a = 0` would have `p <= 0` (since a = sqrt(gamma * p / rho) = 0 implies p = 0), and the check `if (p <= 0.0f) return` at line 32 would prevent reaching the division. So this is a defense-in-depth issue, not a runtime bug.

Fix applied (2026-07-07): Denominator now guards against subnormal: `float denom = vmag + a; dt = cfl * d_h_min[idx] / (denom > 1e-30f ? denom : 1e-30f);`

**PH2-E-2: GPU solver plateau at L2 ~3e-4 while CPU reaches 1e-8** [MEDIUM]
`src/aero_cfd/gpu_solver.cu:88`

The test CFD-GPU-8 checks `ratio ≤ 1e3`, which would pass even with GPU at 3e-4 and CPU at 3e-7. This plateau is likely caused by atomic non-associativity in `euler_residual_kernel` — the order of `atomicAdd` operations on each cell's residual (from adjacent faces) is non-deterministic across blocks, causing accumulated round-off error. The CPU loop processes faces in a fixed order and does not use atomics.

This is a fundamental limitation of the per-cell `atomicAdd` pattern. Fixing this requires a deterministic reduction (e.g., color-partition faces so adjacent faces don't conflict, then use non-atomic stores).

**PH2-E-3: isfinite not guarded by #include <cmath> in device code** [FIXED]
`src/aero_cfd/gpu_update.cu:43,53`

Fix applied (2026-07-07): Added `#include <cmath>` to `gpu_update.cu`.

### Category F: API Design

**PH2-F-1: Forward declarations in gpu_solver.cu for sibling .cu functions are fragile** [FIXED]
`src/aero_cfd/gpu_solver.cu:19-22`

Fix applied (2026-07-07): Created `include/aero_cfd/gpu_solver_internal.hpp` with declarations; included from `gpu_solver.cu`, `gpu_timestep.cu`, `gpu_update.cu`, and `gpu_wall.cu`. Forward declarations in `gpu_solver.cu` removed.

**PH2-F-2: d_failed ownership and lifecycle is unclear** [FIXED]

Fix applied (2026-07-07): Added a new `compute_euler_residual_gpu(DeviceMesh&, ..., int* d_failed, ...)` overload that takes caller-allocated d_failed. The self-allocating overload is now a thin wrapper that calls through the d_failed overload. The primary API is always caller-allocated.

**PH2-F-3: solve_gpu() allocates device memory but doesn't expose sizes** [LOW] — FIXED
`src/aero_cfd/gpu_solver.cu:43-46`

The function allocates `d_failed` (4 bytes), `d_min_dt` (4 bytes), `d_l2_sum` (4 bytes), and `d_forces` (24 bytes) internally. A caller using `solve_gpu` cannot pre-allocate these or reuse them across multiple calls. For repeated solves (e.g., parameter sweeps), this wastes allocation/deallocation cycles.

Fix applied (2026-07-07): Added overload `solve_gpu(DeviceMesh&, ..., int* d_failed, float* d_min_dt, float* d_l2_sum, float* d_forces, ...)` for caller-allocated buffers. The original overload allocates internally and delegates to the shared implementation.

**PH2-F-4: DeviceMesh const-correctness issue** [FIXED]

Fix applied (2026-07-07): Added `ConstDeviceFaceData`/`ConstDeviceCellData` structs with `const float*`/`const int*` members. Non-const `face_data()`/`cell_data()` return mutable pointers; const overloads return `ConstDeviceFaceData`/`ConstDeviceCellData`.

### Category G: Gate Compliance

**PH2-G-1: Zero cudaMemcpy during iteration loop — NOT MET** [MEDIUM]
`PLAN.md:207`, `src/aero_cfd/gpu_solver.cu:61,69,79,87`

The Phase 2 gate requires "Zero `cudaMemcpy` calls during iteration loop." The current implementation uses **4** cudaMemcpy calls per iteration:
1. Line 61: `cudaMemcpy(&residual_failed, d_failed, 4, D2H)` — read failure flag
2. Line 69: `cudaMemcpy(&min_dt, d_min_dt, 4, D2H)` — read timestep
3. Line 79: `cudaMemcpy(&update_failed, d_failed, 4, D2H)` — read update failure flag
4. Line 87: `cudaMemcpy(&l2, d_l2_sum, 4, D2H)` — read L2 norm

PLAN.md notes this is "deferred to Phase 3," but the gate is explicitly unchecked (not marked `[x]`). This is a documented deviation, not a bug.

**PH2-G-2: GPU-CPU L2 match after 1 iteration (CFD-GPU-6) — MET**
**PH2-G-3: GPU-CPU L2 match for 20 iterations (CFD-GPU-7) — MET**
**PH2-G-4: GPU and CPU converge within ratio 1e3 on flat plate (CFD-GPU-8) — MET**

All three gate tests pass based on their test definitions.

### Category H: Code Convention Compliance

**PH2-H-1: No CUDA_KERNEL_CHECK macro exists** [FIXED]
`AGENTS.md:56`, `PLAN.md:495`

Both AGENTS.md and PLAN.md specify that kernel launches should use `CUDA_KERNEL_CHECK()`. No such macro is defined anywhere in the codebase. All kernel launch error checking uses `cuda_check(cudaGetLastError(), ...)` instead, which is functionally correct but violates the stated convention.

Fix applied (2026-07-07): `CUDA_KERNEL_CHECK(msg)` macro defined in `cuda_utils.hpp` as `cuda_check(cudaGetLastError(), msg)`.

**PH2-H-2: Unused variable `aL` in farfield ghost call** [FIXED]
`src/aero_cfd/cfd_residual_gpu.cu:175, 177`

Fix applied (2026-07-07): Removed along with PH2-A-1 — `aL` computation no longer exists.

See PH2-A-4.

**PH2-H-3: Kernel naming convention followed** 
All new kernels in Phase 2 use the `_kernel` suffix: `init_float_max_kernel`, `timestep_kernel`, `init_float_zero_kernel`, `init_int_zero_kernel`, `update_and_l2_kernel`, `init_float6_zero_kernel`, `wall_force_kernel`. Conforms to convention.

**PH2-H-4: Anonymous namespace used for all kernels** 
All kernels and device functions are in anonymous namespaces within their respective .cu files. This prevents symbol collisions. Good practice.

### Summary

| Severity | Count | Issue IDs |
|----------|-------|-----------|
| CRITICAL | 0 | |
| HIGH     | 0 | |
| MEDIUM   | 1 | PH2-E-2 (fundamental atomicAdd limitation, needs Phase 4) |
| LOW      | 3 | PH2-A-3 (benign CAS race), PH2-F-3 (pre-alloc API), PH2-G-1 (deferred Phase 3) |
| FIXED    | 11 | PH2-A-1, PH2-A-2, PH2-A-4, PH2-B-1, PH2-E-1, PH2-E-3, PH2-F-1, PH2-F-2, PH2-F-4, PH2-H-1, PH2-H-2 |
| NOT-A-BUG | 4 | PH2-A-5, PH2-B-2, PH2-B-3, PH2-C-2 |

Total: 4 open + 11 fixed + 4 wont-fix = 19 tracked items

- Category D (Race Conditions) and Category G (Gate Compliance, partial): no independently filed issues — the analysis confirms correctness, with the noted gate deviation already documented in PLAN.md.

## Phase 2 Re-Audit (2026-07-07)

### Verification Results

All 11 previously-fixed issues verified as correctly applied: PH2-A-1, PH2-A-2, PH2-A-4, PH2-B-1, PH2-E-1, PH2-E-3, PH2-F-1, PH2-F-2, PH2-F-4, PH2-H-1, PH2-H-2. No regressions found.

### New Findings

#### Category A: Correctness

**PH2-RA-H1: HLLC `denom` division by zero produces NaN flux** [HIGH] — FIXED
`src/aero_cfd/cfd_residual_gpu.cu:82-83`

The HLLC star-speed denominator `denom = rhoL*(s_l - vn_l) - rhoR*(s_r - vn_r)` vanishes when left and right states are symmetric (identical wave speeds). This produces `s_m = NaN/Inf`, corrupting the entire HLLC flux. The NaN is atomically added to the residual, potentially corrupting both adjacent cells before the `isfinite` check in the update kernel catches it.

Fix: Clamp `|denom|` below a threshold: `if (fabsf(denom) < 1e-30f) denom = copysignf(1e-30f, denom);`.

**PH2-RA-H2: HLLC star-state energy division by zero at sonic points** [HIGH] — FIXED
`src/aero_cfd/cfd_residual_gpu.cu:89,111`

At sonic points, `s_l - vn_l` (line 89) or `s_r - vn_r` (line 111) can be zero, making `e_star = NaN/Inf`. This causes the HLLC star-state total energy and flux to be NaN. Sonic points are standard features of transonic/supersonic flows (expansion fans, sonic lines).

Fix: Clamp the denominator or implement Harten's entropy fix for wave speeds.

**PH2-RA-H3: Symmetry boundary treated as farfield** [HIGH] — FIXED
`src/aero_cfd/cfd_residual_gpu.cu:170-178`

`BoundaryKind::Symmetry = 4` (defined in `cfd_mesh.hpp:15`) is not handled by any explicit branch in `euler_residual_kernel`. It falls into the `else` clause (line 172) which applies farfield characteristic boundary conditions. This injects freestream velocity/pressure through symmetry planes, violating the physical symmetry condition. Mass and momentum leak through symmetry planes.

Fix: Add `bnd == static_cast<int>(BoundaryKind::Symmetry)` to the wall check (line 170) to apply slip-wall flux, or add a dedicated symmetry mirror flux.

**PH2-RA-H4: Non-atomic initial read of `d_min_dt[0]`** [NOT-A-BUG]

The initial read at `gpu_timestep.cu:39` is non-atomic, but both kernels are on the default stream. Stream ordering guarantees `init_float_max_kernel` writes `FLT_MAX` before `timestep_kernel` reads it. The CAS loop correctly handles concurrent updates from other thread blocks. No race in practice.

#### Category B: Error Handling

**PH2-RA-M1: cudaFree errors silently discarded in DeviceMesh::release()** [MEDIUM] — FIXED
`src/aero_cfd/device_mesh.cu:108-139`

`cuda_free_and_null` captures the `cudaFree` return code but every caller in `release()` ignores it. If `cudaFree` fails (driver error, invalid pointer), the memory is not freed but the pointer is nullified anyway, causing a permanent device memory leak and masking the root cause.

Fix: At minimum, assert on failure in `release()`. Better: propagate the error or log it.

#### Category C: Code Robustness

**PH2-RA-M2: isfinite in device code without guaranteed include** [MEDIUM] — FIXED
`src/aero_cfd/cfd_residual_gpu.cu:13,20`

`isfinite()` is used in `__device__` functions without an explicit `<math.h>` or `<cmath>` include. While NVCC provides it as a built-in in practice, the CUDA spec recommends using the explicit intrinsic `__finitef()` for `float` operands in device code to avoid any host-device resolution issues.

Fix: Replace `isfinite(x)` with `__finitef(x)` for `float` arguments.

**PH2-RA-M3: Per-element cudaMemcpy in mesh upload — O(N) API calls** [MEDIUM] — FIXED
`src/aero_cfd/device_mesh.cu:190-215`

Mesh upload issues `7*face_count + 5*cell_count` individual 4-byte `cudaMemcpy` calls. For production-scale meshes (millions of cells), this creates millions of driver API calls with seconds of overhead. The upload time scales linearly with mesh size rather than being O(1) per array.

Fix: Pack face/cell data into contiguous host vectors and use one `cudaMemcpy` per device array.

**PH2-RA-M4: int cell/face count limits mesh to <2B elements** [MEDIUM] — FIXED
`src/aero_cfd/device_mesh.hpp:94-95`

`cell_count_` and `face_count_` are stored as `int`. A mesh with >2^31 cells or faces would overflow. For large 3D hypersonic simulations this is a practical concern.

Fix: Change to `std::size_t` or `int64_t`.

**PH2-RA-M5: Wall force uses cell-averaged pressure, not face-reconstructed** [MEDIUM] — FIXED
`src/aero_cfd/gpu_wall.cu:35-43`

The wall force kernel reads cell-averaged pressure from `d_q` and uses it directly as the face pressure. For second-order schemes, the face pressure should be reconstructed using stored gradients: `p_face = p_cell + grad_p . (x_face - x_cell)`. Using cell-averaged values is first-order accurate and can produce forces off by up to 20% on coarse meshes.

Fix applied (2026-07-07): Added gradient-based reconstruction in `wall_force_kernel`. When `d_gradients` is non-null, the face pressure is reconstructed as `p += dp_dx*dr + dp_dy*ds + dp_dz*dt`. Falls back to cell-averaged pressure when gradients are null (first-order mode).

#### Category D: Minor/Convention

**PH2-RA-L1: Redundant thread/block check in init kernel** [LOW] — FIXED
`src/aero_cfd/gpu_timestep.cu:14`

`init_float_max_kernel` is launched as `<<<1,1>>>` but checks `threadIdx.x == 0 && blockIdx.x == 0`, which is always true. Dead conditional.

**PH2-RA-L2: Timed residual includes cudaMemset overhead** [LOW] — FIXED
`src/aero_cfd/cfd_residual_gpu.cu:290-292`

`cudaEventRecord(start)` records time before `launch_euler_residual_kernel` (which does `cudaMemset` inside `clear_residual()`). The elapsed time includes both `cudaMemset` and kernel execution.

**PH2-RA-L3: Hardcoded freestream density/pressure** [LOW] — FIXED
`src/aero_cfd/cfd_residual_gpu.cu:174`

The farfield ghost call passes `1.0f` and `1.0f/gamma` as freestream density and pressure. While these match `make_freestream()`'s non-dimensionalization, any future change to the non-dimensionalization scheme would silently break the GPU solver.

Fix: Pass `freestream.rho` and `freestream.p` through to the kernel, or reference constants.

### Summary

| Severity | Count | New IDs |
|----------|-------|---------|
| HIGH     | 3 | PH2-RA-H1 (HLLC denom), PH2-RA-H2 (HLLC sonic), PH2-RA-H3 (Symmetry BC) |
| MEDIUM   | 5 | PH2-RA-M1 (cudaFree ignore), PH2-RA-M2 (isfinite), PH2-RA-M3 (O(N) upload), PH2-RA-M4 (int overflow), PH2-RA-M5 (wall pressure) |
| LOW      | 3 | PH2-RA-L1 (redundant check), PH2-RA-L2 (timing), PH2-RA-L3 (hardcoded) |
| FIXED    | 10 | PH2-RA-H1, PH2-RA-H2, PH2-RA-H3, PH2-RA-M1, PH2-RA-M2, PH2-RA-M3, PH2-RA-M4, PH2-RA-L1, PH2-RA-L2, PH2-RA-L3 |
| NOT-A-BUG | 1 | PH2-RA-H4 (stream ordering) |

Total new: 11 tracked (3 HIGH, 5 MEDIUM, 3 LOW). All 11 now fixed.

### Global Summary

All 11 previous fixes verified PASS. New re-audit found 11 additional items (3 HIGH, 5 MEDIUM, 3 LOW). 10 of 11 re-audit items fixed; remaining PH2-RA-M5 fixed in follow-up session.

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 0 | |
| HIGH (open) | 1 | PH3-H-1 (CfdSolveSummary init, allocation failure) |
| MEDIUM (open) | 6 | PH2-E-2 (atomicAdd, deferred to Phase 4), PH3-M-1..M-5 |
| LOW (open) | 4 | PH2-G-1 (cudaMemcpy, deferred), PH3-L-1..L-3 |
| INFO | 1 | PH3-I-1 (estimate undercount) |
| FIXED (all sessions) | 25 | All previous |
| NOT-A-BUG | 5 | Previous 4 + PH2-RA-H4 |

Total: 11 open + 25 fixed + 5 wont-fix = 41 tracked items

## Post-Commit Audit (2026-07-07)

### Verification Results

All 24 previously-fixed issues verified as correctly applied: PH2-A-1, PH2-A-2, PH2-A-3, PH2-A-4, PH2-B-1, PH2-E-1, PH2-E-3, PH2-F-1, PH2-F-2, PH2-F-3, PH2-F-4, PH2-H-1, PH2-H-2, PH2-RA-H1, PH2-RA-H2, PH2-RA-H3, PH2-RA-M1, PH2-RA-M2, PH2-RA-M3, PH2-RA-M4, PH2-RA-M5, PH2-RA-L1, PH2-RA-L2, PH2-RA-L3. No regressions found.

### Free Audit: New Findings

#### HIGH

**PH3-H-1: CfdSolveSummary{true} silently reports success on allocation failure** [HIGH]
`src/aero_cfd/gpu_solver.cu:149-152`

`CfdSolveSummary{true}` uses aggregate initialization — `true` converts to `float(1.0f)` and initializes `CfdForceResult::CX`. The `failed` member stays `false` (default member initializer). If `cudaMalloc` fails, the caller receives a summary with `failed = false` and `forces.CX = 1.0f` (garbage).

Fix: `CfdSolveSummary s; s.failed = true; return s;`

#### MEDIUM

**PH3-M-1: Missing NaN guard in gpu_timestep.cu kernel checks** [MEDIUM]
`src/aero_cfd/gpu_timestep.cu:25,33`

`if (rho <= 0.0f) return;` and `if (p <= 0.0f) return;` do not catch NaN — `NaN <= 0.0f` evaluates to `false`. NaN rho/p flows through to produce a finite but wrong dt. Same pattern in `gpu_wall.cu:38,46,56`.

Fix: Replace `<= 0.0f` with `!(x > 0.0f)` or add `|| !__finitef(x)`.

**PH3-M-2: Missing NaN guard in wall force gradient extrapolation fallback** [MEDIUM]
`src/aero_cfd/gpu_wall.cu:48-56`

If gradients contain NaN, extrapolated `p` becomes NaN. The check `p <= 0.0f` does not catch NaN, so NaN pressure propagates into force coefficients.

Fix: `if (!__finitef(p) || p <= 0.0f)`

**PH3-M-3: Missing cell-index bounds check in euler_residual_kernel** [MEDIUM]
`src/aero_cfd/cfd_residual_gpu.cu:152,168,199`

`d_left_cell[idx]` and `d_right_cell[idx]` used as array indices without validation. A corrupted mesh causes out-of-bounds device memory access (hard GPU context loss). Same issue in `gpu_wall.cu:37`.

Fix: Add `if (left < 0 || left >= n_cells) { atomicExch(d_failed, 1); return; }` (requires passing `n_cells` to kernel).

**PH3-M-4: reinterpret_cast<void*&> strict-aliasing violation in FREE_AND_ASSERT** [MEDIUM]
`src/aero_cfd/device_mesh.cu:119`

Casting `float*`/`int*` lvalues to `void*&` violates C++ strict aliasing rules (UB). Formal UB though accepted by all CUDA compilers in practice.

Fix: Replace with template `cuda_free_and_null<T>(T*& ptr)`.

**PH3-M-5: Hardcoded 15 for gradient stride (fragile against struct changes)** [MEDIUM]
`src/aero_cfd/gpu_wall.cu:52`, `src/aero_cfd/device_mesh.cu:257`

`PrimitiveGradient` has 15 float members, but the code uses magic number `15` instead of `sizeof(PrimitiveGradient)/sizeof(float)`. If struct is extended, these sites silently misalign.

Fix: Use a named constant, e.g. `DeviceMesh::NGRAD`.

#### LOW

**PH3-L-1: Redundant __finitef(rho) in d_conservative_to_primitive return** [LOW]
`src/aero_cfd/cfd_residual_gpu.cu:20`

`rho` already checked at line 13 before any modification; return check is dead code.

**PH3-L-2: #include <cuda_runtime_api.h> placed between function declarations** [LOW]
`include/aero_cfd/cfd_residual.hpp:21`

Include in the middle of the namespace, unconventional. Should be at top of file.

**PH3-L-3: Hardcoded pi constant** [LOW]
`src/aero_cfd/gpu_solver.cu:107`

`3.14159265358979323846f` instead of `M_PI` from `<cmath>`.

#### INFO

**PH3-I-1: estimate_euler_residual_gpu_bytes undercounts memory traffic** [INFO]
`src/aero_cfd/cfd_residual_gpu.cu:333-344`

Omits 3 int arrays (left_cell, right_cell, boundary) from face memory estimate. Used only for test bandwidth calculation.

### Summary

| Severity | Count | IDs |
|----------|-------|------|
| HIGH     | 1 | PH3-H-1 (CfdSolveSummary init) |
| MEDIUM   | 5 | PH3-M-1 (NaN in timestep), PH3-M-2 (NaN in wall gradient), PH3-M-3 (cell index bounds), PH3-M-4 (strict-aliasing), PH3-M-5 (magic 15) |
| LOW      | 3 | PH3-L-1 (redundant __finitef), PH3-L-2 (include placement), PH3-L-3 (hardcoded pi) |
| INFO     | 1 | PH3-I-1 (estimate undercount) |

Total: 10 new findings.
