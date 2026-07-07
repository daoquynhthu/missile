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

- [ ] `timestep_kernel`: per-cell dt with block-reduction for min dt (shared memory tree reduction)
- [ ] `update_state_kernel`: vectorized float4/float access for component update
- [ ] `l2_norm_kernel`: per-cell sum of squares, block-reduction, single float result
- [ ] `state_validity_kernel`: finite + positive rho/p check, `atomicOr` failure flag
- [ ] `wall_force_kernel`: per-face pressure * area * normal, block-reduce force + moment
- [ ] `solve_gpu()`: loop with convergence/failure exit conditions
- [ ] `cfd_solver.cpp`: `solve()` uses GPU when `config.use_gpu==true`, else CPU
- [ ] Conditional compilation: `#ifndef __CUDACC__` stub for CPU-only builds

Gate:

- `solve_gpu()` on uniform freestream: L2 residual ≤ 1e-14 after 1 iteration (exact preservation).
- `solve_gpu()` on symmetric cube at alpha=beta=0: `|CY| < 1e-12`, `|CZ| < 1e-12`.
- Zero `cudaMemcpy` calls during iteration loop (verified with `cudaMemcpy` counter or profiling).
- GPU and CPU converge to same residual level on flat plate mesh (Mach=2, alpha=0).

---

## Phase 3 — CPU Oracle & Regression Verification

Goal: establish CPU as reference oracle for GPU validation. Every GPU operation has a CPU equivalent that can be compared in debug/CI mode.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero_cfd/cfd_config.hpp` | MODIFY | Add `bool cpu_oracle = false` |
| `src/aero_cfd/cfd_solver.cpp` | MODIFY | Oracle mode: both solves, assert match |
| `tests/cfd/test_cfd_gpu.cpp` | REWRITE | Real-mesh CPU/GPU equivalence tests |

Oracle mode (`cpu_oracle=true`):

```
solve():
  gpu_result = solve_gpu(...)
  if config.cpu_oracle:
    cpu_config = config; cpu_config.use_gpu = false
    cpu_result = solve_cpu(cpu_config)
    assert_equivalent(gpu_result, cpu_result, tolerance)
  return gpu_result
```

Tests:

- [ ] `CFD-ORACLE-EULER-1`: freestream preservation, CPU=GPU, tol=1e-12
- [ ] `CFD-ORACLE-EULER-2`: symmetric cube forces, CPU=GPU, tol=1e-10
- [ ] `CFD-ORACLE-EULER-3`: flat plate farfield-only, CPU=GPU, tol=1e-12
- [ ] `CFD-ORACLE-EULER-4`: residual convergence history, iter-by-iter relative match ≤ 1e-12
- [ ] `CFD-ORACLE-EULER-5`: wall force components, CPU=GPU, tol=1e-10 absolute
- [ ] `CFD-ORACLE-MESH-1`: device mesh counts match host mesh
- [ ] `CFD-ORACLE-BW-1`: GPU memory bandwidth ≥ 50% of theoretical (deviceQuery)

Gate:

- All oracle tests pass at stated tolerances.
- `cpu_oracle=true` in CI; `cpu_oracle=false` in production.
- GPU path is `CfdSolver` default. CPU requires explicit `use_gpu=false`.

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

- [ ] `gg_gradient_kernel`: per-face contribution to left/right cell gradients (atomicAdd per component)
- [ ] `limiter_kernel`: per-cell neighbor-extrema scan + limiting factor
- [ ] Reconstruction in `euler_residual_kernel`: face QL/QR from limited gradients
- [ ] `diagnostics_kernel`: per-iteration min/max rho/p/mach, device-side accumulation
- [ ] `failure_snapshot_kernel`: capture first invalid cell state on device

Tests:

- [ ] `CFD-ORACLE-RECON-1`: constant-state zero gradients, CPU=GPU, tol=1e-12
- [ ] `CFD-ORACLE-RECON-2`: CPU/GPU gradient match on flat plate, tol=1e-8 per component
- [ ] `CFD-ORACLE-RECON-3`: second-order disabled forces match first-order, tol=1e-12
- [ ] `CFD-ORACLE-RECON-4`: ordered=2 converged forces match CPU ordered=2
- [ ] `CFD-ORACLE-DIAG-1`: GPU state bounds match CPU state bounds
- [ ] `CFD-ORACLE-DIAG-2`: GPU failure detection matches CPU

Gate:

- `reconstruction_order=1` produces bitwise-identical result to Phase 2/3.
- No negative reconstructed rho/p in any test case (positive guard enforced on device).
- CPU/GPU gradient components match within 1e-8 relative on flat plate mesh.

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

- [ ] `viscous_gradient_kernel`: velocity gradients + temperature gradient from primitive gradients
- [ ] `viscous_face_gradient_kernel`: face-averaged gradient + orthogonal correction
- [ ] `viscous_flux_kernel`: per-face viscous flux assembly (stress + heat conduction)
- [ ] `combined_residual_kernel`: inviscid + viscous residual in single kernel
- [ ] `wall_shear_kernel`: wall shear stress + heat flux integration
- [ ] `combined_timestep_kernel`: dt = min(inviscid_dt, viscous_dt) per cell, reduce min

Solver:

- [ ] `solve_gpu()` viscous branch: `if config.viscous`, add viscous gradients + flux + combined dt
- [ ] `viscous=false` (default for now): Euler-only, zero added overhead
- [ ] Wall integration includes shear + heat flux → populate `Q_wall`, `Cf`, `St`

Tests:

- [ ] `CFD-ORACLE-VISC-1`: Sutherland normalization constant on device
- [ ] `CFD-ORACLE-VISC-2`: uniform flow zero viscous gradients, CPU=GPU
- [ ] `CFD-ORACLE-VISC-3`: CPU/GPU wall flux match on flat plate
- [ ] `CFD-ORACLE-VISC-4`: `viscous=false` regression to Phase 2/4 Euler result
- [ ] `CFD-ORACLE-VISC-5`: flat plate `Cf` within `[0.5, 2.0]` of Blasius at Re=10^5
- [ ] `CFD-ORACLE-VISC-6`: heat flux sign correct for isothermal hot/cold wall

Gate:

- `viscous=false` matches Phase 2/4 Euler results (regression).
- Flat plate `Cf_avg / Cf_blasius ∈ [0.5, 2.0]` at Re=10^5.
- Wall heat flux sign convention: `Q_wall > 0` when wall is colder than fluid.
- CPU/GPU wall forces and Q_wall match within 1e-8 absolute.

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

- [ ] `CfdForceResult.fidelity = "cfd-gpu"` or `"cfd-cpu"`
- [ ] Supported-condition range check: `Mach ∈ [1.2, 30]`, `|alpha| ≤ 30°`, `|beta| ≤ 10°`
- [ ] Fail with clear error for unsupported conditions (no silent fallback)
- [ ] CSV fidelity source column: `cfd-gpu` vs `newtonian` vs `engineering`
- [ ] Integration test: 3×3 Mach×alpha grid, all forces finite, symmetry holds
- [ ] `use_fvm=true` no longer fails for supported conditions

Gate:

- `use_fvm=true` with supported conditions produces valid CSV with `fidelity=cfd-gpu`.
- `use_fvm=true` with unsupported conditions produces clear error message.
- CSV column `fidelity` appears and distinguishes GPU CFD from other sources.

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

- [ ] Extend state from 5 to 6 variables (`rho*nu_tilde`)
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
