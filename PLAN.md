# CFD Rebuild Plan

> Design: `AERO_ACCURACY_UPGRADE.md`
> Progress log: `progress.md`
> Active blockers: `ISSUES.md`

## Phase 0 — Clean Baseline

Goal: remove the contaminated CFD implementation and make the repository build without the old solver.

- [x] Remove old `cfd_solver.*`
- [x] Remove old `mesh_generator.*`
- [x] Remove old CFD/mesh tests
- [x] Disable old CFD override in aero table generation
- [x] Remove stale CFD test targets from CMake
- [x] Verify configure: `cmake -B build`
- [x] Verify focused retained targets: `missile_lib`, `TestAero`, `TestAeroTableGen`
- [ ] Verify full build: `cmake --build build --config Release --clean-first`
- [ ] Verify remaining non-CFD tests that still apply

Gate:

- No references to removed `CfdSolver` or old `TetMesh` remain in build targets.
- `use_fvm=true` fails explicitly instead of silently using stale code.

## Phase 1 — Mesh And Metrics Foundation

Goal: implement a clean mesh representation with explicit faces and validated metrics.

Files planned:

- `include/aero_cfd/cfd_mesh.hpp`
- `include/aero_cfd/cfd_result.hpp`
- `src/aero_cfd/mesh_metrics.cpp`
- `tests/cfd/test_cfd_mesh.cpp`

Tasks:

- [x] Define `CfdNode`, `CfdCell`, `CfdFace`, `BoundaryKind`
- [x] Implement structured cube mesh using hex-cull wall classification
- [x] Implement structured flat plate mesh
- [x] Compute face normals, areas, centers
- [x] Compute cell volumes, centers, `h_min`
- [x] Compute wall distance for wall-adjacent cells
- [x] Add mesh quality report
- [x] Add hard failure for negative/zero volume
- [x] Add tests for cube wall/farfield face counts
- [x] Add cube wall normal-area closure test
- [x] Add tests for flat plate wall area and first-layer height

Gate:

- Mesh tests pass.
- Cube wall classification does not depend on vertices landing at `±1`.
- Flat plate wall area equals `Lx * Ly` within tolerance.
- Cube wall `sum(n * area)` is near zero.

## Phase 2 — Euler First-Order Solver

Goal: build the smallest trustworthy finite-volume Euler solver.

Files planned:

- `include/aero_cfd/cfd_config.hpp`
- `include/aero_cfd/cfd_state.hpp`
- `include/aero_cfd/cfd_solver.hpp`
- `src/aero_cfd/cfd_solver.cu`
- `src/aero_cfd/flux_euler.cu`
- `src/aero_cfd/boundary_conditions.cu`
- `src/aero_cfd/timestep.cu`
- `src/aero_cfd/force_integrator.cu`
- `tests/cfd/test_cfd_euler.cu`

Tasks:

- [x] Define 5-variable Euler state only
- [x] Implement conservative-to-primitive with validity result
- [x] Implement HLLC flux
- [ ] Implement farfield boundary for supported supersonic cases
- [x] Implement slip wall direct pressure flux or validated reflected ghost
- [x] Implement global inviscid timestep
- [x] Implement first-order update
- [x] Implement body force integration over wall faces only
- [x] Add residual history output
- [x] Add uniform freestream preservation test
- [x] Add slip wall zero mass flux test
- [x] Add symmetric cube force test

Gate:

- Freestream remains unchanged.
- No negative rho/p in supported cases.
- Symmetric cube has near-zero lateral force.

## Phase 3 — Euler Reconstruction And Diagnostics

Goal: add second-order reconstruction and diagnostics without changing Stage E1 behavior when disabled.

Files planned:

- `include/aero_cfd/diagnostics.hpp`
- `src/aero_cfd/diagnostics.cpp`
- `src/aero_cfd/reconstruction.cu`
- `tests/cfd/test_cfd_diagnostics.cpp`

Tasks:

- [ ] Add Green-Gauss gradient
- [ ] Add least-squares gradient option if needed
- [ ] Add limiter
- [ ] Add positive reconstruction guard
- [ ] Add diagnostic levels OFF/BASIC/DETAILED/VERBOSE
- [ ] Add state bounds history
- [ ] Add dt limiter source history
- [ ] Add failure snapshot for first bad state
- [ ] Add VTK cell output

Gate:

- First-order mode remains bitwise or numerically unchanged.
- Reconstruction never creates negative reconstructed rho/p in tests.
- Injected bad state produces actionable diagnostic output.

## Phase 4 — Laminar Navier-Stokes

Goal: add viscous stress, heat conduction, no-slip wall, and thermal wall behavior.

Files planned:

- `src/aero_cfd/flux_viscous.cu`
- `tests/cfd/test_cfd_viscous.cu`

Tasks:

- [ ] Implement Sutherland viscosity
- [ ] Implement velocity and temperature gradients
- [ ] Implement orthogonal face-gradient correction
- [ ] Implement no-slip wall primitive
- [ ] Implement isothermal wall
- [ ] Implement adiabatic wall
- [ ] Implement viscous timestep
- [ ] Add wall shear and heat flux integration
- [ ] Add flat plate laminar `Cf` sanity test
- [ ] Add hot/cold wall heat flux sign test
- [ ] Add Euler regression with `viscous=false`

Gate:

- Flat plate `CX/Cf_ref` initially within `[0.5, 2.0]`.
- Heat flux sign changes correctly with wall temperature.
- `viscous=false` returns Euler result.

## Phase 5 — CFD Table Integration

Goal: reconnect the rebuilt solver to aerodynamic table generation behind a strict feature gate.

Tasks:

- [ ] Add CFD capability query
- [ ] Add supported-condition range
- [ ] Add CSV fidelity source column
- [ ] Fail or fallback for unsupported conditions
- [ ] Add integration test with small Mach/alpha grid

Gate:

- `use_fvm=true` no longer fails for supported conditions.
- Unsupported CFD requests do not silently produce results.

## Phase 6 — RANS And Beyond

Goal: add SA only after laminar NS is trustworthy.

Tasks:

- [ ] Extend state to 6 variables
- [ ] Add SA transport
- [ ] Add SA source treatment
- [ ] Add SA wall/farfield conditions
- [ ] Add turbulent flat plate tests
- [ ] Add regression to laminar when disabled

Gate:

- `turbulence=false` matches laminar NS.
- Turbulent flat plate `Cf` is physically plausible.

## Work Rules

- Do not add a new physics model until the previous phase has passing gates.
- Do not accept tests that only assert finite/positive outputs.
- Do not silently clamp bad states and report success.
- Keep diagnostics read-only.
- Update `progress.md` after completed work.
- Add blockers to `ISSUES.md` when progress is blocked by a reproducible failure.
