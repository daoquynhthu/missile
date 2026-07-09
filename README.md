# AeroHighPrecisionSim

High-fidelity ballistic simulation engine. C++17/CUDA, zero third-party CFD dependency.

## Features

- **6-DOF Simulation**: Rigid-body dynamics with RK4 integration, variable mass, ECEF/NED/body frames
- **Aerodynamics**: Newtonian panel method (GPU-accelerated), engineering component-buildup (DATCOM-style), FVM Euler/NS/RANS CFD solver (GPU production path with CPU oracle)
- **Atmosphere**: US Standard Atmosphere 1976 (≤86 km) + NRLMSISE-00 (Fortran, full altitude, host-only)
- **Gravity**: EGM2008 (4/12 order), J2/J3/J4 corrections
- **Guidance & Control**: Multi-phase guidance (boost/coast/glide/terminal), PID autopilot, PN guidance
- **CFD Solver**: Finite-volume method on GPU with face coloring for deterministic reduction, SA turbulence, 2nd-order reconstruction, viscous NS, zero-cudaMemcpy iteration loop, MPI halo interface (reserved)
- **Aero Table Generation**: Batch CFD/Newtonian table generation with fidelity tracking and condition-range enforcement

## Project Structure

```
├── AGENTS.md              # Workspace specification (AI agent instructions)
├── LICENSE                # GPL 3.0
├── README.md              # This file
├── CMakeLists.txt         # Build entry
├── docs/                  # Documentation
│   ├── AERO_ACCURACY_UPGRADE.md  # CFD accuracy upgrade architecture (read-only)
│   ├── REPO_SPEC.md       # Engineering specification
│   ├── PLAN.md            # Execution plan (active tasks)
│   ├── ISSUES.md          # Active blockers
│   ├── ARCHITECTURE.md    # Pipeline architecture
│   ├── PIPELINE.md        # Data flow documentation
│   ├── HANDOFF.md         # Project handoff documentation
│   ├── progress.md        # Progress log (append-only)
│   ├── ARCH_STABILIZE.md  # GPU architecture stabilization plan
│   └── DEVELOPMENT_LOG.md # Research & development log
├── src/                   # Source code
│   ├── infra/             # Math, utilities, CUDA helpers
│   ├── sim/               # Physics: coord, gravity, atmosphere, propulsion, dynamics, control
│   ├── aero/              # Aerodynamics: panel, engineering, cfd
│   └── config/            # Vehicle/scenario configuration
├── include/               # Public headers (parallel structure to src/)
├── app/                   # Executable entry points
│   ├── sim_main/          # Main simulation
│   ├── aero_calc/         # Command-line aero calculator
│   ├── aero_table_gen/    # Aero table generator
│   └── shape_optimizer/   # Shape optimization tool
├── examples/              # Validation cases
│   └── dart/              # Low-speed dart validation (Dart heritage)
├── data/                  # Input data (STL, tables, configs)
├── scripts/               # Python utilities
└── tests/                 # Test suite (custom TEST/FAIL/PASS, CTest)
```

## Build

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

### Requirements

- Windows 10+ (MSVC 2022)
- CUDA 12+ (sm_75, RDC ON)
- CMake 3.28+
- Eigen 3.4 (FetchContent, automatic)
- MinGW-W64 gfortran (for NRLMSISE-00, optional)

### Options

```bash
cmake -B build -DAEROSIM_REAL_DOUBLE=ON   # Double precision (default: float)
```

## Namespace Convention

All code lives under `aerosp::` with sub-namespaces mirroring the directory layout:

| Directory | Namespace | Description |
|-----------|-----------|-------------|
| `src/infra/math/` | `aerosp::infra::math` | Math utilities, constants |
| `src/sim/coord/` | `aerosp::sim::coord` | Coordinate transforms |
| `src/sim/control/` | `aerosp::sim::control` | Guidance & autopilot |
| `src/aero/cfd/` | `aerosp::aero::cfd` | FVM CFD solver |
| `src/aero/panel/` | `aerosp::aero::panel` | Panel method |
| `src/config/` | `aerosp::config` | Configuration |

## License

GNU General Public License v3.0. See `LICENSE`.
