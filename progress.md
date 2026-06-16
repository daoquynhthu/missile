2026-06-16
- Created clean CFD rebuild baseline commit `d86393d`.
- Started Phase 1 mesh foundation under `include/aero_cfd/`, `src/aero_cfd/`, and `tests/cfd/`.
- Added explicit node/cell/face mesh model, boundary labels, quality report, structured cube generator with hex-cull wall classification, structured flat plate generator, metric computation, validation, and boundary area utility.
- Added `TestCfdMesh` CMake target.
- Verification: `cmake -B build` passed; `cmake --build build --target TestCfdMesh --config Release` passed; `build\bin\Release\TestCfdMesh.exe` passed 4/4.
- Committed Phase 1 as `8fb9c10`.
- Started Phase 2 Euler foundation with 5-variable state, primitive/conservative conversion, HLLC flux, slip-wall direct pressure flux, CPU first-order update skeleton, residual history, and `TestCfdEuler`.
- Verification: `cmake -B build` passed; `cmake --build build --target TestCfdEuler --config Release` passed; `TestCfdEuler.exe` passed 4/4; `ctest --test-dir build -C Release -R "Cfd(Mesh|Euler)" --output-on-failure` passed 2/2.
