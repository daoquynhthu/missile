2026-06-16
- Created clean CFD rebuild baseline commit `d86393d`.
- Started Phase 1 mesh foundation under `include/aero_cfd/`, `src/aero_cfd/`, and `tests/cfd/`.
- Added explicit node/cell/face mesh model, boundary labels, quality report, structured cube generator with hex-cull wall classification, structured flat plate generator, metric computation, validation, and boundary area utility.
- Added `TestCfdMesh` CMake target.
- Verification: `cmake -B build` passed; `cmake --build build --target TestCfdMesh --config Release` passed; `build\bin\Release\TestCfdMesh.exe` passed 4/4.
