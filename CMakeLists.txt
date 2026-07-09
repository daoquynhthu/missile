cmake_minimum_required(VERSION 3.18)
project(AeroHighPrecisionSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Real type precision: OFF=float (default), ON=double
option(AEROSIM_REAL_DOUBLE "Use double precision for Real type" OFF)
if(AEROSIM_REAL_DOUBLE)
    add_compile_definitions(AEROSIM_REAL_DOUBLE)
endif()

# Enable Relocatable Device Code (RDC) for cross-file __device__ calls
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Force MSVC runtime library (MD)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Force CUDA to use the same runtime library
# Using hybrid instead of shared to avoid deprecation warning
set(CMAKE_CUDA_RUNTIME_LIBRARY "Hybrid")

# NVCC architectures (sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

# Add relaxed constexpr flag for Eigen compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Find Eigen3 or Fetch it
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for Eigen" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen test suite" FORCE)
# Disable Fortran compiler detection before Eigen processes its blas/ subdirectory
set(CMAKE_Fortran_COMPILER "FALSE" CACHE INTERNAL "Fortran compiler (disabled, not needed)" FORCE)
include(FetchContent)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(eigen)

# Collect all library sources
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.c" "src/*.cu")

# Remove the C API source from library (it's a separate SHARED target)
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/aero/panel/aero_solver_c_api.cu")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/tests/test_gravity.cu")

# Core library (static)
add_library(missile_lib STATIC ${LIB_SOURCES})

# Include directories
target_include_directories(missile_lib PUBLIC 
    include 
    ${eigen_SOURCE_DIR}
)

# Enable RDC for the library (required for cross-file __device__ calls)
set_target_properties(missile_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# NRLMSISE-00 Fortran compilation via gfortran
# MinGW gfortran → DLL → .def → MSVC lib.exe → COFF .lib for MSVC linker
find_program(GFORTRAN gfortran)
if(GFORTRAN AND CMAKE_AR)
    set(NRLMSISE00_DEF "${CMAKE_BINARY_DIR}/nrlmsise00.def")
    set(NRLMSISE00_LIB "${CMAKE_BINARY_DIR}/nrlmsise00.lib")
    set(NRLMSISE00_DLL "${CMAKE_BINARY_DIR}/nrlmsise00.dll")
    add_custom_command(OUTPUT "${NRLMSISE00_DLL}" "${NRLMSISE00_DEF}"
        COMMAND ${GFORTRAN} -shared -o "${NRLMSISE00_DLL}"
            "${CMAKE_SOURCE_DIR}/src/sim/atmosphere/nrlmsise00.f"
            -Wl,--output-def,"${NRLMSISE00_DEF}"
            -static -std=legacy -Wno-unused-variable
        DEPENDS "${CMAKE_SOURCE_DIR}/src/sim/atmosphere/nrlmsise00.f"
        COMMENT "Compiling NRLMSISE-00 (Fortran)..."
    )
    add_custom_command(OUTPUT "${NRLMSISE00_LIB}"
        COMMAND ${CMAKE_AR} /def:"${NRLMSISE00_DEF}" /out:"${NRLMSISE00_LIB}" /machine:x64
        DEPENDS "${NRLMSISE00_DEF}"
        COMMENT "Generating MSVC-compatible NRLMSISE-00 import library..."
    )
    add_custom_target(nrlmsise00 DEPENDS "${NRLMSISE00_LIB}" "${NRLMSISE00_DLL}")
    add_dependencies(missile_lib nrlmsise00)
    target_link_libraries(missile_lib PUBLIC "${NRLMSISE00_LIB}")
else()
    message(WARNING "gfortran or lib.exe not found — NRLMSISE-00 disabled. Atmosphere::calculate() will fall back to USSA76 on host.")
endif()

# Helper: copy nrlmsise00.dll to target output directory
function(target_copy_nrlmsise_dll TARGET)
    if(GFORTRAN)
        add_dependencies(${TARGET} nrlmsise00)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/nrlmsise00.dll"
            $<TARGET_FILE_DIR:${TARGET}>
        )
    endif()
endfunction()

# Link dependencies to the library
target_link_libraries(missile_lib PUBLIC 
    CUDA::cudart 
)

# Suppress warnings in NVCC and MSVC
if(MSVC)
    target_compile_options(missile_lib PUBLIC 
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819 /wd4068>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819 -Xcompiler=/wd4068 -Xcudafe --diag_suppress=20012>
    )
endif()

# C API DLL for Python ctypes
add_library(aero_solver_c SHARED src/aero/panel/aero_solver_c_api.cu)
target_compile_definitions(aero_solver_c PRIVATE AERO_SOLVER_EXPORTS)
target_link_libraries(aero_solver_c PRIVATE missile_lib)
target_include_directories(aero_solver_c PRIVATE
    include ${eigen_SOURCE_DIR}
)
set_target_properties(aero_solver_c PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# Main Executable
add_executable(AeroSim app/sim_main/main.cpp)
target_link_libraries(AeroSim PRIVATE missile_lib)
set_target_properties(AeroSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(AeroSim PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# AeroCalc Tool (For fast aerodynamic evaluation)
add_executable(AeroCalc app/aero_calc/main.cpp)
target_link_libraries(AeroCalc PRIVATE missile_lib)
set_target_properties(AeroCalc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(AeroCalc PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Shape Optimizer (C++ based for speed)
add_executable(ShapeOptimizer app/shape_optimizer/main.cpp)
target_link_libraries(ShapeOptimizer PRIVATE missile_lib)
set_target_properties(ShapeOptimizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ShapeOptimizer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Aero Table Generator
add_executable(AeroTableGen app/aero_table_gen/main.cpp)
target_link_libraries(AeroTableGen PRIVATE missile_lib)
set_target_properties(AeroTableGen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(AeroTableGen PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(AeroTableGen PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# RM Dart Simulation
add_executable(RMDartSim examples/dart/sim.cpp)
target_include_directories(RMDartSim PRIVATE examples/dart)
target_link_libraries(RMDartSim PRIVATE missile_lib)
set_target_properties(RMDartSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RMDartSim PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# RM Dart GPU Monte Carlo
add_executable(RMDartMC examples/dart/rm_dart_mc.cu)
target_include_directories(RMDartMC PRIVATE examples/dart)
target_link_libraries(RMDartMC PRIVATE missile_lib)
set_target_properties(RMDartMC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RMDartMC PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Target properties
set_target_properties(AeroSim PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# 8. RMDartDebug
add_executable(RMDartDebug examples/dart/rm_dart_debug.cu)
target_link_libraries(RMDartDebug PRIVATE missile_lib)
set_target_properties(RMDartDebug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RMDartDebug PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(RMDartDebug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Copy nrlmsise00.dll to executable output directories
foreach(TGT AeroSim AeroCalc ShapeOptimizer RMDartSim RMDartMC RMDartDebug aero_solver_c)
    target_copy_nrlmsise_dll(${TGT})
endforeach()

# -----------------------------------------------------------------------------
# 5. Tests
# -----------------------------------------------------------------------------

enable_testing()

# 1. Propulsion Test
add_executable(TestPropulsion tests/test_propulsion.cpp)
set_source_files_properties(tests/test_propulsion.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestPropulsion PRIVATE missile_lib)
set_target_properties(TestPropulsion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestPropulsion PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(TestPropulsion PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_test(NAME PropulsionTest COMMAND TestPropulsion)

# 2. Aerodynamics Test
add_executable(TestAero tests/test_aero.cpp)
set_source_files_properties(tests/test_aero.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestAero PRIVATE missile_lib)
set_target_properties(TestAero PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestAero PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(TestAero PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_test(NAME AeroTest COMMAND TestAero)

# 3. Guidance Test
add_executable(TestGuidance tests/test_guidance.cpp)
set_source_files_properties(tests/test_guidance.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestGuidance PRIVATE missile_lib)
set_target_properties(TestGuidance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestGuidance PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME GuidanceTest COMMAND TestGuidance)

# Target properties
set_target_properties(AeroSim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(AeroCalc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(ShapeOptimizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(TestPropulsion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(TestAero PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(TestGuidance PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 4. Autopilot Test
add_executable(TestAutopilot tests/test_autopilot.cpp)
set_source_files_properties(tests/test_autopilot.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestAutopilot PRIVATE missile_lib)
set_target_properties(TestAutopilot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestAutopilot PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME AutopilotTest COMMAND TestAutopilot)
set_target_properties(TestAutopilot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 5. Launch Test
add_executable(TestLaunch tests/test_launch.cpp)
set_source_files_properties(tests/test_launch.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestLaunch PRIVATE missile_lib)
set_target_properties(TestLaunch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestLaunch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME LaunchTest COMMAND TestLaunch)
set_target_properties(TestLaunch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 6. Aero Table Generation Test
add_executable(TestAeroTableGen tests/test_aero_table_gen.cpp)
set_source_files_properties(tests/test_aero_table_gen.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestAeroTableGen PRIVATE missile_lib)
set_target_properties(TestAeroTableGen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestAeroTableGen PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME AeroTableGenTest COMMAND TestAeroTableGen)
set_target_properties(TestAeroTableGen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_custom_command(TARGET TestAeroTableGen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:TestAeroTableGen>/data"
)

# 7. Aero Viscous Test (van Driest II + GPU skin friction)
add_executable(TestAeroViscous tests/test_aero_viscous.cu)
set_source_files_properties(tests/test_aero_viscous.cu PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestAeroViscous PRIVATE missile_lib)
set_target_properties(TestAeroViscous PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestAeroViscous PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME AeroViscousTest COMMAND TestAeroViscous)
set_target_properties(TestAeroViscous PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_custom_command(TARGET TestAeroViscous POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:TestAeroViscous>/data"
)

# 8. Integrator Test (no CUDA device code — compiled as plain C++)
add_executable(TestIntegrator tests/test_integrator.cpp)
target_link_libraries(TestIntegrator PRIVATE missile_lib)
add_test(NAME IntegratorTest COMMAND TestIntegrator)
set_target_properties(TestIntegrator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 9. CFD Mesh Foundation Test
add_executable(TestCfdMesh tests/cfd/test_cfd_mesh.cpp)
set_source_files_properties(tests/cfd/test_cfd_mesh.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdMesh PRIVATE missile_lib)
add_test(NAME CfdMeshTest COMMAND TestCfdMesh)
set_target_properties(TestCfdMesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 10. CFD Euler Foundation Test
add_executable(TestCfdEuler tests/cfd/test_cfd_euler.cpp)
set_source_files_properties(tests/cfd/test_cfd_euler.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdEuler PRIVATE missile_lib)
add_test(NAME CfdEulerTest COMMAND TestCfdEuler)
set_target_properties(TestCfdEuler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 11. CFD Diagnostics Test
add_executable(TestCfdDiagnostics tests/cfd/test_cfd_diagnostics.cpp)
set_source_files_properties(tests/cfd/test_cfd_diagnostics.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdDiagnostics PRIVATE missile_lib)
add_test(NAME CfdDiagnosticsTest COMMAND TestCfdDiagnostics)
set_target_properties(TestCfdDiagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 12. CFD Reconstruction Test
add_executable(TestCfdReconstruction tests/cfd/test_cfd_reconstruction.cpp)
set_source_files_properties(tests/cfd/test_cfd_reconstruction.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdReconstruction PRIVATE missile_lib)
add_test(NAME CfdReconstructionTest COMMAND TestCfdReconstruction)
set_target_properties(TestCfdReconstruction PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 13. CFD Viscous Foundation Test
add_executable(TestCfdViscous tests/cfd/test_cfd_viscous.cpp)
set_source_files_properties(tests/cfd/test_cfd_viscous.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdViscous PRIVATE missile_lib)
add_test(NAME CfdViscousTest COMMAND TestCfdViscous)
set_target_properties(TestCfdViscous PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 14. CFD GPU Production Path Test
add_executable(TestCfdGpu tests/cfd/test_cfd_gpu.cpp)


set_source_files_properties(tests/cfd/test_cfd_gpu.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdGpu PRIVATE missile_lib)
set_target_properties(TestCfdGpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestCfdGpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME CfdGpuTest COMMAND TestCfdGpu)
set_target_properties(TestCfdGpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 15. Atmosphere model comparison test (USSA76 vs NRLMSISE-00)
add_executable(TestCompareAtm tests/compare_atm.cpp)
set_source_files_properties(tests/compare_atm.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCompareAtm PRIVATE missile_lib)
set_target_properties(TestCompareAtm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestCompareAtm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME CompareAtmTest COMMAND TestCompareAtm)
set_target_properties(TestCompareAtm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Copy nrlmsise00.dll to all test output directories
foreach(TGT TestPropulsion TestAero TestGuidance TestAutopilot TestLaunch
             TestAeroTableGen TestAeroViscous TestIntegrator
             TestCfdMesh TestCfdEuler TestCfdDiagnostics
             TestCfdReconstruction TestCfdViscous TestCfdGpu
             TestCompareAtm)
    target_copy_nrlmsise_dll(${TGT})
endforeach()
