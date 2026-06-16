cmake_minimum_required(VERSION 3.18)
project(AeroHighPrecisionSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
include(FetchContent)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(eigen)

# Define source files for the core library
file(GLOB_RECURSE LIB_SOURCES 
    "src/*.cpp" 
    "src/*.c" 
    "src/*.cu" 
    "src/aero_solver/*.cu"
)

# Remove main.cpp and tests from library sources
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/rm_dart_sim.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/rm_dart_mc.cu")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/shape_optimizer.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/aero_calc_tool.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/rm_dart_debug.cu")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/aero_solver/aero_solver_c_api.cu")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/tests/test_gravity.cu")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/tests/compare_atm.cpp")

# Core library (static)
add_library(missile_lib STATIC ${LIB_SOURCES})

# Include directories
target_include_directories(missile_lib PUBLIC 
    include 
    include/aero_solver
    ${eigen_SOURCE_DIR} # Add Eigen source dir
)

# Link dependencies to the library
target_link_libraries(missile_lib PUBLIC 
    CUDA::cudart 
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.lib"
)

# Suppress warnings in NVCC and MSVC
if(MSVC)
    target_compile_options(missile_lib PUBLIC 
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819 /wd4068>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819 -Xcompiler=/wd4068 -Xcudafe --diag_suppress=20012>
    )
endif()

# C API DLL for Python ctypes
add_library(aero_solver_c SHARED src/aero_solver/aero_solver_c_api.cu)
target_compile_definitions(aero_solver_c PRIVATE AERO_SOLVER_EXPORTS)
target_link_libraries(aero_solver_c PRIVATE missile_lib)
target_include_directories(aero_solver_c PRIVATE
    include include/aero_solver ${eigen_SOURCE_DIR}
)
set_target_properties(aero_solver_c PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET aero_solver_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:aero_solver_c>
)

# Main Executable
add_executable(AeroSim src/main.cpp)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(AeroSim PRIVATE missile_lib)
set_target_properties(AeroSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(AeroSim PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# AeroCalc Tool (For fast aerodynamic evaluation)
add_executable(AeroCalc src/aero_calc_tool.cpp)
set_source_files_properties(src/aero_calc_tool.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(AeroCalc PRIVATE missile_lib)
set_target_properties(AeroCalc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(AeroCalc PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Shape Optimizer (C++ based for speed)
add_executable(ShapeOptimizer src/shape_optimizer.cpp)
set_source_files_properties(src/shape_optimizer.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(ShapeOptimizer PRIVATE missile_lib)
set_target_properties(ShapeOptimizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ShapeOptimizer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# RM Dart Simulation
add_executable(RMDartSim src/rm_dart_sim.cpp)
set_source_files_properties(src/rm_dart_sim.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(RMDartSim PRIVATE missile_lib)
set_target_properties(RMDartSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RMDartSim PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# RM Dart GPU Monte Carlo
add_executable(RMDartMC src/rm_dart_mc.cu)
target_link_libraries(RMDartMC PRIVATE missile_lib)
set_target_properties(RMDartMC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RMDartMC PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Copy DLL to output directory
add_custom_command(TARGET AeroSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:AeroSim>
)
add_custom_command(TARGET AeroCalc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:AeroCalc>
)
add_custom_command(TARGET ShapeOptimizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:ShapeOptimizer>
)
add_custom_command(TARGET RMDartSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:RMDartSim>
)
add_custom_command(TARGET RMDartMC POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:RMDartMC>
)

# Target properties
set_target_properties(AeroSim PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# 8. RMDartDebug
add_executable(RMDartDebug src/rm_dart_debug.cu)
set_source_files_properties(src/rm_dart_debug.cu PROPERTIES LANGUAGE CUDA)
target_link_libraries(RMDartDebug PRIVATE missile_lib)
set_target_properties(RMDartDebug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RMDartDebug PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(RMDartDebug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

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
add_custom_command(TARGET TestPropulsion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:TestPropulsion>
)
add_custom_command(TARGET TestPropulsion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:missile_lib>
    $<TARGET_FILE_DIR:TestPropulsion>
)
add_test(NAME PropulsionTest COMMAND TestPropulsion)

# 2. Aerodynamics Test
add_executable(TestAero tests/test_aero.cpp)
set_source_files_properties(tests/test_aero.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestAero PRIVATE missile_lib)
set_target_properties(TestAero PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestAero PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(TestAero PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_custom_command(TARGET TestAero POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:TestAero>
)
add_test(NAME AeroTest COMMAND TestAero)

# 3. Guidance Test
add_executable(TestGuidance tests/test_guidance.cpp)
set_source_files_properties(tests/test_guidance.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestGuidance PRIVATE missile_lib)
set_target_properties(TestGuidance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestGuidance PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_custom_command(TARGET TestGuidance POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:TestGuidance>
)
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
set_target_properties(TestCfdMesh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestCfdMesh PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME CfdMeshTest COMMAND TestCfdMesh)
set_target_properties(TestCfdMesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 10. CFD Euler Foundation Test
add_executable(TestCfdEuler tests/cfd/test_cfd_euler.cpp)
set_source_files_properties(tests/cfd/test_cfd_euler.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdEuler PRIVATE missile_lib)
set_target_properties(TestCfdEuler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestCfdEuler PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME CfdEulerTest COMMAND TestCfdEuler)
set_target_properties(TestCfdEuler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 11. CFD Diagnostics Test
add_executable(TestCfdDiagnostics tests/cfd/test_cfd_diagnostics.cpp)
set_source_files_properties(tests/cfd/test_cfd_diagnostics.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestCfdDiagnostics PRIVATE missile_lib)
set_target_properties(TestCfdDiagnostics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestCfdDiagnostics PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME CfdDiagnosticsTest COMMAND TestCfdDiagnostics)
set_target_properties(TestCfdDiagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
