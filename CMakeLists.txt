cmake_minimum_required(VERSION 3.18)
project(AeroHighPrecisionSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/shape_optimizer.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/tests/test_gravity.cu")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/tests/compare_atm.cpp")

# Create the core library
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

# Main Executable
add_executable(AeroSim src/main.cpp)
target_link_libraries(AeroSim PRIVATE missile_lib)

# AeroCalc Tool (For fast aerodynamic evaluation)
add_executable(AeroCalc src/aero_calc_tool.cpp)
target_link_libraries(AeroCalc PRIVATE missile_lib)

# Shape Optimizer (C++ based for speed)
add_executable(ShapeOptimizer src/shape_optimizer.cpp)
target_link_libraries(ShapeOptimizer PRIVATE missile_lib)

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

# Target properties
set_target_properties(AeroSim PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# --- Testing Section ---
enable_testing()

# 1. Propulsion Test
add_executable(TestPropulsion tests/test_propulsion.cpp)
target_link_libraries(TestPropulsion PRIVATE missile_lib)
add_custom_command(TARGET TestPropulsion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:TestPropulsion>
)
add_test(NAME PropulsionTest COMMAND TestPropulsion)

# 2. Aerodynamics Test
add_executable(TestAero tests/test_aero.cpp)
target_link_libraries(TestAero PRIVATE missile_lib)
add_custom_command(TARGET TestAero POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:TestAero>
)
add_test(NAME AeroTest COMMAND TestAero)

# 3. Guidance Test
add_executable(TestGuidance tests/test_guidance.cpp)
target_link_libraries(TestGuidance PRIVATE missile_lib)
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
target_link_libraries(TestAutopilot PRIVATE missile_lib)
add_test(NAME AutopilotTest COMMAND TestAutopilot)
set_target_properties(TestAutopilot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 5. Launch Test
add_executable(TestLaunch tests/test_launch.cpp)
target_link_libraries(TestLaunch PRIVATE missile_lib)
add_test(NAME LaunchTest COMMAND TestLaunch)
set_target_properties(TestLaunch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 6. Integrator Test
add_executable(TestIntegrator tests/test_integrator.cpp)
target_link_libraries(TestIntegrator PRIVATE missile_lib)
add_test(NAME IntegratorTest COMMAND TestIntegrator)
set_target_properties(TestIntegrator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
