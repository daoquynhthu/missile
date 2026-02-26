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

# Source files
set(GRAVITY_SOURCES
    src/gravity_model.cu
    src/coordinate_transform.cu
    src/atmosphere_model.cu
    src/dynamics_6dof.cu
    src/missile_config.cpp
)

# Gravity Model Library
add_library(gravity_lib STATIC ${GRAVITY_SOURCES})

# Link against NRLMSISE-00 import library
target_link_libraries(gravity_lib PUBLIC eigen CUDA::cudart "${CMAKE_SOURCE_DIR}/src/nrlmsise00.lib")

# Use SYSTEM to treat Eigen headers as system headers (suppresses warnings)
target_include_directories(gravity_lib SYSTEM PUBLIC ${eigen_SOURCE_DIR})
target_include_directories(gravity_lib PUBLIC include core)

# Suppress warnings in NVCC and MSVC
if(MSVC)
    target_compile_options(gravity_lib PUBLIC 
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819 /wd4068>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819 -Xcompiler=/wd4068 -Xcudafe --diag_suppress=20012>
    )
endif()

# Executable
add_executable(AeroSim src/main.cpp)
target_link_libraries(AeroSim gravity_lib)

# Copy DLL to output directory
add_custom_command(TARGET AeroSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/src/nrlmsise00.dll"
    $<TARGET_FILE_DIR:AeroSim>
)

# Test executable
add_executable(test_gravity tests/test_gravity.cu)
target_link_libraries(test_gravity gravity_lib)

# Comparison executable
add_executable(compare_atm tests/compare_atm.cpp)
target_link_libraries(compare_atm gravity_lib)

# Target properties
set_target_properties(AeroSim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(test_gravity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(compare_atm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
