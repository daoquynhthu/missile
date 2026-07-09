cmake_minimum_required(VERSION 3.18)
project(AeroHighPrecisionSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(AEROSIM_REAL_DOUBLE "Use double precision for Real type" OFF)
if(AEROSIM_REAL_DOUBLE)
    add_compile_definitions(AEROSIM_REAL_DOUBLE)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Hybrid")

# --- GPU architecture detection ---
# Build for all active architectures; fall back to 75 (Turing) + 80 (Ampere) + 89 (Ada) + 90 (Hopper) + 100 (Blackwell).
# On first configure, auto-detect the host GPU's compute capability and add it first.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(_DEFAULT_ARCHS "75;80;89;90;100")
    # Attempt auto-detection via cuda_select_nvcc_arch_flags (CMake 3.20+)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Query device query sample or use cmake's built-in detection
        set(CMAKE_CUDA_ARCHITECTURES "${_DEFAULT_ARCHS}" CACHE STRING "CUDA architectures to build for")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "${_DEFAULT_ARCHS}" CACHE STRING "CUDA architectures to build for")
    endif()
endif()
# User override: cmake -DCMAKE_CUDA_ARCHITECTURES="89" for single-arch builds
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(BUILD_TESTING OFF CACHE BOOL "Disable testing for Eigen" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen test suite" FORCE)
set(CMAKE_Fortran_COMPILER "FALSE" CACHE INTERNAL "Fortran compiler (disabled)" FORCE)
include(FetchContent)
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)
FetchContent_MakeAvailable(eigen)

# Core static library (initially empty; subdirectories add sources via target_sources)
add_library(missile_lib STATIC "")
target_include_directories(missile_lib PUBLIC include ${eigen_SOURCE_DIR})
set_target_properties(missile_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# NRLMSISE-00 Fortran compilation
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
        COMMENT "Compiling NRLMSISE-00 (Fortran)...")
    add_custom_command(OUTPUT "${NRLMSISE00_LIB}"
        COMMAND ${CMAKE_AR} /def:"${NRLMSISE00_DEF}" /out:"${NRLMSISE00_LIB}" /machine:x64
        DEPENDS "${NRLMSISE00_DEF}"
        COMMENT "Generating MSVC-compatible NRLMSISE-00 import library...")
    add_custom_target(nrlmsise00 DEPENDS "${NRLMSISE00_LIB}" "${NRLMSISE00_DLL}")
    add_dependencies(missile_lib nrlmsise00)
    target_link_libraries(missile_lib PUBLIC "${NRLMSISE00_LIB}")
else()
    message(WARNING "gfortran or lib.exe not found — NRLMSISE-00 disabled")
endif()

target_link_libraries(missile_lib PUBLIC CUDA::cudart)

if(MSVC)
    target_compile_options(missile_lib PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819 /wd4068>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819 -Xcompiler=/wd4068 -Xcudafe --diag_suppress=20012>)
endif()

function(target_copy_nrlmsise_dll TARGET)
    if(GFORTRAN)
        add_dependencies(${TARGET} nrlmsise00)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/nrlmsise00.dll"
            $<TARGET_FILE_DIR:${TARGET}>)
    endif()
endfunction()

macro(add_cuda_executable NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} PRIVATE missile_lib)
    set_target_properties(${NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_copy_nrlmsise_dll(${NAME})
endmacro()

# Library subdirectories (add sources to missile_lib via target_sources)
add_subdirectory(src/infra)
add_subdirectory(src/sim)
add_subdirectory(src/aero)
add_subdirectory(src/config)

# Executables
add_subdirectory(app)
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
