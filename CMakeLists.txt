cmake_minimum_required(VERSION 3.10)
project(AeroHighPrecisionSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for precision and performance
if(MSVC)
    add_compile_options(/O2 /fp:precise /W4)
else()
    add_compile_options(-O3 -march=native -ffp-contract=fast)
endif()

# Directories
include_directories(include)
include_directories(core)

# Source files
set(SOURCES
    src/main.cpp
)

# Executable
add_executable(AeroSim ${SOURCES})

# Target properties
set_target_properties(AeroSim PROPERTIES
    OUTPUT_NAME "AeroSim"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
