# CPU sources (plain C++17 compiled by MSVC — no CUDA device code)
target_sources(missile_cpu PRIVATE
    panel/aero_table_gen.cpp
    cfd/cfd_residual.cpp
    cfd/gpu_communicator.cpp
    cfd/gpu_topology.cpp
    cfd/partition.cpp
    cfd/cfd_solver.cpp
    cfd/cfd_solver_gpu.cpp
    cfd/cuda_utils.cpp
    cfd/diagnostics.cpp
    cfd/mesh_metrics.cpp
    cfd/mesh_io_su2.cpp
    cfd/mesh_io_cgns.cpp
    cfd/mesh_validator.cpp
    cfd/rans.cpp
    cfd/reconstruction.cpp
    cfd/viscous.cpp
)

# GPU sources (compiled by nvcc, need separable compilation)
target_sources(missile_lib PRIVATE
    panel/aero_solver.cu
    cfd/cfd_residual_gpu.cu
    cfd/exchange_halo.cu
    cfd/fgmres_gpu.cu
    cfd/jacobian_free.cu
    cfd/lusgs_gpu.cu
    cfd/device_mesh.cu
    cfd/gpu_buffers.cu
    cfd/gpu_diagnostics.cu
    cfd/gpu_rans.cu
    cfd/gpu_solver.cu
    cfd/gpu_timestep.cu
    cfd/gpu_update.cu
    cfd/gpu_viscous.cu
    cfd/gpu_wall.cu
    cfd/reconstruction_gpu.cu
)
