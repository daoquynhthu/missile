set(TEST_CUDA_SOURCES
    test_propulsion.cpp
    test_aero.cpp
    test_guidance.cpp
    test_autopilot.cpp
    test_launch.cpp
    test_aero_table_gen.cpp
    compare_atm.cpp
    cfd/test_cfd_mesh.cpp
    cfd/test_cfd_euler.cpp
    cfd/test_cfd_diagnostics.cpp
    cfd/test_cfd_reconstruction.cpp
    cfd/test_cfd_viscous.cpp
    cfd/test_cfd_gpu.cpp
)
set_source_files_properties(${TEST_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

add_cuda_executable(TestPropulsion test_propulsion.cpp)
add_cuda_executable(TestAero test_aero.cpp)
add_cuda_executable(TestGuidance test_guidance.cpp)
add_cuda_executable(TestAutopilot test_autopilot.cpp)
add_cuda_executable(TestLaunch test_launch.cpp)
add_cuda_executable(TestAeroTableGen test_aero_table_gen.cpp)
add_cuda_executable(TestAeroViscous test_aero_viscous.cu)
add_cuda_executable(TestIntegrator test_integrator.cpp)
add_cuda_executable(TestCfdMesh cfd/test_cfd_mesh.cpp)
add_cuda_executable(TestCfdEuler cfd/test_cfd_euler.cpp)
add_cuda_executable(TestCfdDiagnostics cfd/test_cfd_diagnostics.cpp)
add_cuda_executable(TestCfdReconstruction cfd/test_cfd_reconstruction.cpp)
add_cuda_executable(TestCfdViscous cfd/test_cfd_viscous.cpp)
add_cuda_executable(TestCfdGpu cfd/test_cfd_gpu.cpp)
add_cuda_executable(TestCompareAtm compare_atm.cpp)

function(add_test_with_cwd name target)
  add_test(NAME ${name} COMMAND ${target})
  set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

add_test_with_cwd(PropulsionTest TestPropulsion)
add_test_with_cwd(AeroTest TestAero)
add_test_with_cwd(GuidanceTest TestGuidance)
add_test_with_cwd(AutopilotTest TestAutopilot)
add_test_with_cwd(LaunchTest TestLaunch)
add_test_with_cwd(AeroTableGenTest TestAeroTableGen)
add_test_with_cwd(AeroViscousTest TestAeroViscous)
add_test_with_cwd(IntegratorTest TestIntegrator)
add_test_with_cwd(CfdMeshTest TestCfdMesh)
add_test_with_cwd(CfdEulerTest TestCfdEuler)
add_test_with_cwd(CfdDiagnosticsTest TestCfdDiagnostics)
add_test_with_cwd(CfdReconstructionTest TestCfdReconstruction)
add_test_with_cwd(CfdViscousTest TestCfdViscous)
add_test_with_cwd(CfdGpuTest TestCfdGpu)
add_test_with_cwd(CompareAtmTest TestCompareAtm)
