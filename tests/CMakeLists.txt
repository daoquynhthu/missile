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

add_test(NAME PropulsionTest COMMAND TestPropulsion)
add_test(NAME AeroTest COMMAND TestAero)
add_test(NAME GuidanceTest COMMAND TestGuidance)
add_test(NAME AutopilotTest COMMAND TestAutopilot)
add_test(NAME LaunchTest COMMAND TestLaunch)
add_test(NAME AeroTableGenTest COMMAND TestAeroTableGen)
add_test(NAME AeroViscousTest COMMAND TestAeroViscous)
add_test(NAME IntegratorTest COMMAND TestIntegrator)
add_test(NAME CfdMeshTest COMMAND TestCfdMesh)
add_test(NAME CfdEulerTest COMMAND TestCfdEuler)
add_test(NAME CfdDiagnosticsTest COMMAND TestCfdDiagnostics)
add_test(NAME CfdReconstructionTest COMMAND TestCfdReconstruction)
add_test(NAME CfdViscousTest COMMAND TestCfdViscous)
add_test(NAME CfdGpuTest COMMAND TestCfdGpu)
add_test(NAME CompareAtmTest COMMAND TestCompareAtm)
