include(GoogleTest)

add_executable(pgo_tests
    assembly/test_cpu_assembler.cpp
    base/test_assert.cpp
    dof/test_dof.cpp
    energy/test_assembled_energy.cpp
    energy/test_energy_concepts.cpp
    energy/test_energy_sum.cpp
    energy/test_mass_spring_local_energy_provider.cpp
    energy/test_reduced_energy.cpp
    geometry/test_rest_mesh.cpp
    io/test_obj_io.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    math/test_finite_difference.cpp
    solver/test_line_search.cpp
    solver/test_solver.cpp
    storage/test_storage.cpp
)

target_link_libraries(pgo_tests
    PRIVATE
        pgo::core
        GTest::gtest_main
)

gtest_discover_tests(pgo_tests)
