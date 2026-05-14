include(GoogleTest)

add_executable(pgo_tests
    assembly/test_cpu_assembler.cpp
    base/test_assert.cpp
    dof/test_dof.cpp
    energy/test_assembled_energy.cpp
    energy/test_constant_force_energy.cpp
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
    integrator/test_backward_euler.cpp
    solver/test_solver.cpp
    storage/test_storage.cpp
)

target_link_libraries(pgo_tests
    PRIVATE
        pgo::core
        pgo::io
        GTest::gtest_main
)

gtest_discover_tests(pgo_tests)

add_executable(pgo_log_tests
    log/test_log.cpp
)

target_link_libraries(pgo_log_tests
    PRIVATE
        pgo::log
        GTest::gtest_main
)

gtest_discover_tests(pgo_log_tests)

if(PGO_ENABLE_SPDLOG)
    add_executable(pgo_spdlog_tests
        log/test_spdlog_sink.cpp
    )

    target_link_libraries(pgo_spdlog_tests
        PRIVATE
            pgo::log
            GTest::gtest_main
    )

    gtest_discover_tests(pgo_spdlog_tests)
endif()

if(PGO_BUILD_C_API AND TARGET pgo_c)
    add_subdirectory(c_api)
endif()
