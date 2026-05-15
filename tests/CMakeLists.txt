include(GoogleTest)

function(pgo_copy_runtime_dlls target_name)
    if(WIN32)
        set(pgo_mkl_runtime_artifacts)
        if(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "MKL" AND DEFINED ENV{MKLROOT})
            cmake_path(CONVERT "$ENV{MKLROOT}" TO_CMAKE_PATH_LIST pgo_mkl_root NORMALIZE)
            set(pgo_mkl_runtime_dirs
                "${pgo_mkl_root}/redist/intel64"
                "${pgo_mkl_root}/redist/intel64/mkl"
                "${pgo_mkl_root}/bin"
                "${pgo_mkl_root}/bin/intel64"
                "${pgo_mkl_root}/lib/intel64"
            )
            set(pgo_mkl_runtime_names
                mkl_core.2.dll
                mkl_def.2.dll
                mkl_intel_lp64.2.dll
                mkl_intel_thread.2.dll
            )
            get_filename_component(pgo_oneapi_root "${pgo_mkl_root}/../.." ABSOLUTE)
            file(GLOB pgo_mkl_openmp_candidates
                "${pgo_oneapi_root}/compiler/latest/bin/libiomp5md.dll"
                "${pgo_oneapi_root}/compiler/latest/redist/intel64/compiler/libiomp5md.dll"
            )

            foreach(pgo_mkl_runtime_dir IN LISTS pgo_mkl_runtime_dirs)
                foreach(pgo_mkl_runtime_name IN LISTS pgo_mkl_runtime_names)
                    if(EXISTS "${pgo_mkl_runtime_dir}/${pgo_mkl_runtime_name}")
                        list(APPEND pgo_mkl_runtime_artifacts "${pgo_mkl_runtime_dir}/${pgo_mkl_runtime_name}")
                    endif()
                endforeach()
            endforeach()
            list(APPEND pgo_mkl_runtime_artifacts ${pgo_mkl_openmp_candidates})
            list(REMOVE_DUPLICATES pgo_mkl_runtime_artifacts)
        endif()

        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy -t
                "$<TARGET_FILE_DIR:${target_name}>"
                "$<TARGET_RUNTIME_DLLS:${target_name}>"
                ${pgo_mkl_runtime_artifacts}
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

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
    parallel/test_parallel_for.cpp
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

pgo_copy_runtime_dlls(pgo_tests)
gtest_discover_tests(pgo_tests)

add_executable(pgo_log_tests
    log/test_log.cpp
)

target_link_libraries(pgo_log_tests
    PRIVATE
        pgo::log
        GTest::gtest_main
)

pgo_copy_runtime_dlls(pgo_log_tests)
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

    pgo_copy_runtime_dlls(pgo_spdlog_tests)
    gtest_discover_tests(pgo_spdlog_tests)
endif()

if(PGO_BUILD_C_API AND TARGET pgo_c)
    add_subdirectory(c_api)
endif()
