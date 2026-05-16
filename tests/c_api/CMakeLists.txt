add_executable(pgo_c_api_tests test_c_api.c)
target_link_libraries(pgo_c_api_tests PRIVATE pgo::c_api pgo_project_sanitizers)
add_test(NAME pgo_c_api_tests COMMAND pgo_c_api_tests)
set_tests_properties(pgo_c_api_tests PROPERTIES
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:pgo_c>;PGO_TEST_TMPDIR=set:${CMAKE_CURRENT_BINARY_DIR}"
)

if(TARGET pgo_c_mass_spring_cloth)
    add_test(
        NAME pgo_c_mass_spring_cloth_example_smoke
        COMMAND pgo_c_mass_spring_cloth
            --frames 2
            --resolution 3
            --output "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/smoke-cloth"
    )
    set_tests_properties(pgo_c_mass_spring_cloth_example_smoke PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:pgo_c>"
    )
endif()
