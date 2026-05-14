add_executable(pgo_c_api_tests test_c_api.c)
target_link_libraries(pgo_c_api_tests PRIVATE pgo::c_api pgo_project_sanitizers)
add_test(NAME pgo_c_api_tests COMMAND pgo_c_api_tests)
