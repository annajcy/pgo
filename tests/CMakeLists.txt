include(GoogleTest)

add_executable(pgo_tests
    base/test_assert.cpp
    dof/test_dof.cpp
    geometry/test_rest_mesh.cpp
    io/test_obj_io.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    storage/test_storage.cpp
)

target_link_libraries(pgo_tests
    PRIVATE
        pgo::core
        GTest::gtest_main
)

gtest_discover_tests(pgo_tests)
