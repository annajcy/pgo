include(GoogleTest)

add_executable(pgo_tests
    base/test_assert.cpp
    math/test_backend.cpp
)

target_link_libraries(pgo_tests
    PRIVATE
        pgo::core
        GTest::gtest_main
)

gtest_discover_tests(pgo_tests)
