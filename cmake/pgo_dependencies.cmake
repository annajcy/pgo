find_package(Eigen3 REQUIRED CONFIG)

if(PGO_BUILD_EXAMPLES)
    find_package(CLI11 REQUIRED CONFIG)
endif()

if(PGO_BUILD_TESTS)
    find_package(GTest REQUIRED CONFIG)
endif()
