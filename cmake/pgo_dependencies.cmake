find_package(Eigen3 REQUIRED CONFIG)

if(PGO_BUILD_EXAMPLES)
    find_package(CLI11 REQUIRED CONFIG)
endif()

if(PGO_BUILD_TOOLS)
    find_package(CLI11 REQUIRED CONFIG)
    find_package(Alembic REQUIRED CONFIG)
endif()

if(PGO_BUILD_TESTS)
    find_package(GTest REQUIRED CONFIG)
endif()

if(PGO_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED CONFIG)
endif()
