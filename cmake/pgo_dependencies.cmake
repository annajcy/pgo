find_package(Eigen3 REQUIRED CONFIG)
find_package(tinyobjloader REQUIRED CONFIG)

if(PGO_BUILD_EXAMPLES OR PGO_BUILD_TOOLS)
    find_package(CLI11 REQUIRED CONFIG)
endif()

if(PGO_ENABLE_ALEMBIC)
    find_package(Alembic CONFIG REQUIRED)
endif()

if(PGO_BUILD_TESTS)
    find_package(GTest REQUIRED CONFIG)
endif()

if(PGO_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED CONFIG)
endif()

if(PGO_ENABLE_SPDLOG)
    find_package(spdlog CONFIG REQUIRED)
endif()
