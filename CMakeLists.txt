cmake_minimum_required(VERSION 3.28)

project(pgo VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/pgo_options.cmake)
include(cmake/pgo_dependencies.cmake)
include(cmake/pgo_eigen.cmake)
include(cmake/pgo_project_options.cmake)
include(cmake/pgo_warnings.cmake)
include(cmake/pgo_sanitizers.cmake)

add_custom_target(copy_compile_commands
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${PROJECT_SOURCE_DIR}/compile_commands.json"
    BYPRODUCTS "${PROJECT_SOURCE_DIR}/compile_commands.json"
    COMMENT "Copying compile_commands.json to the source tree"
    VERBATIM
)

add_library(pgo_core INTERFACE)
add_library(pgo::core ALIAS pgo_core)

target_compile_features(pgo_core INTERFACE cxx_std_23)
target_include_directories(pgo_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_core INTERFACE pgo::eigen_config pgo::project_options)
target_link_libraries(pgo_core INTERFACE pgo_project_warnings pgo_project_sanitizers)

add_subdirectory(src/io)
add_subdirectory(src/log)

target_link_libraries(pgo_log PUBLIC pgo::project_options)

if(PGO_BUILD_C_API AND EXISTS "${PROJECT_SOURCE_DIR}/src/c_api/CMakeLists.txt")
    add_subdirectory(src/c_api)
endif()

if(PGO_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

if(PGO_BUILD_TOOLS AND EXISTS "${PROJECT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
endif()

if(PGO_BUILD_BENCHMARKS AND EXISTS "${PROJECT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()

if(PGO_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()
