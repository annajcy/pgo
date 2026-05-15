add_library(pgo_parallel_runtime STATIC
    runtime.cpp
)
add_library(pgo::parallel_runtime ALIAS pgo_parallel_runtime)

target_compile_features(pgo_parallel_runtime PUBLIC cxx_std_23)
target_include_directories(pgo_parallel_runtime PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(PGO_ENABLE_TBB)
    target_link_libraries(pgo_parallel_runtime PUBLIC TBB::tbb)
    target_compile_definitions(pgo_parallel_runtime PUBLIC PGO_ENABLE_TBB)
endif()
