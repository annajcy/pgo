add_library(pgo_log
    registry.cpp
    stderr_sink.cpp
)
add_library(pgo::log ALIAS pgo_log)

target_compile_features(pgo_log PUBLIC cxx_std_23)
target_include_directories(pgo_log PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_log PUBLIC pgo_project_warnings pgo_project_sanitizers)

if(PGO_ENABLE_SPDLOG)
    target_sources(pgo_log PRIVATE spdlog_sink.cpp)
    target_compile_definitions(pgo_log PUBLIC PGO_ENABLE_SPDLOG)
    target_link_libraries(pgo_log PUBLIC spdlog::spdlog)
endif()
