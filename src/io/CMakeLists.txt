add_library(pgo_io STATIC
    obj_reader.cpp
)
add_library(pgo::io ALIAS pgo_io)

target_compile_features(pgo_io PUBLIC cxx_std_23)
target_include_directories(pgo_io PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_io
    PUBLIC pgo::core
    PRIVATE tinyobjloader::tinyobjloader_double
)
