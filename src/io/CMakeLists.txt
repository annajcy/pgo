add_library(pgo_io STATIC
    obj_reader.cpp
    obj_writer.cpp
)

if(PGO_ENABLE_ALEMBIC)
    target_sources(pgo_io PRIVATE abc_writer.cpp)
endif()
add_library(pgo::io ALIAS pgo_io)

set_target_properties(pgo_io PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_features(pgo_io PUBLIC cxx_std_23)
target_include_directories(pgo_io PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_io
    PUBLIC pgo::core
    PRIVATE tinyobjloader::tinyobjloader_double
)

if(PGO_ENABLE_ALEMBIC)
    target_link_libraries(pgo_io PUBLIC Alembic::Alembic)
    target_compile_definitions(pgo_io PUBLIC PGO_ENABLE_ALEMBIC)
endif()
