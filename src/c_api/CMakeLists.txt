add_library(pgo_c SHARED
    pgo_c.cpp
)
add_library(pgo::c_api ALIAS pgo_c)

target_compile_features(pgo_c PRIVATE cxx_std_23)
target_compile_definitions(pgo_c PRIVATE PGO_C_BUILDING_LIBRARY)
target_include_directories(pgo_c PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_c
    PRIVATE
        pgo::core
        pgo::io
        pgo_project_warnings
        pgo_project_sanitizers
)

set_target_properties(pgo_c PROPERTIES
    OUTPUT_NAME pgo_c
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON
)

if(APPLE)
    target_link_options(pgo_c PRIVATE
        "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/pgo_c.symbols"
    )
elseif(UNIX)
    target_link_options(pgo_c PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/pgo_c.version"
    )
endif()

install(TARGETS pgo_c
    RUNTIME DESTINATION pgo COMPONENT python
    LIBRARY DESTINATION pgo COMPONENT python
    ARCHIVE DESTINATION lib COMPONENT dev
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/pgo_c"
    DESTINATION include
    COMPONENT dev
)
