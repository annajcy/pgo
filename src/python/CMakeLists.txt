find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)

if(NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        RESULT_VARIABLE pgo_nanobind_cmake_dir_result
        OUTPUT_VARIABLE pgo_nanobind_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pgo_nanobind_cmake_dir_result EQUAL 0)
        set(nanobind_DIR "${pgo_nanobind_cmake_dir}" CACHE PATH "nanobind CMake package directory")
    endif()
endif()

find_package(nanobind CONFIG REQUIRED)

set(pgo_python_sources pgo_ext.cpp)

if(PGO_ENABLE_PYTHON_STABLE_ABI AND NOT Python_Development.SABIModule_FOUND)
    message(FATAL_ERROR "PGO_ENABLE_PYTHON_STABLE_ABI=ON requires Python Development.SABIModule; build stable-ABI wheels from Python 3.12+ and pass -Cwheel.py-api=cp312")
endif()

if(PGO_ENABLE_PYTHON_STABLE_ABI)
    nanobind_add_module(${PGO_PYTHON_EXTENSION_NAME}
        STABLE_ABI
        NB_STATIC
        ${pgo_python_sources}
    )
else()
    nanobind_add_module(${PGO_PYTHON_EXTENSION_NAME}
        NB_STATIC
        ${pgo_python_sources}
    )
endif()

target_link_libraries(${PGO_PYTHON_EXTENSION_NAME} PRIVATE pgo::c_api)
target_include_directories(${PGO_PYTHON_EXTENSION_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/include")

if(APPLE)
    set_target_properties(${PGO_PYTHON_EXTENSION_NAME} PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(${PGO_PYTHON_EXTENSION_NAME} PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

install(TARGETS ${PGO_PYTHON_EXTENSION_NAME}
    LIBRARY DESTINATION pgo COMPONENT python
    RUNTIME DESTINATION pgo COMPONENT python
)
