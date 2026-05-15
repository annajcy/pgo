# Prefer the project-local venv Python so dependencies like nanobind are discoverable.
if(NOT Python_EXECUTABLE AND EXISTS "${PROJECT_SOURCE_DIR}/.venv")
    if(WIN32)
        find_program(_pgo_python NAMES python python3
            PATHS "${PROJECT_SOURCE_DIR}/.venv/Scripts"
            NO_DEFAULT_PATH
        )
    else()
        find_program(_pgo_python NAMES python3 python
            PATHS "${PROJECT_SOURCE_DIR}/.venv/bin"
            NO_DEFAULT_PATH
        )
    endif()
    if(_pgo_python)
        set(Python_EXECUTABLE "${_pgo_python}" CACHE FILEPATH "Python executable from project-local venv")
    endif()
    unset(_pgo_python)
endif()

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

function(pgo_collect_symlink_chain output_variable initial_path)
    set(paths "${initial_path}")
    set(current_path "${initial_path}")
    while(IS_SYMLINK "${current_path}")
        file(READ_SYMLINK "${current_path}" next_path)
        if(NOT IS_ABSOLUTE "${next_path}")
            get_filename_component(current_dir "${current_path}" DIRECTORY)
            set(next_path "${current_dir}/${next_path}")
        endif()
        list(APPEND paths "${next_path}")
        set(current_path "${next_path}")
    endwhile()
    set(${output_variable} ${paths} PARENT_SCOPE)
endfunction()

function(pgo_append_existing_runtime_file output_variable runtime_path)
    if(EXISTS "${runtime_path}")
        pgo_collect_symlink_chain(runtime_artifacts "${runtime_path}")
        set(normalized_runtime_artifacts)
        foreach(runtime_artifact IN LISTS runtime_artifacts)
            cmake_path(CONVERT "${runtime_artifact}" TO_CMAKE_PATH_LIST normalized_runtime_artifact NORMALIZE)
            list(APPEND normalized_runtime_artifacts "${normalized_runtime_artifact}")
        endforeach()
        set(${output_variable} ${${output_variable}} ${normalized_runtime_artifacts} PARENT_SCOPE)
    endif()
endfunction()

if(PGO_ENABLE_TBB)
    set(pgo_tbb_runtime_artifacts)
    foreach(pgo_tbb_runtime_dir IN LISTS
        onetbb_BIN_DIRS_RELEASE
        onetbb_LIB_DIRS_RELEASE
        onetbb_TBB_tbb_BIN_DIRS_RELEASE
        onetbb_TBB_tbb_LIB_DIRS_RELEASE
    )
        if(WIN32)
            file(GLOB pgo_tbb_runtime_candidates "${pgo_tbb_runtime_dir}/*.dll")
        elseif(APPLE)
            file(GLOB pgo_tbb_runtime_candidates "${pgo_tbb_runtime_dir}/libtbb*.dylib*")
        elseif(UNIX)
            file(GLOB pgo_tbb_runtime_candidates "${pgo_tbb_runtime_dir}/libtbb*.so*")
        else()
            set(pgo_tbb_runtime_candidates)
        endif()

        foreach(pgo_tbb_runtime_candidate IN LISTS pgo_tbb_runtime_candidates)
            pgo_append_existing_runtime_file(
                pgo_tbb_runtime_artifacts
                "${pgo_tbb_runtime_candidate}"
            )
        endforeach()
    endforeach()

    list(REMOVE_DUPLICATES pgo_tbb_runtime_artifacts)
    if(NOT pgo_tbb_runtime_artifacts)
        message(FATAL_ERROR "TBB is enabled, but no TBB runtime libraries were found to bundle in the Python wheel")
    endif()

    install(FILES ${pgo_tbb_runtime_artifacts}
        DESTINATION pgo
        COMPONENT python
    )
endif()

if(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "MKL")
    set(pgo_mkl_runtime_roots)
    if(DEFINED ENV{MKLROOT})
        list(APPEND pgo_mkl_runtime_roots "$ENV{MKLROOT}")
    endif()
    if(MKL_DIR)
        get_filename_component(pgo_mkl_root_from_dir "${MKL_DIR}/../../.." ABSOLUTE)
        list(APPEND pgo_mkl_runtime_roots "${pgo_mkl_root_from_dir}")
    endif()
    list(REMOVE_DUPLICATES pgo_mkl_runtime_roots)

    set(pgo_mkl_runtime_artifacts)
    foreach(pgo_mkl_runtime_root IN LISTS pgo_mkl_runtime_roots)
        if(WIN32)
            set(pgo_mkl_runtime_dirs
                "${pgo_mkl_runtime_root}/redist/intel64"
                "${pgo_mkl_runtime_root}/redist/intel64/mkl"
                "${pgo_mkl_runtime_root}/bin"
                "${pgo_mkl_runtime_root}/bin/intel64"
                "${pgo_mkl_runtime_root}/lib/intel64"
            )
            set(pgo_mkl_runtime_names
                mkl_core.2.dll
                mkl_def.2.dll
                mkl_intel_lp64.2.dll
                mkl_intel_thread.2.dll
            )
            file(GLOB pgo_mkl_openmp_candidates
                "${pgo_mkl_runtime_root}/../../compiler/latest/bin/libiomp5md.dll"
                "${pgo_mkl_runtime_root}/../../compiler/latest/redist/intel64/compiler/libiomp5md.dll"
            )
        elseif(UNIX AND NOT APPLE)
            set(pgo_mkl_runtime_dirs
                "${pgo_mkl_runtime_root}/lib/intel64"
            )
            set(pgo_mkl_runtime_names
                libmkl_core.so
                libmkl_core.so.2
                libmkl_intel_lp64.so
                libmkl_intel_lp64.so.2
                libmkl_intel_thread.so
                libmkl_intel_thread.so.2
            )
            file(GLOB pgo_mkl_openmp_candidates
                "${pgo_mkl_runtime_root}/../../compiler/latest/lib/libiomp5.so"
                "${pgo_mkl_runtime_root}/../../compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so"
            )
        else()
            set(pgo_mkl_runtime_dirs)
            set(pgo_mkl_runtime_names)
            set(pgo_mkl_openmp_candidates)
        endif()

        foreach(pgo_mkl_runtime_dir IN LISTS pgo_mkl_runtime_dirs)
            foreach(pgo_mkl_runtime_name IN LISTS pgo_mkl_runtime_names)
                pgo_append_existing_runtime_file(
                    pgo_mkl_runtime_artifacts
                    "${pgo_mkl_runtime_dir}/${pgo_mkl_runtime_name}"
                )
            endforeach()
        endforeach()
        foreach(pgo_mkl_openmp_candidate IN LISTS pgo_mkl_openmp_candidates)
            pgo_append_existing_runtime_file(
                pgo_mkl_runtime_artifacts
                "${pgo_mkl_openmp_candidate}"
            )
        endforeach()
    endforeach()

    list(REMOVE_DUPLICATES pgo_mkl_runtime_artifacts)
    if(NOT pgo_mkl_runtime_artifacts)
        message(FATAL_ERROR "MKL acceleration is enabled, but no MKL runtime libraries were found to bundle in the Python wheel")
    endif()

    install(FILES ${pgo_mkl_runtime_artifacts}
        DESTINATION pgo
        COMPONENT python
    )
endif()
