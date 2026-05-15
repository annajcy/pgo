add_library(pgo_eigen_config INTERFACE)
add_library(pgo::eigen_config ALIAS pgo_eigen_config)

target_link_libraries(pgo_eigen_config INTERFACE Eigen3::Eigen)

if(PGO_EIGEN_DONT_PARALLELIZE)
    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_DONT_PARALLELIZE
    )
endif()

if(NOT PGO_EIGEN_MAX_ALIGN_BYTES STREQUAL "")
    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_MAX_ALIGN_BYTES=${PGO_EIGEN_MAX_ALIGN_BYTES}
    )
endif()

set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "NONE")

if(PGO_ENABLE_EIGEN_ACCELERATION)
    if(PGO_EIGEN_ACCELERATION_BACKEND STREQUAL "AUTO")
        if(APPLE)
            set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "ACCELERATE")
        elseif(UNIX OR WIN32)
            set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "MKL")
        else()
            set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "NONE")
        endif()
    else()
        set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "${PGO_EIGEN_ACCELERATION_BACKEND}")
    endif()
endif()

if(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "MKL")
    if(NOT MKL_DIR AND DEFINED ENV{MKLROOT})
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}")

        if(EXISTS "$ENV{MKLROOT}/lib/cmake/mkl/MKLConfig.cmake")
            set(MKL_DIR "$ENV{MKLROOT}/lib/cmake/mkl" CACHE PATH "Path to oneMKL CMake package")
        endif()
    endif()

    if(NOT MKL_DIR)
        set(PGO_MKL_ROOT_CANDIDATES)

        if(WIN32)
            list(APPEND PGO_MKL_ROOT_CANDIDATES
                "$ENV{ProgramFiles(x86)}/Intel/oneAPI/mkl/latest"
                "$ENV{ProgramFiles}/Intel/oneAPI/mkl/latest"
            )
        elseif(UNIX)
            list(APPEND PGO_MKL_ROOT_CANDIDATES
                "/opt/intel/oneapi/mkl/latest"
            )
        endif()

        foreach(PGO_MKL_ROOT_CANDIDATE IN LISTS PGO_MKL_ROOT_CANDIDATES)
            if(EXISTS "${PGO_MKL_ROOT_CANDIDATE}/lib/cmake/mkl/MKLConfig.cmake")
                list(PREPEND CMAKE_PREFIX_PATH "${PGO_MKL_ROOT_CANDIDATE}")
                set(MKL_DIR "${PGO_MKL_ROOT_CANDIDATE}/lib/cmake/mkl" CACHE PATH "Path to oneMKL CMake package")
                break()
            endif()
        endforeach()
    endif()

    find_package(MKL CONFIG REQUIRED)

    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_USE_MKL_ALL
        PGO_EIGEN_ACCELERATION_MKL
    )

    if(PGO_EIGEN_MKL_NO_DIRECT_CALL)
        target_compile_definitions(pgo_eigen_config INTERFACE
            EIGEN_MKL_NO_DIRECT_CALL
        )
    endif()

    target_link_libraries(pgo_eigen_config INTERFACE MKL::MKL)
elseif(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "ACCELERATE")
    if(NOT APPLE)
        message(FATAL_ERROR "PGO_EIGEN_ACCELERATION_BACKEND=ACCELERATE is only supported on Apple platforms.")
    endif()

    find_library(PGO_ACCELERATE_FRAMEWORK Accelerate REQUIRED)

    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_USE_BLAS
        PGO_EIGEN_ACCELERATION_ACCELERATE
    )

    target_link_libraries(pgo_eigen_config INTERFACE "${PGO_ACCELERATE_FRAMEWORK}")
elseif(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "NONE")
    target_compile_definitions(pgo_eigen_config INTERFACE
        PGO_EIGEN_ACCELERATION_NONE
    )
else()
    message(FATAL_ERROR "Unknown PGO_SELECTED_EIGEN_ACCELERATION_BACKEND=${PGO_SELECTED_EIGEN_ACCELERATION_BACKEND}")
endif()
