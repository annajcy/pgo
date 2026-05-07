add_library(pgo_eigen_config INTERFACE)
add_library(pgo::eigen_config ALIAS pgo_eigen_config)

target_link_libraries(pgo_eigen_config INTERFACE Eigen3::Eigen)

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
    find_package(MKL CONFIG REQUIRED)

    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_USE_MKL_ALL
        PGO_EIGEN_ACCELERATION_MKL
    )

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
