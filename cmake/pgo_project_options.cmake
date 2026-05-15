add_library(pgo_project_options INTERFACE)
add_library(pgo::project_options ALIAS pgo_project_options)

option(PGO_ENABLE_NATIVE_ARCH "Enable native CPU tuning flags for release builds" ON)

set(PGO_MSVC_ARCH "DEFAULT" CACHE STRING "MSVC CPU ISA: DEFAULT, AVX, AVX2, AVX512")
set_property(CACHE PGO_MSVC_ARCH PROPERTY STRINGS DEFAULT AVX AVX2 AVX512)

if(PGO_ENABLE_NATIVE_ARCH)
    message(STATUS "Enabling native CPU tuning flags for release builds")
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(pgo_project_options INTERFACE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-march=native>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-mtune=native>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-mtune=native>
        )
        target_link_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-mtune=native>
        )
    endif()
endif()

if(MSVC)
    target_compile_options(pgo_project_options INTERFACE
        /MP
        /bigobj
        /Zc:__cplusplus
    )
endif()

if(MSVC)
    set(_pgo_msvc_resolved_arch "${PGO_MSVC_ARCH}")
    if(_pgo_msvc_resolved_arch STREQUAL "DEFAULT" AND PGO_ENABLE_NATIVE_ARCH)
        set(_pgo_msvc_resolved_arch "AVX2")
        message(STATUS "PGO_MSVC_ARCH=DEFAULT with PGO_ENABLE_NATIVE_ARCH=ON resolves to AVX2 (legacy default)")
    endif()

    if(_pgo_msvc_resolved_arch STREQUAL "AVX")
        target_compile_options(pgo_project_options INTERFACE $<$<CONFIG:Release>:/arch:AVX>)
    elseif(_pgo_msvc_resolved_arch STREQUAL "AVX2")
        target_compile_options(pgo_project_options INTERFACE $<$<CONFIG:Release>:/arch:AVX2>)
    elseif(_pgo_msvc_resolved_arch STREQUAL "AVX512")
        target_compile_options(pgo_project_options INTERFACE $<$<CONFIG:Release>:/arch:AVX512>)
    elseif(_pgo_msvc_resolved_arch STREQUAL "DEFAULT")
        # NATIVE_ARCH=OFF + DEFAULT: keep MSVC compiler default ISA for portable Windows binaries
    else()
        message(FATAL_ERROR "Unknown PGO_MSVC_ARCH=${PGO_MSVC_ARCH}")
    endif()
endif()

option(PGO_ENABLE_RELEASE_DEBUG_SYMBOLS "Emit debug symbols in release builds" OFF)

if(PGO_ENABLE_RELEASE_DEBUG_SYMBOLS)
    if(MSVC)
        target_compile_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/Zi>
        )
        target_link_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/DEBUG>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(pgo_project_options INTERFACE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-g>
        )
        target_link_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:-g>
        )
    endif()
endif()
