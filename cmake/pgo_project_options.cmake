add_library(pgo_project_options INTERFACE)
add_library(pgo::project_options ALIAS pgo_project_options)

option(PGO_ENABLE_NATIVE_ARCH "Enable native CPU tuning flags for release builds" ON)

if(PGO_ENABLE_NATIVE_ARCH)
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
    elseif(MSVC)
        target_compile_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    endif()
endif()
