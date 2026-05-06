add_library(pgo_project_sanitizers INTERFACE)

if(PGO_ENABLE_SANITIZERS)
    if(MSVC)
        message(WARNING "PGO_ENABLE_SANITIZERS is currently configured for Clang/GCC style sanitizers only")
    else()
        target_compile_options(pgo_project_sanitizers INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(pgo_project_sanitizers INTERFACE
            -fsanitize=address,undefined
        )
    endif()
endif()
