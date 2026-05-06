add_library(pgo_project_warnings INTERFACE)

if(MSVC)
    target_compile_options(pgo_project_warnings INTERFACE /W4)
    if(PGO_WARNINGS_AS_ERRORS)
        target_compile_options(pgo_project_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(pgo_project_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
    if(PGO_WARNINGS_AS_ERRORS)
        target_compile_options(pgo_project_warnings INTERFACE -Werror)
    endif()
endif()
