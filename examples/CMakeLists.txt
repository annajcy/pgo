add_subdirectory(io)
add_subdirectory(mass_spring)

if(PGO_BUILD_C_API AND TARGET pgo_c)
    add_subdirectory(c_api)
endif()
