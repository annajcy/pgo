add_subdirectory(io)

add_executable(pgo_mass_spring_cloth mass_spring_cloth.cpp)
target_link_libraries(pgo_mass_spring_cloth
    PRIVATE
        pgo::core
        pgo::io
        CLI11::CLI11
)
