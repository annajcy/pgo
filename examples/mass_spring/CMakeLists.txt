add_executable(pgo_mass_spring_cloth cloth.cpp)
target_link_libraries(pgo_mass_spring_cloth
    PRIVATE
        pgo::core
        pgo::io
        pgo::log
        CLI11::CLI11
)

add_executable(pgo_mass_spring_bunny_cloth bunny_cloth.cpp)
target_link_libraries(pgo_mass_spring_bunny_cloth
    PRIVATE
        pgo::core
        pgo::io
        pgo::log
        CLI11::CLI11
)
