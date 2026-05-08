add_executable(pgo_obj_load_and_write_frames obj_load_and_write_frames.cpp)

target_link_libraries(pgo_obj_load_and_write_frames
    PRIVATE
        pgo::core
        CLI11::CLI11
)
