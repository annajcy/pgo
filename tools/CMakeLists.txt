if(PGO_ENABLE_ALEMBIC)
    add_executable(pgo_obj_frames_to_abc obj_frames_to_abc.cpp)
    target_link_libraries(pgo_obj_frames_to_abc PRIVATE pgo::io pgo::log CLI11::CLI11)
endif()
