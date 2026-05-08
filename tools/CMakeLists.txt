add_executable(pgo_obj_frames_to_abc obj_frames_to_abc.cpp)

target_link_libraries(pgo_obj_frames_to_abc PRIVATE CLI11::CLI11 Alembic::Alembic)
