add_executable(pgo_benchmarks
    math/bench_eigen_config.cpp
    parallel/bench_parallel_for.cpp
)

target_link_libraries(pgo_benchmarks
    PRIVATE
        pgo::core
        benchmark::benchmark
)
