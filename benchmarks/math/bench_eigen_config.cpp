#include <Eigen/Dense>
#include <benchmark/benchmark.h>

#include <cstdint>
#include <string_view>

namespace {

constexpr std::string_view pgo_backend_name() {
#if defined(PGO_EIGEN_ACCELERATION_NONE)
    return "NONE";
#elif defined(PGO_EIGEN_ACCELERATION_MKL)
    return "MKL";
#elif defined(PGO_EIGEN_ACCELERATION_ACCELERATE)
    return "ACCELERATE";
#else
    return "UNKNOWN";
#endif
}

constexpr std::string_view eigen_backend_macros() {
#if defined(EIGEN_USE_MKL_ALL)
    return "EIGEN_USE_MKL_ALL";
#elif defined(EIGEN_USE_BLAS)
    return "EIGEN_USE_BLAS";
#else
    return "none";
#endif
}

void benchmark_dense_matrix_multiply(benchmark::State& state) {
    const auto size = static_cast<Eigen::Index>(state.range(0));
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(size, size);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(size, size);
    Eigen::MatrixXd c(size, size);

    c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());

    for (auto _ : state) {
        c.noalias() = a * b;
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }

    const auto iterations = static_cast<std::int64_t>(state.iterations());
    const auto scalar_count = static_cast<std::int64_t>(size * size);
    const auto flop_count = static_cast<double>(iterations) * 2.0 * static_cast<double>(size) *
                            static_cast<double>(size) * static_cast<double>(size);

    state.SetItemsProcessed(iterations * scalar_count);
    state.SetBytesProcessed(iterations * scalar_count * static_cast<std::int64_t>(sizeof(double)) * 3);
    state.counters["flop/s"] = benchmark::Counter(flop_count, benchmark::Counter::kIsRate);
}

BENCHMARK(benchmark_dense_matrix_multiply)->Name("EigenDenseMatMul")->Arg(128)->Arg(256)->Arg(512);

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::AddCustomContext("pgo_eigen_backend", std::string(pgo_backend_name()));
    benchmark::AddCustomContext("eigen_backend_macros", std::string(eigen_backend_macros()));
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
