#include <pgo/parallel/parallel_for.hpp>
#include <pgo/parallel/runtime.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

struct Particle {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
};

std::vector<Particle> make_particles(std::size_t count) {
    std::vector<Particle> particles(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto t = static_cast<double>(i);
        particles[i] = Particle{t * 0.001, t * 0.002, t * 0.003, 0.0, 0.0, 0.0};
    }
    return particles;
}

void benchmark_particle_update(benchmark::State& state) {
    const auto particle_count = static_cast<std::size_t>(state.range(0));
    const auto thread_count = static_cast<int>(state.range(1));
    auto particles = make_particles(particle_count);

    pgo::parallel::set_thread_count(thread_count);

    for (auto _ : state) {
        pgo::parallel::parallel_for(std::size_t{0}, particles.size(), [&](std::size_t i) {
            auto& p = particles[i];
            const double force = std::sin(p.x) + std::cos(p.y) + std::sqrt(p.z + 1.0);
            p.vx += 0.001 * force;
            p.vy += 0.002 * force;
            p.vz += 0.003 * force;
            p.x += p.vx;
            p.y += p.vy;
            p.z += p.vz;
        });
        benchmark::DoNotOptimize(particles.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(particle_count));
}

BENCHMARK(benchmark_particle_update)
    ->Name("ParallelParticleUpdate")
    ->Args({1 << 12, 1})
    ->Args({1 << 12, 0})
    ->Args({1 << 18, 1})
    ->Args({1 << 18, 0});

} // namespace
