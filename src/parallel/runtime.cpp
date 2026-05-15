#include <pgo/parallel/runtime.hpp>

#if defined(PGO_ENABLE_TBB)
#include <oneapi/tbb/global_control.h>
#endif

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace pgo::parallel {
namespace {

std::atomic<int> runtime_thread_count{0};

#if defined(PGO_ENABLE_TBB)
std::mutex global_control_mutex;
std::unique_ptr<oneapi::tbb::global_control> runtime_global_control;
#endif

} // namespace

bool is_tbb_enabled() noexcept {
#if defined(PGO_ENABLE_TBB)
    return true;
#else
    return false;
#endif
}

void set_thread_count(int thread_count) {
    if (thread_count < 0) {
        throw std::invalid_argument("pgo::parallel::set_thread_count requires a non-negative thread count");
    }

    runtime_thread_count.store(thread_count, std::memory_order_release);

#if defined(PGO_ENABLE_TBB)
    std::lock_guard<std::mutex> lock(global_control_mutex);
    runtime_global_control.reset();
    if (thread_count > 0) {
        runtime_global_control = std::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(thread_count));
    }
#endif
}

int configured_thread_count() noexcept {
    return runtime_thread_count.load(std::memory_order_acquire);
}

} // namespace pgo::parallel
