#pragma once

#include <pgo/parallel/runtime.hpp>

#if defined(PGO_ENABLE_TBB)
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#endif

#include <type_traits>

namespace pgo::parallel {

template <class Index, class Func>
void serial_for(Index begin, Index end, Func&& func) {
    static_assert(std::is_integral_v<Index>, "pgo::parallel::serial_for requires an integral index type");
    for (Index i = begin; i < end; ++i) {
        func(i);
    }
}

template <class Index, class Func>
void parallel_for(Index begin, Index end, Func&& func) {
    static_assert(std::is_integral_v<Index>, "pgo::parallel::parallel_for requires an integral index type");

    if (end <= begin) {
        return;
    }

    if (configured_thread_count() == 1) {
        serial_for(begin, end, static_cast<Func&&>(func));
        return;
    }

#if defined(PGO_ENABLE_TBB)
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<Index>(begin, end),
        [&](const oneapi::tbb::blocked_range<Index>& range) {
            for (Index i = range.begin(); i < range.end(); ++i) {
                func(i);
            }
        });
#else
    serial_for(begin, end, static_cast<Func&&>(func));
#endif
}

} // namespace pgo::parallel
