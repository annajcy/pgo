#pragma once

namespace pgo::parallel {

[[nodiscard]] bool is_tbb_enabled() noexcept;

void set_thread_count(int thread_count);
[[nodiscard]] int configured_thread_count() noexcept;

} // namespace pgo::parallel
