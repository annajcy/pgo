#pragma once

#include <cstddef>

namespace pgo::solver {

enum class SolverStatus {
    converged,
    max_iterations,
    regularization_failed,
    line_search_failed,
};

template <typename T>
struct SolverResult {
    SolverStatus status = SolverStatus::max_iterations;
    std::size_t iterations = 0;
    T final_value{};
    T final_gradient_norm{};
};

} // namespace pgo::solver
