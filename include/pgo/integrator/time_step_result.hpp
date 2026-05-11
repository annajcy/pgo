#pragma once

#include "pgo/solver/solver_result.hpp"

#include <cstddef>

namespace pgo::integrator {

template <typename T>
struct TimeStepResult {
    pgo::solver::SolverStatus status = pgo::solver::SolverStatus::max_iterations;
    std::size_t solver_iterations = 0;
    T final_value{};
    T final_gradient_norm{};
    T dt{};
};

} // namespace pgo::integrator
