#pragma once

#include "pgo/solver/solver_result.hpp"

#include <string_view>

namespace pgo::solver {

[[nodiscard]] constexpr std::string_view status_name(const SolverStatus status) {
    switch (status) {
    case SolverStatus::converged:
        return "converged";
    case SolverStatus::max_iterations:
        return "max_iterations";
    case SolverStatus::regularization_failed:
        return "regularization_failed";
    case SolverStatus::line_search_failed:
        return "line_search_failed";
    }
    return "unknown";
}

} // namespace pgo::solver
