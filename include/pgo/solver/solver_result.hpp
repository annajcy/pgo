#pragma once

#include <cstddef>
#include <string>

namespace pgo::solver {

template <typename T>
struct SolverResult {
    bool converged = false;
    std::size_t iterations = 0;
    T final_value{};
    T final_gradient_norm{};
    std::string message;
};

} // namespace pgo::solver
