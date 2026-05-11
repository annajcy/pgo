#pragma once

#include <cstddef>

namespace pgo::solver {

template <typename T>
struct LineSearchOptions {
    T armijo_c = T{1e-4};
    T shrink = T{0.5};
    T min_step = T{1e-12};
    T feasibility_safety = T{0.99};
};

template <typename T>
struct NewtonOptions {
    std::size_t max_iterations = 50;
    T gradient_tolerance = T{1e-8};
    T initial_regularization = T{0};
    T min_regularization = T{1e-12};
    T regularization_growth = T{10};
    T max_regularization = T{1e8};
    LineSearchOptions<T> line_search;
};

} // namespace pgo::solver
