#pragma once

#include "pgo/energy/energy_concepts.hpp"
#include "pgo/solver/feasible_set.hpp"
#include "pgo/solver/solver_options.hpp"

namespace pgo::solver {

template <typename T, typename Energy>
    requires pgo::energy::DifferentiableEnergy<Energy, T>
[[nodiscard]] T armijo_backtrack(
    const Energy& energy,
    const pgo::math::DVec<T>& z,
    const pgo::math::DVec<T>& dz,
    const pgo::math::DVec<T>& gradient,
    const T current_value,
    const LineSearchOptions<T>& options) {

    T alpha = T{1};

    while (alpha >= options.min_step) {
        const pgo::math::DVec<T> trial = z + alpha * dz;

        if (energy.value(trial) <= current_value + options.armijo_c * alpha * gradient.dot(dz)) {
            return alpha;
        }

        alpha *= options.shrink;
    }

    return T{0};
}

template <typename T, typename Energy, typename FeasibleSet>
    requires pgo::energy::DifferentiableEnergy<Energy, T> && FeasibleSetLike<FeasibleSet, T>
[[nodiscard]] T feasible_armijo_line_search(
    const Energy& energy,
    const FeasibleSet& feasible,
    const pgo::math::DVec<T>& z,
    const pgo::math::DVec<T>& dz,
    const pgo::math::DVec<T>& gradient,
    const T current_value,
    const LineSearchOptions<T>& options) {

    const T feasible_alpha = options.feasibility_safety * feasible.max_step(z, dz);
    const pgo::math::DVec<T> d_scaled = feasible_alpha * dz;
    const T backtrack_alpha = armijo_backtrack(energy, z, d_scaled, gradient, current_value, options);
    return feasible_alpha * backtrack_alpha;
}

} // namespace pgo::solver
