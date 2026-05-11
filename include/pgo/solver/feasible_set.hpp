#pragma once

#include "pgo/math/types.hpp"

#include <concepts>

namespace pgo::solver {

template <typename FeasibleSet, typename T>
concept FeasibleSetLike = requires(const FeasibleSet& feasible,
                                    const pgo::math::DVec<T>& z,
                                    const pgo::math::DVec<T>& dz) {
    { feasible.is_feasible(z) } -> std::same_as<bool>;
    { feasible.max_step(z, dz) } -> std::same_as<T>;
};

template <typename T>
struct AlwaysFeasible {
    [[nodiscard]] bool is_feasible(const pgo::math::DVec<T>&) const {
        return true;
    }

    [[nodiscard]] T max_step(const pgo::math::DVec<T>&, const pgo::math::DVec<T>&) const {
        return T{1};
    }
};

} // namespace pgo::solver
