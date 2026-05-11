#pragma once

#include "pgo/math/types.hpp"

namespace pgo::integrator {

template <typename T>
struct DynamicState {
    pgo::math::DVec<T> u;
    pgo::math::DVec<T> v;
    pgo::math::DVec<T> a;
};

} // namespace pgo::integrator
