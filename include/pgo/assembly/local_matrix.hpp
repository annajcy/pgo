#pragma once

#include "pgo/math/backend.hpp"

namespace pgo::assembly {

template <pgo::math::RealScalar T>
using LocalVector = pgo::math::DVec<T>;

template <pgo::math::RealScalar T>
using LocalMatrix = pgo::math::DMat<T>;

} // namespace pgo::assembly
