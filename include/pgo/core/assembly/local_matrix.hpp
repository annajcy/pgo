#pragma once

#include "pgo/core/math/types.hpp"

namespace pgo::assembly {

template <typename T>
using LocalVector = pgo::math::DVec<T>;

template <typename T>
using LocalMatrix = pgo::math::DMat<T>;

} // namespace pgo::assembly
