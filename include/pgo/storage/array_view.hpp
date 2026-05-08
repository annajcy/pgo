#pragma once

#include <span>

namespace pgo::storage {

template <typename T>
using ArrayView = std::span<T>;

template <typename T>
using ConstArrayView = std::span<const T>;

} // namespace pgo::storage
