#pragma once

#include <span>

namespace pgo::storage {

template <class T>
using ArrayView = std::span<T>;

template <class T>
using ConstArrayView = std::span<const T>;

} // namespace pgo::storage
