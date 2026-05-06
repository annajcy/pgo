#pragma once

#include <concepts>

namespace pgo::math {

template <class T>
concept ScalarLike = requires(T a, T b) {
    T{0};
    T{1};
    a + b;
    a - b;
    a * b;
    a / b;
    -a;
};

template <class T>
concept RealScalar = std::floating_point<T>;

} // namespace pgo::math
