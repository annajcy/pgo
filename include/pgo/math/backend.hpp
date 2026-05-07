#pragma once

#include "pgo/math/eigen_backend.hpp"
#include "pgo/math/scalar.hpp"

#include <cstddef>
#include <cstdint>

namespace pgo::math {

using Index = std::uint32_t;

template <class Backend>
concept MathBackend = requires {
    typename Backend::DenseIndex;
    typename Backend::template Vec<double, 3>;
    typename Backend::template Mat<double, 3, 3>;
    typename Backend::template DVec<double>;
    typename Backend::template DMat<double>;
    typename Backend::template SparseMat<double>;
    typename Backend::template Triplet<double>;
};

using DefaultBackend = pgo::math::eigen::EigenBackend;
static_assert(MathBackend<DefaultBackend>);

using DenseIndex = DefaultBackend::DenseIndex;

[[nodiscard]] constexpr DenseIndex dense_index(const std::size_t index) {
    return static_cast<DenseIndex>(index);
}

template <ScalarLike T, int Dim>
using Vec = DefaultBackend::template Vec<T, Dim>;

template <ScalarLike T, int Rows, int Cols>
using Mat = DefaultBackend::template Mat<T, Rows, Cols>;

template <ScalarLike T>
using DVec = DefaultBackend::template DVec<T>;

template <ScalarLike T>
using DMat = DefaultBackend::template DMat<T>;

template <RealScalar T>
using SparseMat = DefaultBackend::template SparseMat<T>;

template <RealScalar T>
using Triplet = DefaultBackend::template Triplet<T>;

} // namespace pgo::math
