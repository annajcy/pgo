#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstddef>
#include <cstdint>

namespace pgo::math {

using Index = std::uint32_t;

using DenseIndex = Eigen::Index;

[[nodiscard]] constexpr DenseIndex dense_index(const std::size_t index) {
    return static_cast<DenseIndex>(index);
}

template <typename T, int Dim>
using Vec = Eigen::Matrix<T, Dim, 1>;

template <typename T, int Rows, int Cols>
using Mat = Eigen::Matrix<T, Rows, Cols>;

template <typename T>
using DVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using DMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template <typename T>
using Triplet = Eigen::Triplet<T>;

} // namespace pgo::math
