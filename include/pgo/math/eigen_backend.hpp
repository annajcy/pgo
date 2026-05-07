#pragma once

#include "pgo/math/scalar.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace pgo::math::eigen {

struct EigenBackend {
    using DenseIndex = Eigen::Index;

    template <pgo::math::ScalarLike T, int Dim>
    using Vec = Eigen::Matrix<T, Dim, 1>;

    template <pgo::math::ScalarLike T, int Rows, int Cols>
    using Mat = Eigen::Matrix<T, Rows, Cols>;

    template <pgo::math::ScalarLike T>
    using DVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    template <pgo::math::ScalarLike T>
    using DMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template <pgo::math::RealScalar T>
    using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

    template <pgo::math::RealScalar T>
    using Triplet = Eigen::Triplet<T>;
};

} // namespace pgo::math::eigen
