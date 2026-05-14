#include "pgo/core/math/types.hpp"

#include <gtest/gtest.h>
#include <type_traits>

namespace pgo::math::test {

TEST(MathTypes, FixedVectorHasExpectedSize) {
    pgo::math::Vec<double, 3> vector{};
    EXPECT_EQ(3, vector.size());
}

TEST(MathTypes, DynamicVectorCanResize) {
    pgo::math::DVec<double> vector{};
    vector.resize(4);
    EXPECT_EQ(4, vector.size());
}

TEST(MathTypes, SparseMatrixUsesRowMajorStorage) {
    using SparseMatrix = pgo::math::SparseMat<double>;
    EXPECT_TRUE((std::is_same_v<SparseMatrix, Eigen::SparseMatrix<double, Eigen::RowMajor>>));
}

} // namespace pgo::math::test
