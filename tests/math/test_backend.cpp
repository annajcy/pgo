#include <gtest/gtest.h>

#include <type_traits>

#include "pgo/math/backend.hpp"

namespace pgo::math::test {

static_assert(pgo::math::MathBackend<pgo::math::DefaultBackend>);
static_assert(pgo::math::ScalarLike<double>);
static_assert(pgo::math::RealScalar<double>);

TEST(MathBackend, FixedVectorHasExpectedSize)
{
    pgo::math::Vec<double, 3> vector{};

    EXPECT_EQ(3, vector.size());
}

TEST(MathBackend, DynamicVectorCanResize)
{
    pgo::math::DVec<double> vector{};

    vector.resize(4);

    EXPECT_EQ(4, vector.size());
}

TEST(MathBackend, SparseMatrixUsesRowMajorStorage)
{
    using SparseMatrix = pgo::math::SparseMat<double>;

    EXPECT_TRUE((std::is_same_v<SparseMatrix, Eigen::SparseMatrix<double, Eigen::RowMajor>>));
}

} // namespace pgo::math::test
