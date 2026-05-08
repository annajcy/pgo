#include "pgo/math/finite_difference.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace pgo::math::test {

TEST(finite_difference, ComputesGradientOfQuadraticFunction) {
    pgo::math::DVec<double> x{2};
    x << 0.7, -1.2;

    const auto gradient = pgo::math::finite_difference_gradient(
        [](const pgo::math::DVec<double>& value_x) {
            const auto a = value_x[0];
            const auto b = value_x[1];
            return a * a + 3.0 * a * b + 2.0 * b * b;
        },
        x, 1e-6);

    EXPECT_NEAR(2.0 * x[0] + 3.0 * x[1], gradient[0], 1e-9);
    EXPECT_NEAR(3.0 * x[0] + 4.0 * x[1], gradient[1], 1e-9);
    EXPECT_DOUBLE_EQ(0.7, x[0]);
    EXPECT_DOUBLE_EQ(-1.2, x[1]);
}

TEST(finite_difference, ComputesHessianFromGradientOfQuadraticFunction) {
    pgo::math::DVec<double> x{2};
    x << 0.7, -1.2;

    const auto hessian = pgo::math::finite_difference_hessian_from_gradient(
        [](const pgo::math::DVec<double>& value_x) {
            pgo::math::DVec<double> gradient{2};
            gradient << 2.0 * value_x[0] + 3.0 * value_x[1], 3.0 * value_x[0] + 4.0 * value_x[1];
            return gradient;
        },
        x, 1e-6);

    EXPECT_NEAR(2.0, hessian(0, 0), 1e-9);
    EXPECT_NEAR(3.0, hessian(0, 1), 1e-9);
    EXPECT_NEAR(3.0, hessian(1, 0), 1e-9);
    EXPECT_NEAR(4.0, hessian(1, 1), 1e-9);
}

TEST(finite_difference, RejectsNonPositiveEpsilon) {
    pgo::math::DVec<double> x{2};
    x.setZero();

    EXPECT_THROW(static_cast<void>(
                     pgo::math::finite_difference_gradient([](const pgo::math::DVec<double>&) { return 0.0; }, x, 0.0)),
                 std::runtime_error);
    EXPECT_THROW(static_cast<void>(pgo::math::finite_difference_hessian_from_gradient(
                     [](const pgo::math::DVec<double>& value_x) { return value_x; }, x, -1.0)),
                 std::runtime_error);
}

} // namespace pgo::math::test
