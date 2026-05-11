#include "pgo/energy/constant_force_energy.hpp"
#include "pgo/energy/energy_concepts.hpp"
#include "pgo/solver/status_name.hpp"

#include <gtest/gtest.h>

namespace pgo::energy::test {

static_assert(FullEnergy<ConstantForceEnergy<double>, double>);

TEST(ConstantForceEnergy, ValueIsNegativeForceDotU) {
    pgo::math::DVec<double> f(3);
    f << 1.0, -2.0, 0.5;
    const ConstantForceEnergy<double> energy{f};

    pgo::math::DVec<double> u(3);
    u << 3.0, 1.0, -2.0;

    const double expected = -(f.dot(u)); // -(1*3 + (-2)*1 + 0.5*(-2)) = -(3 - 2 - 1) = 0
    EXPECT_DOUBLE_EQ(expected, energy.value(u));
}

TEST(ConstantForceEnergy, GradientIsNegativeForce) {
    pgo::math::DVec<double> f(3);
    f << 1.0, -2.0, 0.5;
    const ConstantForceEnergy<double> energy{f};

    pgo::math::DVec<double> u(3);
    u << 3.0, 1.0, -2.0;

    pgo::math::DVec<double> g;
    energy.gradient(u, g);

    ASSERT_EQ(g.size(), f.size());
    const pgo::math::DVec<double> expected = -f;
    EXPECT_TRUE(expected.isApprox(g));
}

TEST(ConstantForceEnergy, HessianIsZero) {
    pgo::math::DVec<double> f(3);
    f << 1.0, -2.0, 0.5;
    const ConstantForceEnergy<double> energy{f};

    pgo::math::DVec<double> u(3);
    u << 3.0, 1.0, -2.0;

    pgo::math::SparseMat<double> H;
    energy.hessian(u, H);

    ASSERT_EQ(H.rows(), 3);
    ASSERT_EQ(H.cols(), 3);
    EXPECT_EQ(H.nonZeros(), 0);
}

TEST(ConstantForceEnergy, FusedAPIEqualsSeparateCalls) {
    pgo::math::DVec<double> f(3);
    f << 1.0, -2.0, 0.5;
    const ConstantForceEnergy<double> energy{f};

    pgo::math::DVec<double> u(3);
    u << 3.0, 1.0, -2.0;

    double fused_value = 0;
    pgo::math::DVec<double> fused_g;
    pgo::math::SparseMat<double> fused_H;
    energy.value_gradient_hessian(u, fused_value, fused_g, fused_H);

    EXPECT_DOUBLE_EQ(energy.value(u), fused_value);
    pgo::math::DVec<double> g;
    energy.gradient(u, g);
    EXPECT_TRUE(g.isApprox(fused_g));
    EXPECT_EQ(fused_H.nonZeros(), 0);
}

} // namespace pgo::energy::test

namespace pgo::solver::test {

TEST(SolverStatus, StatusNameCoversAllEnumerators) {
    EXPECT_EQ(status_name(SolverStatus::converged), "converged");
    EXPECT_EQ(status_name(SolverStatus::max_iterations), "max_iterations");
    EXPECT_EQ(status_name(SolverStatus::regularization_failed), "regularization_failed");
    EXPECT_EQ(status_name(SolverStatus::line_search_failed), "line_search_failed");
}

} // namespace pgo::solver::test
