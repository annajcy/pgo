#include "pgo/energy/energy_sum.hpp"
#include "pgo/math/types.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace pgo::energy::test {

class ParametricQuadraticEnergy {
    pgo::math::DMat<double> m_A;
    pgo::math::DVec<double> m_b;
    double m_c;

public:
    ParametricQuadraticEnergy(const pgo::math::DMat<double>& A, const pgo::math::DVec<double>& b,
                              const double c)
        : m_A{A}, m_b{b}, m_c{c} {}

    [[nodiscard]] double value(const pgo::math::DVec<double>& u) const {
        return 0.5 * u.dot(m_A * u) - m_b.dot(u) + m_c;
    }

    void gradient(const pgo::math::DVec<double>& u, pgo::math::DVec<double>& g) const {
        g = m_A * u - m_b;
    }

    void hessian(const pgo::math::DVec<double>&, pgo::math::SparseMat<double>& H) const {
        std::vector<pgo::math::Triplet<double>> triplets;
        for (pgo::math::DenseIndex i = 0; i < m_A.rows(); ++i) {
            for (pgo::math::DenseIndex j = 0; j < m_A.cols(); ++j) {
                if (m_A(i, j) != 0.0) {
                    triplets.emplace_back(i, j, m_A(i, j));
                }
            }
        }
        H.resize(m_A.rows(), m_A.cols());
        H.setFromTriplets(triplets.begin(), triplets.end());
    }

    void value_gradient_hessian(const pgo::math::DVec<double>& u, double& value,
                                pgo::math::DVec<double>& g, pgo::math::SparseMat<double>& H) const {
        g = m_A * u - m_b;
        value = 0.5 * u.dot(g + m_b) - m_b.dot(u) + m_c;
        hessian(u, H);
    }
};

static_assert(DifferentiableEnergy<ParametricQuadraticEnergy, double>);
static_assert(FullEnergy<ParametricQuadraticEnergy, double>);

static_assert(FullEnergy<EnergySum<double, ParametricQuadraticEnergy, ParametricQuadraticEnergy>, double>);

TEST(EnergySum, ValueIsSumOfTwoEnergies) {
    const pgo::math::DMat<double> A1 = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    const pgo::math::DMat<double> A2 = (pgo::math::DMat<double>(3, 3) << 1, 0, 0, 0, 2, 0, 0, 0, 4).finished();

    pgo::math::DVec<double> b1(3);
    b1 << 1, 2, 3;
    pgo::math::DVec<double> b2(3);
    b2 << 0.5, -1, 2;

    const ParametricQuadraticEnergy e1{A1, b1, 0.5};
    const ParametricQuadraticEnergy e2{A2, b2, -0.3};
    const EnergySum<double, ParametricQuadraticEnergy, ParametricQuadraticEnergy> sum{e1, e2};

    pgo::math::DVec<double> u(3);
    u << 1.0, -0.5, 2.0;

    EXPECT_DOUBLE_EQ(e1.value(u) + e2.value(u), sum.value(u));
}

TEST(EnergySum, GradientIsSumOfTwoEnergies) {
    const pgo::math::DMat<double> A1 = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    const pgo::math::DMat<double> A2 = (pgo::math::DMat<double>(3, 3) << 1, 0, 0, 0, 2, 0, 0, 0, 4).finished();

    pgo::math::DVec<double> b1(3);
    b1 << 1, 2, 3;
    pgo::math::DVec<double> b2(3);
    b2 << 0.5, -1, 2;

    const ParametricQuadraticEnergy e1{A1, b1, 0.5};
    const ParametricQuadraticEnergy e2{A2, b2, -0.3};
    const EnergySum<double, ParametricQuadraticEnergy, ParametricQuadraticEnergy> sum{e1, e2};

    pgo::math::DVec<double> u(3);
    u << 1.0, -0.5, 2.0;

    pgo::math::DVec<double> g1, g2, g_sum;
    e1.gradient(u, g1);
    e2.gradient(u, g2);
    sum.gradient(u, g_sum);

    const pgo::math::DVec<double> expected = g1 + g2;
    ASSERT_EQ(expected.size(), g_sum.size());
    EXPECT_TRUE(expected.isApprox(g_sum));
}

TEST(EnergySum, HessianIsSumOfTwoEnergies) {
    const pgo::math::DMat<double> A1 = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    const pgo::math::DMat<double> A2 = (pgo::math::DMat<double>(3, 3) << 1, 0, 0, 0, 2, 0, 0, 0, 4).finished();

    pgo::math::DVec<double> b1(3);
    b1 << 1, 2, 3;
    pgo::math::DVec<double> b2(3);
    b2 << 0.5, -1, 2;

    const ParametricQuadraticEnergy e1{A1, b1, 0.5};
    const ParametricQuadraticEnergy e2{A2, b2, -0.3};
    const EnergySum<double, ParametricQuadraticEnergy, ParametricQuadraticEnergy> sum{e1, e2};

    pgo::math::DVec<double> u(3);
    u << 1.0, -0.5, 2.0;

    pgo::math::SparseMat<double> H1, H2, H_sum;
    e1.hessian(u, H1);
    e2.hessian(u, H2);
    sum.hessian(u, H_sum);

    const pgo::math::DMat<double> expected_dense = A1 + A2;
    const pgo::math::DMat<double> sum_dense = H_sum.toDense();
    EXPECT_TRUE(expected_dense.isApprox(sum_dense));
}

TEST(EnergySum, FusedAPIEqualsSeparateCalls) {
    const pgo::math::DMat<double> A1 = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    const pgo::math::DMat<double> A2 = (pgo::math::DMat<double>(3, 3) << 1, 0, 0, 0, 2, 0, 0, 0, 4).finished();

    pgo::math::DVec<double> b1(3);
    b1 << 1, 2, 3;
    pgo::math::DVec<double> b2(3);
    b2 << 0.5, -1, 2;

    const ParametricQuadraticEnergy e1{A1, b1, 0.5};
    const ParametricQuadraticEnergy e2{A2, b2, -0.3};
    const EnergySum<double, ParametricQuadraticEnergy, ParametricQuadraticEnergy> sum{e1, e2};

    pgo::math::DVec<double> u(3);
    u << 1.0, -0.5, 2.0;

    double fused_value = 0;
    pgo::math::DVec<double> fused_g;
    pgo::math::SparseMat<double> fused_H;
    sum.value_gradient_hessian(u, fused_value, fused_g, fused_H);

    EXPECT_DOUBLE_EQ(sum.value(u), fused_value);
    pgo::math::DVec<double> g;
    sum.gradient(u, g);
    EXPECT_TRUE(g.isApprox(fused_g));
    pgo::math::SparseMat<double> H;
    sum.hessian(u, H);
    EXPECT_TRUE(H.toDense().isApprox(fused_H.toDense()));
}

TEST(EnergySum, ThreeEnergiesSumCorrectly) {
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(2, 2) << 1, 0, 0, 1).finished();
    pgo::math::DVec<double> b1(2);
    b1 << 1, 0;
    pgo::math::DVec<double> b2(2);
    b2 << 0, 2;
    pgo::math::DVec<double> b3(2);
    b3 << -1, 1;

    const ParametricQuadraticEnergy e1{A, b1, 0.0};
    const ParametricQuadraticEnergy e2{A, b2, 1.0};
    const ParametricQuadraticEnergy e3{A, b3, 2.0};
    const EnergySum<double, ParametricQuadraticEnergy, ParametricQuadraticEnergy, ParametricQuadraticEnergy>
        sum{e1, e2, e3};

    pgo::math::DVec<double> u(2);
    u << 3.0, -1.0;

    const double expected_value = e1.value(u) + e2.value(u) + e3.value(u);
    EXPECT_DOUBLE_EQ(expected_value, sum.value(u));

    pgo::math::DVec<double> g;
    sum.gradient(u, g);
    pgo::math::DVec<double> g_expected;
    e1.gradient(u, g_expected);
    pgo::math::DVec<double> g_tmp;
    e2.gradient(u, g_tmp);
    g_expected += g_tmp;
    e3.gradient(u, g_tmp);
    g_expected += g_tmp;
    EXPECT_TRUE(g_expected.isApprox(g));
}

} // namespace pgo::energy::test
