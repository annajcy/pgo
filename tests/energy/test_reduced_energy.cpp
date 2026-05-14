#include "pgo/core/dof/dirichlet_boundary.hpp"
#include "pgo/core/energy/reduced_energy.hpp"
#include "pgo/core/math/types.hpp"

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
static_assert(DifferentiableEnergy<ReducedEnergyView<double, ParametricQuadraticEnergy>, double>);

TEST(ReducedEnergyView, ReducedGradientMatchesManualProjection_ZeroPrescribed) {
    // 3-DOF system, fix DOF 0 to 0
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    pgo::math::DVec<double> b(3);
    b << 1, 2, 3;
    const ParametricQuadraticEnergy full_energy{A, b, 0.0};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);

    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};
    const ReducedEnergyView<double, ParametricQuadraticEnergy> reduced{full_energy, dof_map};

    // free_u corresponds to DOFs [1, 2]
    pgo::math::DVec<double> free_u(2);
    free_u << 0.5, -1.0;

    const pgo::math::DVec<double> full_u = dof_map.scatter_solution(free_u);
    pgo::math::DVec<double> full_g;
    full_energy.gradient(full_u, full_g);
    const pgo::math::DVec<double> expected_reduced_g = dof_map.restrict_vector_to_free(full_g);

    pgo::math::DVec<double> reduced_g;
    reduced.gradient(free_u, reduced_g);

    ASSERT_EQ(expected_reduced_g.size(), reduced_g.size());
    EXPECT_TRUE(expected_reduced_g.isApprox(reduced_g));
}

TEST(ReducedEnergyView, ReducedHessianMatchesManualProjection_ZeroPrescribed) {
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    pgo::math::DVec<double> b(3);
    b << 1, 2, 3;
    const ParametricQuadraticEnergy full_energy{A, b, 0.0};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);

    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};
    const ReducedEnergyView<double, ParametricQuadraticEnergy> reduced{full_energy, dof_map};

    pgo::math::DVec<double> free_u(2);
    free_u << 0.5, -1.0;

    const pgo::math::DVec<double> full_u = dof_map.scatter_solution(free_u);
    pgo::math::SparseMat<double> full_H;
    full_energy.hessian(full_u, full_H);
    const pgo::math::SparseMat<double> expected_reduced_H = dof_map.restrict_matrix_to_free(full_H);

    pgo::math::SparseMat<double> reduced_H;
    reduced.hessian(free_u, reduced_H);

    EXPECT_TRUE(expected_reduced_H.toDense().isApprox(reduced_H.toDense()));
}

TEST(ReducedEnergyView, NonZeroPrescribedDisplacement_EffectInGradientAndValue) {
    // Fix DOF 0 to prescribed value 0.5
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(3, 3) << 2, 1, 0, 1, 3, 0, 0, 0, 1).finished();
    pgo::math::DVec<double> b(3);
    b << 1, 2, 3;
    const ParametricQuadraticEnergy full_energy{A, b, 0.0};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.prescribe_dof(0, 0.5);

    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};
    const ReducedEnergyView<double, ParametricQuadraticEnergy> reduced{full_energy, dof_map};

    pgo::math::DVec<double> free_u(2);
    free_u << 0.5, -1.0;

    // value should equal full_energy.value(scatter_solution(free_u))
    const pgo::math::DVec<double> full_u = dof_map.scatter_solution(free_u);
    EXPECT_DOUBLE_EQ(full_energy.value(full_u), reduced.value(free_u));

    // gradient: verify that NON-ZERO prescribed value is accounted for
    // (this guards against mistakenly using eliminate_rhs_for_dirichlet on the gradient)
    const pgo::math::DVec<double> expected_reduced_g = dof_map.restrict_vector_to_free(
        [&]() {
            pgo::math::DVec<double> fg;
            full_energy.gradient(full_u, fg);
            return fg;
        }());

    pgo::math::DVec<double> reduced_g;
    reduced.gradient(free_u, reduced_g);
    EXPECT_TRUE(expected_reduced_g.isApprox(reduced_g));

    // The gradient should NOT equal restrict_vector_to_free of the all-zero-prescribed case
    // because prescribed value 0.5 affects the energy gradient
    pgo::dof::DirichletBoundary<double> zero_boundary;
    zero_boundary.fix_dof(0);
    const pgo::dof::ReducedDofMap<double> zero_dof_map{3, zero_boundary};
    const pgo::math::DVec<double> zero_full_u = zero_dof_map.scatter_solution(free_u);
    pgo::math::DVec<double> zero_full_g;
    full_energy.gradient(zero_full_u, zero_full_g);
    const pgo::math::DVec<double> zero_reduced_g = zero_dof_map.restrict_vector_to_free(zero_full_g);

    // With off-diagonal A(0,1)=1, the prescribed value 0.5 at DOF 0 should make a difference
    EXPECT_FALSE(zero_reduced_g.isApprox(reduced_g));
}

TEST(ReducedEnergyView, ValueGradientHessian_FusedMatchesSeparate) {
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    pgo::math::DVec<double> b(3);
    b << 1, 2, 3;
    const ParametricQuadraticEnergy full_energy{A, b, 0.0};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);

    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};
    const ReducedEnergyView<double, ParametricQuadraticEnergy> reduced{full_energy, dof_map};

    pgo::math::DVec<double> free_u(2);
    free_u << 0.5, -1.0;

    double fused_value = 0;
    pgo::math::DVec<double> fused_g;
    pgo::math::SparseMat<double> fused_H;
    reduced.value_gradient_hessian(free_u, fused_value, fused_g, fused_H);

    EXPECT_DOUBLE_EQ(reduced.value(free_u), fused_value);

    pgo::math::DVec<double> g;
    reduced.gradient(free_u, g);
    EXPECT_TRUE(g.isApprox(fused_g));

    pgo::math::SparseMat<double> H;
    reduced.hessian(free_u, H);
    EXPECT_TRUE(H.toDense().isApprox(fused_H.toDense()));
}

TEST(ReducedEnergyView, ScatterSolutionAndScatterDirectionForwardCorrectly) {
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(3, 3) << 2, 0, 0, 0, 3, 0, 0, 0, 1).finished();
    pgo::math::DVec<double> b(3);
    b << 1, 2, 3;
    const ParametricQuadraticEnergy full_energy{A, b, 0.0};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.prescribe_dof(0, 0.5);

    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};
    const ReducedEnergyView<double, ParametricQuadraticEnergy> reduced{full_energy, dof_map};

    pgo::math::DVec<double> free_u(2);
    free_u << 0.5, -1.0;

    const pgo::math::DVec<double> sol = reduced.scatter_solution(free_u);
    const pgo::math::DVec<double> dir = reduced.scatter_direction(free_u);

    EXPECT_DOUBLE_EQ(0.5, sol[0]);               // prescribed value at DOF 0
    EXPECT_DOUBLE_EQ(free_u[0], sol[1]);          // free DOF 1
    EXPECT_DOUBLE_EQ(free_u[1], sol[2]);          // free DOF 2
    EXPECT_DOUBLE_EQ(0.0, dir[0]);                // fixed DOF is zero in direction
    EXPECT_DOUBLE_EQ(free_u[0], dir[1]);
    EXPECT_DOUBLE_EQ(free_u[1], dir[2]);
}

TEST(ReducedEnergyView, FullDofsAndFreeDofsMatchMap) {
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(4, 4) <<
        1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4).finished();
    pgo::math::DVec<double> b(4);
    b << 0, 0, 0, 0;
    const ParametricQuadraticEnergy full_energy{A, b, 0.0};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);
    boundary.fix_dof(3);

    const pgo::dof::ReducedDofMap<double> dof_map{4, boundary};
    const ReducedEnergyView<double, ParametricQuadraticEnergy> reduced{full_energy, dof_map};

    EXPECT_EQ(4, reduced.full_dofs());
    EXPECT_EQ(2, reduced.free_dofs());
}

} // namespace pgo::energy::test
