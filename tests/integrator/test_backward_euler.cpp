#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/dof/reduced_dof_map.hpp"
#include "pgo/energy/energy_concepts.hpp"
#include "pgo/energy/inertial_energy.hpp"
#include "pgo/integrator/backward_euler.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace pgo::integrator::test {

// --- LumpedInertialEnergy tests ---

static_assert(pgo::energy::FullEnergy<pgo::energy::LumpedInertialEnergy<double>, double>);

TEST(InertialEnergy, ValueGradientHessianMatchHandCalc) {
    // mass = [2, 3], u_hat = [1, 0], dt = 0.1 → inv_dt2 = 100
    pgo::math::DVec<double> mass(2);
    mass << 2, 3;
    pgo::math::DVec<double> u_hat(2);
    u_hat << 1, 0;
    const pgo::energy::LumpedInertialEnergy<double> energy{mass, u_hat, 0.1};

    pgo::math::DVec<double> u(2);
    u << 1.5, -0.2;

    // value = 50 * (2*(0.5)^2 + 3*(-0.2)^2) = 50 * (0.5 + 0.12) = 31
    EXPECT_DOUBLE_EQ(31.0, energy.value(u));

    pgo::math::DVec<double> g;
    energy.gradient(u, g);
    // g0 = 100 * 2 * 0.5 = 100, g1 = 100 * 3 * (-0.2) = -60
    EXPECT_DOUBLE_EQ(100.0, g[0]);
    EXPECT_DOUBLE_EQ(-60.0, g[1]);

    pgo::math::SparseMat<double> H;
    energy.hessian(u, H);
    EXPECT_EQ(2, H.rows());
    EXPECT_EQ(2, H.cols());
    EXPECT_DOUBLE_EQ(200.0, H.coeff(0, 0));  // 2 / 0.01
    EXPECT_DOUBLE_EQ(300.0, H.coeff(1, 1));  // 3 / 0.01
    EXPECT_DOUBLE_EQ(0.0, H.coeff(0, 1));
}

TEST(InertialEnergy, FusedAPIEqualsSeparateCalls) {
    pgo::math::DVec<double> mass(2);
    mass << 2, 3;
    pgo::math::DVec<double> u_hat(2);
    u_hat << 1, 0;
    const pgo::energy::LumpedInertialEnergy<double> energy{mass, u_hat, 0.1};

    pgo::math::DVec<double> u(2);
    u << 1.5, -0.2;

    double fused_value = 0;
    pgo::math::DVec<double> fused_g;
    pgo::math::SparseMat<double> fused_H;
    energy.value_gradient_hessian(u, fused_value, fused_g, fused_H);

    EXPECT_DOUBLE_EQ(energy.value(u), fused_value);

    pgo::math::DVec<double> g;
    energy.gradient(u, g);
    EXPECT_TRUE(g.isApprox(fused_g));

    pgo::math::SparseMat<double> H;
    energy.hessian(u, H);
    EXPECT_TRUE(H.toDense().isApprox(fused_H.toDense()));
}

TEST(InertialEnergy, RejectsInvalidInputs) {
    pgo::math::DVec<double> mass(2);
    mass << 2, 3;
    pgo::math::DVec<double> u_hat(2);
    u_hat << 1, 0;

    EXPECT_THROW(pgo::energy::LumpedInertialEnergy<double>(mass, u_hat, 0.0), std::runtime_error);
    EXPECT_THROW(pgo::energy::LumpedInertialEnergy<double>(mass, u_hat, -0.1), std::runtime_error);

    pgo::math::DVec<double> neg_mass(2);
    neg_mass << -1, 3;
    EXPECT_THROW(pgo::energy::LumpedInertialEnergy<double>(neg_mass, u_hat, 0.1), std::runtime_error);
}

// --- BackwardEuler smoke test ---

// FullEnergy quadratic: E(u) = 0.5*u^T*A*u
class SpdQuadraticEnergy {
    pgo::math::DMat<double> m_A;

public:
    explicit SpdQuadraticEnergy(const pgo::math::DMat<double>& A) : m_A{A} {}

    [[nodiscard]] double value(const pgo::math::DVec<double>& z) const {
        return 0.5 * z.dot(m_A * z);
    }

    void gradient(const pgo::math::DVec<double>& z, pgo::math::DVec<double>& g) const {
        g = m_A * z;
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

    void value_gradient_hessian(const pgo::math::DVec<double>& z, double& v,
                                pgo::math::DVec<double>& g, pgo::math::SparseMat<double>& H) const {
        g = m_A * z;
        v = 0.5 * z.dot(g);
        hessian(z, H);
    }
};

static_assert(pgo::energy::FullEnergy<SpdQuadraticEnergy, double>);

TEST(BackwardEuler, UpdatesStateAndPreservesFixedDof) {
    // 2 DOF system: fix DOF 0 to 0, potential = 0.5*u^T*A*u with A = [[2,0],[0,1]]
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(2, 2) << 2, 0, 0, 1).finished();
    const SpdQuadraticEnergy potential{A};

    // fix DOF 0
    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);
    const pgo::dof::ReducedDofMap<double> dof_map{2, boundary};

    const pgo::math::DVec<double> mass = []() {
        pgo::math::DVec<double> m(2);
        m << 1, 1;
        return m;
    }();

    pgo::integrator::DynamicState<double> state;
    state.u.resize(2);
    state.u << 0.5, 0.3;
    state.v.resize(2);
    state.v << 0.1, -0.2;

    const pgo::math::DVec<double> u_old = state.u;
    const double dt = 0.1;
    const pgo::integrator::BackwardEuler<double> integrator{};

    pgo::solver::NewtonOptions<double> options{};
    options.line_search.feasibility_safety = 1.0;

    const auto result = integrator.step(potential, mass, dof_map, state, dt, options);

    // potential energy of new state decreased vs old state
    EXPECT_LT(potential.value(state.u), potential.value(u_old));

    // solver reported convergence
    EXPECT_EQ(pgo::solver::SolverStatus::converged, result.status);
    EXPECT_DOUBLE_EQ(dt, result.dt);
    EXPECT_GT(result.solver_iterations, 0);

    // fixed DOF stays at prescribed value (0)
    EXPECT_DOUBLE_EQ(0.0, state.u[0]);

    // v = (u_new - u_old) / dt
    EXPECT_DOUBLE_EQ((state.u[0] - u_old[0]) / dt, state.v[0]);
    EXPECT_DOUBLE_EQ((state.u[1] - u_old[1]) / dt, state.v[1]);

}

} // namespace pgo::integrator::test
