#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/dof/reduced_dof_map.hpp"
#include "pgo/energy/assembled_energy.hpp"
#include "pgo/energy/energy_sum.hpp"
#include "pgo/energy/mass_spring_local_energy_provider.hpp"
#include "pgo/energy/reduced_energy.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/solver/newton_solver.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace pgo::solver::test {

// --- quadratic convergence test ---

class SpdQuadraticEnergy {
    pgo::math::DMat<double> m_A;
    pgo::math::DVec<double> m_b;

public:
    SpdQuadraticEnergy(const pgo::math::DMat<double>& A, const pgo::math::DVec<double>& b)
        : m_A{A}, m_b{b} {}

    [[nodiscard]] double value(const pgo::math::DVec<double>& z) const {
        const pgo::math::DVec<double> Az = m_A * z;
        return 0.5 * z.dot(Az) - m_b.dot(z);
    }

    void gradient(const pgo::math::DVec<double>& z, pgo::math::DVec<double>& g) const {
        g = m_A * z - m_b;
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
        g = m_A * z - m_b;
        v = 0.5 * z.dot(g + m_b) - m_b.dot(z);
        hessian(z, H);
    }
};

static_assert(pgo::energy::DifferentiableEnergy<SpdQuadraticEnergy, double>);

TEST(NewtonSolver, ConvergesToOneStepOnSpdQuadratic) {
    const pgo::math::DMat<double> A = (pgo::math::DMat<double>(2, 2) << 2, 0, 0, 3).finished();
    pgo::math::DVec<double> b(2);
    b << 1, 2;
    const SpdQuadraticEnergy energy{A, b};

    NewtonOptions<double> options{};

    pgo::math::DVec<double> z(2);
    z << 0, 0;

    const SolverResult<double> result = solve_newton(energy, z, options);

    EXPECT_EQ(SolverStatus::converged, result.status);
    EXPECT_EQ(1, result.iterations);
    EXPECT_NEAR(0.5, z[0], 1e-10);       // x* = 1/2
    EXPECT_NEAR(2.0 / 3.0, z[1], 1e-10); // y* = 2/3
}

// --- diagonal regularization test ---

// E(x,y) = -0.5*x^2 + 0.5*y^2 + 0.25*x^4  (double-well in x, quadratic in y)
// Minimizer: (1, 0) or (-1, 0)
// At (x=0.5, y=0): H = [[-0.25,0],[0,1]] — indefinite, Newton direction is not descent
class DoubleWellEnergy {
public:
    [[nodiscard]] double value(const pgo::math::DVec<double>& z) const {
        const double x = z[0];
        const double y = z[1];
        return -0.5 * x * x + 0.5 * y * y + 0.25 * x * x * x * x;
    }

    void gradient(const pgo::math::DVec<double>& z, pgo::math::DVec<double>& g) const {
        g.resize(2);
        const double x = z[0];
        g[0] = x * x * x - x;
        g[1] = z[1];
    }

    void hessian(const pgo::math::DVec<double>& z, pgo::math::SparseMat<double>& H) const {
        const double h00 = 3.0 * z[0] * z[0] - 1.0;
        std::vector<pgo::math::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, h00);
        triplets.emplace_back(1, 1, 1.0);
        H.resize(2, 2);
        H.setFromTriplets(triplets.begin(), triplets.end());
    }

    void value_gradient_hessian(const pgo::math::DVec<double>& z, double& v,
                                pgo::math::DVec<double>& g, pgo::math::SparseMat<double>& H) const {
        gradient(z, g);
        const double x = z[0];
        const double y = z[1];
        v = -0.5 * x * x + 0.5 * y * y + 0.25 * x * x * x * x;
        hessian(z, H);
    }
};

static_assert(pgo::energy::DifferentiableEnergy<DoubleWellEnergy, double>);

TEST(NewtonSolver, AddsRegularizationWhenHessianIsNotDescent) {
    const DoubleWellEnergy energy{};

    pgo::math::DVec<double> z(2);
    z << 0.5, 0.0;  // H = [[-0.25,0],[0,1]], Newton du = [-1.5,0], g·du = 0.5625 ≥ 0 → not descent

    const SolverResult<double> result = solve_newton(energy, z);

    // Should converge to (1, 0) — the minimizer in the positive x basin
    EXPECT_EQ(SolverStatus::converged, result.status);
    EXPECT_NEAR(1.0, z[0], 1e-6);
    EXPECT_NEAR(0.0, z[1], 1e-8);
}

// --- reduced mass-spring smoke tests ---

TEST(NewtonSolverMassSpring, ConvergesAtRestState) {
    // 3-point 1D chain: vertices at x=0, 1, 2; edges (0,1), (1,2)
    const pgo::storage::HostBuffer<double> rest = {0.0, 1.0, 2.0};
    const pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edges = {0, 1, 1, 2};
    const pgo::geometry::RestMesh<double, 1> mesh{rest, edges, {}};

    const pgo::energy::MassSpringLocalEnergyProvider<double, 1> provider{mesh, 1.0};
    const pgo::energy::AssembledEnergy<double, pgo::energy::MassSpringLocalEnergyProvider<double, 1>> full_energy{
        provider};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);  // fix vertex 0
    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};

    const pgo::energy::ReducedEnergyView<double,
        pgo::energy::AssembledEnergy<double, pgo::energy::MassSpringLocalEnergyProvider<double, 1>>>
        reduced{full_energy, dof_map};

    // start at rest (free DOFs 1,2 = 0)
    pgo::math::DVec<double> free_u(2);
    free_u << 0.0, 0.0;

    const SolverResult<double> result = solve_newton(reduced, free_u);

    EXPECT_EQ(SolverStatus::converged, result.status);
    EXPECT_NEAR(0.0, free_u[0], 1e-10);
    EXPECT_NEAR(0.0, free_u[1], 1e-10);
}

class ConstantForceEnergy {
    pgo::math::DVec<double> m_force;

public:
    explicit ConstantForceEnergy(const pgo::math::DVec<double>& force) : m_force{force} {}

    [[nodiscard]] double value(const pgo::math::DVec<double>& u) const {
        return -m_force.dot(u);
    }

    void gradient(const pgo::math::DVec<double>&, pgo::math::DVec<double>& g) const {
        g = -m_force;
    }

    void hessian(const pgo::math::DVec<double>&, pgo::math::SparseMat<double>& H) const {
        std::vector<pgo::math::Triplet<double>> triplets;
        H.resize(static_cast<pgo::math::DenseIndex>(m_force.size()),
                 static_cast<pgo::math::DenseIndex>(m_force.size()));
        H.setFromTriplets(triplets.begin(), triplets.end());
    }

    void value_gradient_hessian(const pgo::math::DVec<double>& u, double& v,
                                pgo::math::DVec<double>& g, pgo::math::SparseMat<double>& H) const {
        g = -m_force;
        v = -m_force.dot(u);
        hessian(u, H);
    }
};

static_assert(pgo::energy::FullEnergy<ConstantForceEnergy, double>);

TEST(NewtonSolverMassSpring, ForcePullsFreeVertexInExpectedDirection) {
    // 3-point 1D chain, fix vertex 0, apply force +1.0 at vertex 2
    const pgo::storage::HostBuffer<double> rest = {0.0, 1.0, 2.0};
    const pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edges = {0, 1, 1, 2};
    const pgo::geometry::RestMesh<double, 1> mesh{rest, edges, {}};

    const pgo::energy::MassSpringLocalEnergyProvider<double, 1> provider{mesh, 1.0};
    const pgo::energy::AssembledEnergy<double, pgo::energy::MassSpringLocalEnergyProvider<double, 1>> ms_energy{
        provider};

    pgo::math::DVec<double> force(3);
    force << 0.0, 0.0, 1.0;
    const ConstantForceEnergy force_energy{force};

    const pgo::energy::EnergySum<double,
        pgo::energy::AssembledEnergy<double, pgo::energy::MassSpringLocalEnergyProvider<double, 1>>,
        ConstantForceEnergy>
        total{ms_energy, force_energy};

    pgo::dof::DirichletBoundary<double> boundary;
    boundary.fix_dof(0);
    const pgo::dof::ReducedDofMap<double> dof_map{3, boundary};

    const pgo::energy::ReducedEnergyView<double, decltype(total)> reduced{total, dof_map};

    pgo::math::DVec<double> free_u(2);
    free_u << 0.0, 0.0;

    const double initial_value = reduced.value(free_u);

    const SolverResult<double> result = solve_newton(reduced, free_u);

    EXPECT_EQ(SolverStatus::converged, result.status);
    EXPECT_LT(result.final_value, initial_value);  // energy decreased

    // vertex 0 is fixed at 0 (not in free_u)
    // vertex 2 should be pulled to the right (positive displacement)
    EXPECT_GT(free_u[1], 0.0);
    // vertex 1 should be pulled right too (via spring coupling)
    EXPECT_GT(free_u[0], 0.0);
    // vertex 2 displacement > vertex 1 displacement (further from fixed point)
    EXPECT_GT(free_u[1], free_u[0]);
}

} // namespace pgo::solver::test
