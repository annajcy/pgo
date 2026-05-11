#include "pgo/solver/feasible_set.hpp"
#include "pgo/solver/line_search.hpp"
#include "pgo/solver/solver_options.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace pgo::solver::test {

class QuadraticEnergy1D {
public:
    [[nodiscard]] double value(const pgo::math::DVec<double>& z) const {
        return 0.5 * z[0] * z[0];
    }

    void gradient(const pgo::math::DVec<double>& z, pgo::math::DVec<double>& g) const {
        g.resize(1);
        g[0] = z[0];
    }

    void hessian(const pgo::math::DVec<double>&, pgo::math::SparseMat<double>& H) const {
        std::vector<pgo::math::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 1.0);
        H.resize(1, 1);
        H.setFromTriplets(triplets.begin(), triplets.end());
    }

    void value_gradient_hessian(const pgo::math::DVec<double>& z, double& v,
                                pgo::math::DVec<double>& g, pgo::math::SparseMat<double>& H) const {
        v = value(z);
        gradient(z, g);
        hessian(z, H);
    }
};

static_assert(pgo::energy::DifferentiableEnergy<QuadraticEnergy1D, double>);

class ToyFeasibleSet {
public:
    [[nodiscard]] bool is_feasible(const pgo::math::DVec<double>& z) const {
        return z[0] <= 100.0;
    }

    [[nodiscard]] double max_step(const pgo::math::DVec<double>&, const pgo::math::DVec<double>&) const {
        return 0.25;
    }
};

static_assert(FeasibleSetLike<ToyFeasibleSet, double>);

TEST(LineSearch, AcceptsFullStepForQuadraticDescent) {
    const QuadraticEnergy1D energy{};
    const AlwaysFeasible<double> feasible{};
    LineSearchOptions<double> options{};

    pgo::math::DVec<double> z(1);
    z[0] = 1.0;
    pgo::math::DVec<double> dz(1);
    dz[0] = -1.0;           // Newton step for 0.5*z^2: H^-1 * (-g) = -z

    pgo::math::DVec<double> g;
    energy.gradient(z, g);
    const double current_value = energy.value(z);

    const double alpha = feasible_armijo_line_search(energy, feasible, z, dz, g, current_value, options);

    EXPECT_DOUBLE_EQ(1.0, alpha);
}

TEST(LineSearch, BacktracksWhenArmijoNotSatisfied) {
    const QuadraticEnergy1D energy{};
    const AlwaysFeasible<double> feasible{};
    LineSearchOptions<double> options{};
    options.armijo_c = 1.0;  // very strict: requires f(x+αd) ≤ f(x) + α * g·d
                              // With this strict setting, full step for 0.5z^2 won't suffice:
                              // f(0) = 0.5, f(0) = 0
                              // Armijo: 0.5 ≤ 1.0 + 1.0 * 1.0 * 1.0 * (-1.0) = 0 — fails
                              // Actually let me check... current_value = 0.5, gradient.dot(dz) = 1*(-1) = -1
                              // condition: 0.5 ≤ 0.5 + 1.0 * 1.0 * (-1.0) = -0.5 → FALSE
                              // After backtrack: alpha=0.5, trial=0.5, f=0.125
                              // 0.125 ≤ 0.5 + 1.0 * 0.5 * (-1.0) = 0.0 → FALSE
                              // After more backtrack: alpha=0.25, trial=0.75, f=0.28125
                              // 0.28125 ≤ 0.5 + 1.0 * 0.25 * (-1.0) = 0.25 → FALSE...
                              // This will keep going until min_step, then return 0.

    pgo::math::DVec<double> z(1);
    z[0] = 1.0;
    pgo::math::DVec<double> dz(1);
    dz[0] = -1.0;

    pgo::math::DVec<double> g;
    energy.gradient(z, g);
    const double current_value = energy.value(z);

    const double alpha = feasible_armijo_line_search(energy, feasible, z, dz, g, current_value, options);

    EXPECT_LT(alpha, 1.0);
}

TEST(LineSearch, AlwaysFeasibleReturnsTrueAndMaxStepOne) {
    const AlwaysFeasible<double> feasible{};

    pgo::math::DVec<double> z(3);
    z << 1.0, -2.0, 0.5;
    pgo::math::DVec<double> dz(3);
    dz << 0.1, 0.2, -0.3;

    EXPECT_TRUE(feasible.is_feasible(z));
    EXPECT_DOUBLE_EQ(1.0, feasible.max_step(z, dz));
}

TEST(LineSearch, FeasibilityCapsInitialAlpha) {
    const QuadraticEnergy1D energy{};
    const ToyFeasibleSet feasible{};
    LineSearchOptions<double> options{};
    options.armijo_c = 1e-4;

    pgo::math::DVec<double> z(1);
    z[0] = 0.5;
    pgo::math::DVec<double> dz(1);
    dz[0] = -0.5;  // descent

    pgo::math::DVec<double> g;
    energy.gradient(z, g);
    const double current_value = energy.value(z);

    const double alpha = feasible_armijo_line_search(energy, feasible, z, dz, g, current_value, options);

    // max_step=0.25 caps alpha, and the quadratic should accept it
    EXPECT_LE(alpha, 0.25);
    EXPECT_GT(alpha, 0.0);
}

TEST(ArmijoBacktrack, AcceptsFullStep) {
    const QuadraticEnergy1D energy{};
    LineSearchOptions<double> options{};

    pgo::math::DVec<double> z(1);
    z[0] = 1.0;
    pgo::math::DVec<double> dz(1);
    dz[0] = -1.0;  // Newton step for 0.5*z^2

    pgo::math::DVec<double> g;
    energy.gradient(z, g);
    const double current_value = energy.value(z);

    const double alpha = armijo_backtrack(energy, z, dz, g, current_value, options);

    EXPECT_DOUBLE_EQ(1.0, alpha);
}

TEST(ArmijoBacktrack, BacktracksWhenStrict) {
    const QuadraticEnergy1D energy{};
    LineSearchOptions<double> options{};
    options.armijo_c = 1.0;

    pgo::math::DVec<double> z(1);
    z[0] = 1.0;
    pgo::math::DVec<double> dz(1);
    dz[0] = -1.0;

    pgo::math::DVec<double> g;
    energy.gradient(z, g);
    const double current_value = energy.value(z);

    const double alpha = armijo_backtrack(energy, z, dz, g, current_value, options);

    EXPECT_LT(alpha, 1.0);
}

} // namespace pgo::solver::test
