#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/energy/energy_concepts.hpp"
#include "pgo/solver/feasible_set.hpp"
#include "pgo/solver/line_search.hpp"
#include "pgo/solver/solver_options.hpp"
#include "pgo/solver/solver_result.hpp"

#include <Eigen/SparseCholesky>

namespace pgo::solver {

template <typename T>
[[nodiscard]] bool regularized_newton_direction(
    const pgo::math::SparseMat<T>& hessian,
    const pgo::math::DVec<T>& gradient,
    pgo::math::DVec<T>& du,
    const NewtonOptions<T>& options) {

    pgo::base::require(options.regularization_growth > T{1},
                       "regularization_growth must be > 1");
    pgo::base::require(options.min_regularization <= options.max_regularization,
                       "min_regularization must be <= max_regularization");
    pgo::base::require(options.initial_regularization <= options.max_regularization,
                       "initial_regularization must be <= max_regularization");

    T lambda = options.initial_regularization;

    while (lambda <= options.max_regularization) {
        pgo::math::SparseMat<T> H_mod = hessian;
        if (lambda > T{0}) {
            for (pgo::math::DenseIndex i = 0; i < hessian.rows(); ++i) {
                H_mod.coeffRef(i, i) += lambda;
            }
        }

        Eigen::SimplicialLDLT<pgo::math::SparseMat<T>> solver;
        solver.compute(H_mod);

        if (solver.info() != Eigen::Success) {
            lambda = std::max(options.min_regularization, lambda * options.regularization_growth);
            continue;
        }

        du = solver.solve(-gradient);

        if (solver.info() != Eigen::Success) {
            lambda = std::max(options.min_regularization, lambda * options.regularization_growth);
            continue;
        }

        if (!du.allFinite()) {
            lambda = std::max(options.min_regularization, lambda * options.regularization_growth);
            continue;
        }

        if (gradient.dot(du) >= T{0}) {
            lambda = std::max(options.min_regularization, lambda * options.regularization_growth);
            continue;
        }

        return true;
    }

    return false;
}

template <typename T, typename Energy, typename FeasibleSet>
    requires pgo::energy::DifferentiableEnergy<Energy, T> && FeasibleSetLike<FeasibleSet, T>
SolverResult<T> solve_newton(const Energy& energy, const FeasibleSet& feasible, pgo::math::DVec<T>& z,
                              const NewtonOptions<T>& options = {}) {

    T value{};
    pgo::math::DVec<T> gradient;
    pgo::math::SparseMat<T> hessian;
    pgo::math::DVec<T> du;

    for (std::size_t iter = 0; iter < options.max_iterations; ++iter) {
        energy.value_gradient_hessian(z, value, gradient, hessian);

        const T gradient_norm = gradient.norm();
        if (gradient_norm <= options.gradient_tolerance) {
            return {SolverStatus::converged, iter, value, gradient_norm};
        }

        if (!regularized_newton_direction(hessian, gradient, du, options)) {
            return {SolverStatus::regularization_failed, iter, value, gradient_norm};
        }

        const T alpha = feasible_armijo_line_search(energy, feasible, z, du, gradient, value, options.line_search);

        if (alpha <= options.line_search.min_step) {
            return {SolverStatus::line_search_failed, iter, value, gradient_norm};
        }

        z += alpha * du;
    }

    energy.value_gradient_hessian(z, value, gradient, hessian);
    return {SolverStatus::max_iterations, options.max_iterations, value, gradient.norm()};
}

template <typename T, typename Energy>
    requires pgo::energy::DifferentiableEnergy<Energy, T>
SolverResult<T> solve_newton(const Energy& energy, pgo::math::DVec<T>& z,
                              const NewtonOptions<T>& options = {}) {
    return solve_newton<T>(energy, AlwaysFeasible<T>{}, z, options);
}

} // namespace pgo::solver
