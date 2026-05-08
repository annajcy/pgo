#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/math/backend.hpp"

namespace pgo::math {

template <RealScalar T, class F>
[[nodiscard]] DVec<T> finite_difference_gradient(F&& value_function, const DVec<T>& x, const T eps) {
    pgo::base::require(eps > T{0}, "finite difference epsilon must be positive");

    auto&& f = value_function;
    DVec<T> gradient{x.size()};
    for (DenseIndex i = 0; i < x.size(); ++i) {
        auto x_plus = x;
        auto x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        gradient[i] = (f(x_plus) - f(x_minus)) / (T{2} * eps);
    }

    return gradient;
}

template <RealScalar T, class Grad>
[[nodiscard]] DMat<T> finite_difference_hessian_from_gradient(Grad&& gradient_function, const DVec<T>& x, const T eps) {
    pgo::base::require(eps > T{0}, "finite difference epsilon must be positive");

    auto&& grad = gradient_function;
    DMat<T> hessian{x.size(), x.size()};
    for (DenseIndex i = 0; i < x.size(); ++i) {
        auto x_plus = x;
        auto x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        const DVec<T> gradient_plus = grad(x_plus);
        const DVec<T> gradient_minus = grad(x_minus);
        pgo::base::require(gradient_plus.size() == x.size(), "finite difference gradient size mismatch");
        pgo::base::require(gradient_minus.size() == x.size(), "finite difference gradient size mismatch");

        hessian.col(i) = (gradient_plus - gradient_minus) / (T{2} * eps);
    }

    return hessian;
}

} // namespace pgo::math
