#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/math/types.hpp"

namespace pgo::energy {

template <typename T>
class ConstantForceEnergy {
    const pgo::math::DVec<T>* m_force;

public:
    explicit ConstantForceEnergy(const pgo::math::DVec<T>& force)
        : m_force{&force} {}

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) == static_cast<std::size_t>(m_force->size()),
                           "full_u size must match force size");
        return -m_force->dot(full_u);
    }

    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) == static_cast<std::size_t>(m_force->size()),
                           "full_u size must match force size");
        full_g = -(*m_force);
    }

    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) == static_cast<std::size_t>(m_force->size()),
                           "full_u size must match force size");
        const auto n = static_cast<pgo::math::DenseIndex>(m_force->size());
        full_H.resize(n, n);
        full_H.setZero();
    }

    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& val,
                                pgo::math::DVec<T>& full_g, pgo::math::SparseMat<T>& full_H) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) == static_cast<std::size_t>(m_force->size()),
                           "full_u size must match force size");
        val = -m_force->dot(full_u);
        full_g = -(*m_force);
        const auto n = static_cast<pgo::math::DenseIndex>(m_force->size());
        full_H.resize(n, n);
        full_H.setZero();
    }
};

} // namespace pgo::energy
