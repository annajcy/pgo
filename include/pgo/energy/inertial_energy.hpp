#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/math/types.hpp"

#include <vector>

namespace pgo::energy {

template <typename T>
class LumpedInertialEnergy {
    pgo::math::DVec<T> m_lumped_mass;
    pgo::math::DVec<T> m_u_hat;
    T m_inv_dt2;

public:
    LumpedInertialEnergy(const pgo::math::DVec<T>& lumped_mass,
                         const pgo::math::DVec<T>& u_hat,
                         const T dt)
        : m_lumped_mass{lumped_mass}, m_u_hat{u_hat} {
        pgo::base::require(dt > T{0}, "dt must be positive");
        pgo::base::require(static_cast<std::size_t>(lumped_mass.size()) == static_cast<std::size_t>(u_hat.size()),
                           "lumped_mass and u_hat must have the same size");
        for (pgo::math::DenseIndex i = 0; i < lumped_mass.size(); ++i) {
            pgo::base::require(lumped_mass[i] >= T{0}, "lumped mass must be non-negative");
        }
        m_inv_dt2 = T{1} / (dt * dt);
    }

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) ==
                               static_cast<std::size_t>(m_lumped_mass.size()),
                           "full_u size must match lumped_mass size");
        T e = T{0};
        for (pgo::math::DenseIndex i = 0; i < m_lumped_mass.size(); ++i) {
            const T diff = full_u[i] - m_u_hat[i];
            e += m_lumped_mass[i] * diff * diff;
        }
        return T{0.5} * m_inv_dt2 * e;
    }

    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) ==
                               static_cast<std::size_t>(m_lumped_mass.size()),
                           "full_u size must match lumped_mass size");
        full_g.resize(m_lumped_mass.size());
        for (pgo::math::DenseIndex i = 0; i < m_lumped_mass.size(); ++i) {
            full_g[i] = m_inv_dt2 * m_lumped_mass[i] * (full_u[i] - m_u_hat[i]);
        }
    }

    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) ==
                               static_cast<std::size_t>(m_lumped_mass.size()),
                           "full_u size must match lumped_mass size");
        std::vector<pgo::math::Triplet<T>> triplets;
        triplets.reserve(static_cast<std::size_t>(m_lumped_mass.size()));
        for (pgo::math::DenseIndex i = 0; i < m_lumped_mass.size(); ++i) {
            triplets.emplace_back(i, i, m_inv_dt2 * m_lumped_mass[i]);
        }
        const auto n = static_cast<pgo::math::DenseIndex>(m_lumped_mass.size());
        full_H.resize(n, n);
        full_H.setFromTriplets(triplets.begin(), triplets.end());
    }

    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& value,
                                pgo::math::DVec<T>& full_g, pgo::math::SparseMat<T>& full_H) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) ==
                               static_cast<std::size_t>(m_lumped_mass.size()),
                           "full_u size must match lumped_mass size");
        full_g.resize(m_lumped_mass.size());
        std::vector<pgo::math::Triplet<T>> triplets;
        triplets.reserve(static_cast<std::size_t>(m_lumped_mass.size()));

        T e = T{0};
        for (pgo::math::DenseIndex i = 0; i < m_lumped_mass.size(); ++i) {
            const T diff = full_u[i] - m_u_hat[i];
            e += m_lumped_mass[i] * diff * diff;
            full_g[i] = m_inv_dt2 * m_lumped_mass[i] * diff;
            triplets.emplace_back(i, i, m_inv_dt2 * m_lumped_mass[i]);
        }
        value = T{0.5} * m_inv_dt2 * e;

        const auto n = static_cast<pgo::math::DenseIndex>(m_lumped_mass.size());
        full_H.resize(n, n);
        full_H.setFromTriplets(triplets.begin(), triplets.end());
    }
};

} // namespace pgo::energy
