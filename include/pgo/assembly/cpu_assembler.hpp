#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/energy/energy_concepts.hpp"
#include "pgo/math/types.hpp"

#include <cstddef>
#include <vector>

namespace pgo::assembly {

namespace detail {

template <typename T>
inline void require_dofs_in_range(const std::vector<std::size_t>& dofs, const pgo::math::DVec<T>& full_u) {
    for (const auto dof : dofs) {
        pgo::base::require(dof < static_cast<std::size_t>(full_u.size()), "local DOF index is out of range");
    }
}

template <typename T>
inline void require_local_gradient_size(const std::vector<std::size_t>& dofs, const LocalVector<T>& local_g) {
    pgo::base::require(dofs.size() == static_cast<std::size_t>(local_g.size()),
                       "local gradient size does not match local DOF count");
}

template <typename T>
inline void require_local_hessian_size(const std::vector<std::size_t>& dofs, const LocalMatrix<T>& local_H) {
    pgo::base::require(dofs.size() == static_cast<std::size_t>(local_H.rows()) &&
                           dofs.size() == static_cast<std::size_t>(local_H.cols()),
                       "local Hessian size does not match local DOF count");
}

} // namespace detail

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
[[nodiscard]] T assemble_value(const EnergyProvider& energy_provider, const pgo::math::DVec<T>& full_u) {
    T value{};
    for (std::size_t local_id = 0; local_id < energy_provider.local_count(); ++local_id) {
        value += energy_provider.local_value(local_id, full_u);
    }
    return value;
}

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
void assemble_gradient(const EnergyProvider& energy_provider, const pgo::math::DVec<T>& full_u,
                       pgo::math::DVec<T>& full_gradient) {
    full_gradient.resize(full_u.size());
    full_gradient.setZero();

    std::vector<std::size_t> dofs;
    LocalVector<T> local_g;
    for (std::size_t local_id = 0; local_id < energy_provider.local_count(); ++local_id) {
        energy_provider.local_dofs(local_id, dofs);
        detail::require_dofs_in_range(dofs, full_u);
        energy_provider.local_gradient(local_id, full_u, local_g);
        detail::require_local_gradient_size(dofs, local_g);

        for (std::size_t a = 0; a < dofs.size(); ++a) {
            full_gradient[pgo::math::dense_index(dofs[a])] += local_g[pgo::math::dense_index(a)];
        }
    }
}

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
void assemble_hessian(const EnergyProvider& energy_provider, const pgo::math::DVec<T>& full_u,
                      pgo::math::SparseMat<T>& full_hessian) {
    std::vector<pgo::math::Triplet<T>> triplets;
    std::vector<std::size_t> dofs;
    LocalMatrix<T> local_H;

    for (std::size_t local_id = 0; local_id < energy_provider.local_count(); ++local_id) {
        energy_provider.local_dofs(local_id, dofs);
        detail::require_dofs_in_range(dofs, full_u);
        energy_provider.local_hessian(local_id, full_u, local_H);
        detail::require_local_hessian_size(dofs, local_H);

        for (std::size_t a = 0; a < dofs.size(); ++a) {
            for (std::size_t b = 0; b < dofs.size(); ++b) {
                triplets.emplace_back(pgo::math::dense_index(dofs[a]), pgo::math::dense_index(dofs[b]),
                                      local_H(pgo::math::dense_index(a), pgo::math::dense_index(b)));
            }
        }
    }

    full_hessian.resize(full_u.size(), full_u.size());
    full_hessian.setFromTriplets(triplets.begin(), triplets.end());
}

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
void assemble_value_gradient_hessian(const EnergyProvider& energy_provider, const pgo::math::DVec<T>& full_u, T& value,
                                     pgo::math::DVec<T>& full_gradient, pgo::math::SparseMat<T>& full_hessian) {
    value = T{};
    full_gradient.resize(full_u.size());
    full_gradient.setZero();

    std::vector<pgo::math::Triplet<T>> triplets;
    std::vector<std::size_t> dofs;
    LocalVector<T> local_g;
    LocalMatrix<T> local_H;

    for (std::size_t local_id = 0; local_id < energy_provider.local_count(); ++local_id) {
        energy_provider.local_dofs(local_id, dofs);
        detail::require_dofs_in_range(dofs, full_u);

        T local_value{};
        if constexpr (pgo::energy::FusedLocalEnergyProvider<EnergyProvider, T>) {
            energy_provider.local_value_gradient_hessian(local_id, full_u, local_value, local_g, local_H);
        } else {
            local_value = energy_provider.local_value(local_id, full_u);
            energy_provider.local_gradient(local_id, full_u, local_g);
            energy_provider.local_hessian(local_id, full_u, local_H);
        }
        detail::require_local_gradient_size(dofs, local_g);
        detail::require_local_hessian_size(dofs, local_H);
        value += local_value;

        for (std::size_t a = 0; a < dofs.size(); ++a) {
            full_gradient[pgo::math::dense_index(dofs[a])] += local_g[pgo::math::dense_index(a)];
            for (std::size_t b = 0; b < dofs.size(); ++b) {
                triplets.emplace_back(pgo::math::dense_index(dofs[a]), pgo::math::dense_index(dofs[b]),
                                      local_H(pgo::math::dense_index(a), pgo::math::dense_index(b)));
            }
        }
    }

    full_hessian.resize(full_u.size(), full_u.size());
    full_hessian.setFromTriplets(triplets.begin(), triplets.end());
}

} // namespace pgo::assembly
