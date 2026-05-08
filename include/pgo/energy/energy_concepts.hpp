#pragma once

#include "pgo/assembly/local_matrix.hpp"
#include "pgo/math/types.hpp"

#include <concepts>
#include <cstddef>
#include <vector>

namespace pgo::energy {

template <typename Energy, typename T>
concept FullEnergy =
    requires(const Energy& energy, const pgo::math::DVec<T>& u, T& value,
                                         pgo::math::DVec<T>& gradient, pgo::math::SparseMat<T>& hessian) {
        { energy.value(u) } -> std::same_as<T>;
        energy.gradient(u, gradient);
        energy.hessian(u, hessian);
        energy.value_gradient_hessian(u, value, gradient, hessian);
    };

template <typename Model, typename T, typename LocalData>
concept LocalEnergyModel =
                           requires(const LocalData& local_data, const pgo::assembly::LocalVector<T>& local_u,
                                    pgo::assembly::LocalVector<T>& local_g, pgo::assembly::LocalMatrix<T>& local_H) {
                               { Model::local_dof_count(local_data) } -> std::convertible_to<std::size_t>;
                               { Model::value(local_data, local_u) } -> std::same_as<T>;
                               Model::gradient(local_data, local_u, local_g);
                               Model::hessian(local_data, local_u, local_H);
                           };

template <typename Model, typename T, typename LocalData>
concept FusedLocalEnergyModel =
    LocalEnergyModel<Model, T, LocalData> &&
    requires(const LocalData& local_data, const pgo::assembly::LocalVector<T>& local_u, T& value,
             pgo::assembly::LocalVector<T>& local_g, pgo::assembly::LocalMatrix<T>& local_H) {
        Model::value_gradient_hessian(local_data, local_u, value, local_g, local_H);
    };

template <typename EnergyProvider, typename T>
concept LocalEnergyProvider =
                              requires(const EnergyProvider& energy_provider, std::size_t local_id,
                                       const pgo::math::DVec<T>& full_u, std::vector<std::size_t>& dofs,
                                       pgo::assembly::LocalVector<T>& local_g, pgo::assembly::LocalMatrix<T>& local_H) {
                                  { energy_provider.local_count() } -> std::convertible_to<std::size_t>;
                                  energy_provider.local_dofs(local_id, dofs);
                                  { energy_provider.local_value(local_id, full_u) } -> std::same_as<T>;
                                  energy_provider.local_gradient(local_id, full_u, local_g);
                                  energy_provider.local_hessian(local_id, full_u, local_H);
                              };

template <typename EnergyProvider, typename T>
concept FusedLocalEnergyProvider =
    LocalEnergyProvider<EnergyProvider, T> &&
    requires(const EnergyProvider& energy_provider, std::size_t local_id, const pgo::math::DVec<T>& full_u, T& value,
             pgo::assembly::LocalVector<T>& local_g, pgo::assembly::LocalMatrix<T>& local_H) {
        energy_provider.local_value_gradient_hessian(local_id, full_u, value, local_g, local_H);
    };

} // namespace pgo::energy
