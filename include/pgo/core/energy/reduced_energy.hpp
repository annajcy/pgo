#pragma once

#include "pgo/core/dof/reduced_dof_map.hpp"
#include "pgo/core/energy/energy_concepts.hpp"
#include "pgo/core/math/types.hpp"

namespace pgo::energy {

template <typename T, typename Energy>
    requires FullEnergy<Energy, T>
class ReducedEnergyView {
    const Energy* m_full_energy;
    const pgo::dof::ReducedDofMap<T>* m_dof_map;

public:
    ReducedEnergyView(const Energy& full_energy,
                      const pgo::dof::ReducedDofMap<T>& dof_map)
        : m_full_energy(&full_energy), m_dof_map(&dof_map) {}

    [[nodiscard]] std::size_t full_dofs() const {
        return m_dof_map->full_dofs();
    }

    [[nodiscard]] std::size_t free_dofs() const {
        return m_dof_map->free_dofs();
    }

    [[nodiscard]] T value(const pgo::math::DVec<T>& free_u) const {
        const auto full_u = m_dof_map->scatter_solution(free_u);
        return m_full_energy->value(full_u);
    }

    void gradient(const pgo::math::DVec<T>& free_u, pgo::math::DVec<T>& reduced_g) const {
        const auto full_u = m_dof_map->scatter_solution(free_u);
        pgo::math::DVec<T> full_g;
        m_full_energy->gradient(full_u, full_g);
        reduced_g = m_dof_map->restrict_vector_to_free(full_g);
    }

    void hessian(const pgo::math::DVec<T>& free_u, pgo::math::SparseMat<T>& reduced_H) const {
        const auto full_u = m_dof_map->scatter_solution(free_u);
        pgo::math::SparseMat<T> full_H;
        m_full_energy->hessian(full_u, full_H);
        reduced_H = m_dof_map->restrict_matrix_to_free(full_H);
    }

    void value_gradient_hessian(const pgo::math::DVec<T>& free_u, T& value,
                                pgo::math::DVec<T>& reduced_g,
                                pgo::math::SparseMat<T>& reduced_H) const {
        const auto full_u = m_dof_map->scatter_solution(free_u);
        pgo::math::DVec<T> full_g;
        pgo::math::SparseMat<T> full_H;
        m_full_energy->value_gradient_hessian(full_u, value, full_g, full_H);
        reduced_g = m_dof_map->restrict_vector_to_free(full_g);
        reduced_H = m_dof_map->restrict_matrix_to_free(full_H);
    }

    [[nodiscard]] pgo::math::DVec<T> scatter_solution(const pgo::math::DVec<T>& free_u) const {
        return m_dof_map->scatter_solution(free_u);
    }

    [[nodiscard]] pgo::math::DVec<T> scatter_direction(const pgo::math::DVec<T>& free_du) const {
        return m_dof_map->scatter_direction(free_du);
    }
};

} // namespace pgo::energy
