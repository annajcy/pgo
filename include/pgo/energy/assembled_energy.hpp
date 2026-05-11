#pragma once

#include "pgo/assembly/cpu_assembler.hpp"
#include "pgo/energy/energy_concepts.hpp"
#include "pgo/math/types.hpp"

namespace pgo::energy {

template <typename T, typename Provider>
    requires LocalEnergyProvider<Provider, T>
class AssembledEnergy {
    const Provider* m_provider;
    
public:
    explicit AssembledEnergy(const Provider& provider)
        : m_provider(&provider) {}

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const {
        return pgo::assembly::assemble_value<T>(*m_provider, full_u);
    }

    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const {
        pgo::assembly::assemble_gradient<T>(*m_provider, full_u, full_g);
    }

    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const {
        pgo::assembly::assemble_hessian<T>(*m_provider, full_u, full_H);
    }

    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& value, pgo::math::DVec<T>& full_g,
                                pgo::math::SparseMat<T>& full_H) const {
        pgo::assembly::assemble_value_gradient_hessian<T>(*m_provider, full_u, value, full_g, full_H);
    }
};

} // namespace pgo::energy
