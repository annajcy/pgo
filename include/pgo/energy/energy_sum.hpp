#pragma once

#include "pgo/energy/energy_concepts.hpp"
#include "pgo/math/types.hpp"

#include <tuple>
#include <utility>

namespace pgo::energy {

template <typename T, typename... Energies>
    requires (FullEnergy<Energies, T> && ...)
class EnergySumView {
    std::tuple<const Energies*...> m_energies;

public:
    explicit EnergySumView(const Energies&... energies)
        : m_energies(&energies...) {}

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const {
        return value_impl(full_u, std::index_sequence_for<Energies...>{});
    }

    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const {
        gradient_impl(full_u, full_g, std::index_sequence_for<Energies...>{});
    }

    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const {
        hessian_impl(full_u, full_H, std::index_sequence_for<Energies...>{});
    }

    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& value,
                                pgo::math::DVec<T>& full_g,
                                pgo::math::SparseMat<T>& full_H) const {
        value_gradient_hessian_impl(full_u, value, full_g, full_H,
                                    std::index_sequence_for<Energies...>{});
    }

private:
    template <std::size_t... Is>
    [[nodiscard]] T value_impl(const pgo::math::DVec<T>& full_u, std::index_sequence<Is...>) const {
        return (std::get<Is>(m_energies)->value(full_u) + ...);
    }

    template <std::size_t... Is>
    void gradient_impl(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g,
                       std::index_sequence<Is...>) const {
        full_g.resize(full_u.size());
        full_g.setZero();
        pgo::math::DVec<T> sub_g;
        ((std::get<Is>(m_energies)->gradient(full_u, sub_g), full_g += sub_g), ...);
    }

    template <std::size_t... Is>
    void hessian_impl(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H,
                      std::index_sequence<Is...>) const {
        bool first = true;
        pgo::math::SparseMat<T> sub_H;
        auto accumulate = [&](const auto* energy) {
            energy->hessian(full_u, sub_H);
            if (first) {
                full_H = sub_H;
                first = false;
            } else {
                full_H += sub_H;
            }
        };
        (accumulate(std::get<Is>(m_energies)), ...);
    }

    template <std::size_t... Is>
    void value_gradient_hessian_impl(const pgo::math::DVec<T>& full_u, T& value,
                                     pgo::math::DVec<T>& full_g,
                                     pgo::math::SparseMat<T>& full_H,
                                     std::index_sequence<Is...>) const {
        value = T{0};
        full_g.resize(full_u.size());
        full_g.setZero();
        bool first = true;

        T sub_value{};
        pgo::math::DVec<T> sub_g;
        pgo::math::SparseMat<T> sub_H;

        auto accumulate = [&](const auto* energy) {
            energy->value_gradient_hessian(full_u, sub_value, sub_g, sub_H);
            value += sub_value;
            full_g += sub_g;
            if (first) {
                full_H = sub_H;
                first = false;
            } else {
                full_H += sub_H;
            }
        };
        (accumulate(std::get<Is>(m_energies)), ...);
    }
};

} // namespace pgo::energy
