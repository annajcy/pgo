#pragma once

#include "pgo/base/assert.hpp"

#include <cstddef>

namespace pgo::dof {

template <int Dim>
class DofLayout {
    std::size_t m_num_vertices{0};

public:
    static_assert(Dim > 0);

    explicit DofLayout(const std::size_t num_vertices) : m_num_vertices{num_vertices} {}

    [[nodiscard]] std::size_t num_vertices() const {
        return m_num_vertices;
    }

    [[nodiscard]] std::size_t num_dofs() const {
        return m_num_vertices * static_cast<std::size_t>(Dim);
    }

    [[nodiscard]] std::size_t index(const std::size_t vertex, const std::size_t component) const {
        pgo::base::require(vertex < m_num_vertices, "DOF vertex index is out of range");
        pgo::base::require(component < static_cast<std::size_t>(Dim), "DOF component index is out of range");
        return vertex * static_cast<std::size_t>(Dim) + component;
    }
};

} // namespace pgo::dof
