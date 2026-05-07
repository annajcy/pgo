#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/math/backend.hpp"

#include <cstddef>

namespace pgo::dof {

template <pgo::math::ScalarLike T, int Dim>
class Displacement {
public:
    explicit Displacement(const std::size_t num_vertices) : Displacement{DofLayout<Dim>{num_vertices}} {}

    explicit Displacement(DofLayout<Dim> layout)
        : m_layout{layout}, m_vector{static_cast<Eigen::Index>(m_layout.num_dofs())} {
        m_vector.setZero();
    }

    [[nodiscard]] const DofLayout<Dim>& layout() const {
        return m_layout;
    }

    [[nodiscard]] pgo::math::DVec<T>& vector() {
        return m_vector;
    }

    [[nodiscard]] const pgo::math::DVec<T>& vector() const {
        return m_vector;
    }

    [[nodiscard]] pgo::math::Vec<T, Dim> at(const std::size_t vertex) const {
        pgo::base::require(static_cast<std::size_t>(m_vector.size()) == m_layout.num_dofs(),
                           "displacement vector size does not match layout");

        pgo::math::Vec<T, Dim> value{};
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            value[static_cast<Eigen::Index>(component)] =
                m_vector[static_cast<Eigen::Index>(m_layout.index(vertex, component))];
        }
        return value;
    }

private:
    DofLayout<Dim> m_layout;
    pgo::math::DVec<T> m_vector;
};

} // namespace pgo::dof
