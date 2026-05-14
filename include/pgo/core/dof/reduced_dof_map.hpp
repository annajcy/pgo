#pragma once

#include "pgo/core/base/assert.hpp"
#include "pgo/core/dof/dirichlet_boundary.hpp"
#include "pgo/core/math/types.hpp"
#include "pgo/core/storage/array_view.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace pgo::dof {

template <typename T>
class ReducedDofMap {
    std::vector<std::size_t> m_free_to_full;
    std::vector<std::size_t> m_full_to_free;
    pgo::math::DVec<T> m_fixed_values;

public:
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    template <DirichletBoundaryLike<T> Boundary>
    ReducedDofMap(const std::size_t full_dofs, const Boundary& boundary)
        : m_full_to_free(full_dofs, npos), m_fixed_values{pgo::math::dense_index(full_dofs)} {
        m_free_to_full.reserve(full_dofs);
        m_fixed_values.setZero();
        for (std::size_t full_dof = 0; full_dof < full_dofs; ++full_dof) {
            if (const auto fixed = boundary.fixed_value(full_dof)) {
                m_fixed_values[pgo::math::dense_index(full_dof)] = *fixed;
            } else {
                m_full_to_free[full_dof] = m_free_to_full.size();
                m_free_to_full.push_back(full_dof);
            }
        }
    }

    [[nodiscard]] std::size_t full_dofs() const {
        return m_full_to_free.size();
    }

    [[nodiscard]] std::size_t free_dofs() const {
        return m_free_to_full.size();
    }

    [[nodiscard]] pgo::storage::ConstArrayView<std::size_t> free_to_full() const {
        return m_free_to_full;
    }

    [[nodiscard]] pgo::storage::ConstArrayView<std::size_t> full_to_free() const {
        return m_full_to_free;
    }

    [[nodiscard]] std::size_t full_dof(const std::size_t free_dof) const {
        pgo::base::require(free_dof < free_dofs(), "free DOF index is out of range");
        return m_free_to_full[free_dof];
    }

    [[nodiscard]] std::size_t free_dof(const std::size_t full_dof) const {
        pgo::base::require(full_dof < full_dofs(), "full DOF index is out of range");
        pgo::base::require(is_free(full_dof), "full DOF is fixed");
        return m_full_to_free[full_dof];
    }

    [[nodiscard]] bool is_free(const std::size_t full_dof) const {
        pgo::base::require(full_dof < full_dofs(), "full DOF index is out of range");
        return m_full_to_free[full_dof] != npos;
    }

    [[nodiscard]] bool is_fixed(const std::size_t full_dof) const {
        return !is_free(full_dof);
    }

    [[nodiscard]] T value(const std::size_t full_dof) const {
        const auto fixed = fixed_value(full_dof);
        pgo::base::require(fixed.has_value(), "full DOF is not fixed");
        return *fixed;
    }

    [[nodiscard]] std::optional<T> fixed_value(const std::size_t full_dof) const {
        pgo::base::require(full_dof < full_dofs(), "full DOF index is out of range");
        if (is_free(full_dof)) {
            return std::nullopt;
        }
        return m_fixed_values[pgo::math::dense_index(full_dof)];
    }

    [[nodiscard]] pgo::math::DVec<T> scatter_solution(const pgo::math::DVec<T>& free_solution) const {
        require_free_vector_size(free_solution);

        pgo::math::DVec<T> full_solution{pgo::math::dense_index(full_dofs())};
        for (std::size_t full_dof = 0; full_dof < full_dofs(); ++full_dof) {
            if (m_full_to_free[full_dof] == npos) {
                full_solution[pgo::math::dense_index(full_dof)] = m_fixed_values[pgo::math::dense_index(full_dof)];
            } else {
                full_solution[pgo::math::dense_index(full_dof)] =
                    free_solution[pgo::math::dense_index(m_full_to_free[full_dof])];
            }
        }
        return full_solution;
    }

    [[nodiscard]] pgo::math::DVec<T> scatter_direction(const pgo::math::DVec<T>& free_direction) const {
        require_free_vector_size(free_direction);

        pgo::math::DVec<T> full_direction{pgo::math::dense_index(full_dofs())};
        full_direction.setZero();
        for (std::size_t free_dof = 0; free_dof < m_free_to_full.size(); ++free_dof) {
            full_direction[pgo::math::dense_index(m_free_to_full[free_dof])] =
                free_direction[pgo::math::dense_index(free_dof)];
        }
        return full_direction;
    }

    [[nodiscard]] pgo::math::DVec<T> restrict_vector_to_free(const pgo::math::DVec<T>& full_vector) const {
        require_full_vector_size(full_vector);

        pgo::math::DVec<T> free_vector{pgo::math::dense_index(free_dofs())};
        for (std::size_t free_dof = 0; free_dof < m_free_to_full.size(); ++free_dof) {
            free_vector[pgo::math::dense_index(free_dof)] =
                full_vector[pgo::math::dense_index(m_free_to_full[free_dof])];
        }

        return free_vector;
    }

    [[nodiscard]] pgo::math::SparseMat<T> restrict_matrix_to_free(const pgo::math::SparseMat<T>& full_mat) const {
        require_full_matrix_size(full_mat);

        std::vector<pgo::math::Triplet<T>> triplets;
        triplets.reserve(static_cast<std::size_t>(full_mat.nonZeros()));

        for (pgo::math::DenseIndex k = 0; k < full_mat.outerSize(); ++k) {
            for (typename pgo::math::SparseMat<T>::InnerIterator it(full_mat, k); it; ++it) {
                const auto full_row = static_cast<std::size_t>(it.row());
                const auto full_col = static_cast<std::size_t>(it.col());

                if (m_full_to_free[full_row] == npos || m_full_to_free[full_col] == npos) {
                    continue;
                }

                triplets.emplace_back(pgo::math::dense_index(m_full_to_free[full_row]),
                                      pgo::math::dense_index(m_full_to_free[full_col]), it.value());
            }
        }

        const auto n = pgo::math::dense_index(free_dofs());
        pgo::math::SparseMat<T> free_mat(n, n);
        free_mat.setFromTriplets(triplets.begin(), triplets.end());
        return free_mat;
    }

    [[nodiscard]] pgo::math::DVec<T> eliminate_rhs_for_dirichlet(const pgo::math::DVec<T>& full_rhs,
                                                                 const pgo::math::SparseMat<T>& full_matrix) const {
        require_full_vector_size(full_rhs);
        require_full_matrix_size(full_matrix);

        pgo::math::DVec<T> reduced_rhs = restrict_vector_to_free(full_rhs);

        for (pgo::math::DenseIndex k = 0; k < full_matrix.outerSize(); ++k) {
            for (typename pgo::math::SparseMat<T>::InnerIterator it(full_matrix, k); it; ++it) {
                const auto row = static_cast<std::size_t>(it.row());
                const auto col = static_cast<std::size_t>(it.col());

                if (m_full_to_free[row] != npos && m_full_to_free[col] == npos) {
                    reduced_rhs[pgo::math::dense_index(m_full_to_free[row])] -=
                        it.value() * m_fixed_values[pgo::math::dense_index(col)];
                }
            }
        }

        return reduced_rhs;
    }

private:
    void require_full_vector_size(const pgo::math::DVec<T>& vector) const {
        pgo::base::require(static_cast<std::size_t>(vector.size()) == full_dofs(),
                           "full vector size does not match DOF map");
    }

    void require_free_vector_size(const pgo::math::DVec<T>& vector) const {
        pgo::base::require(static_cast<std::size_t>(vector.size()) == free_dofs(),
                           "free vector size does not match DOF map");
    }

    void require_full_matrix_size(const pgo::math::SparseMat<T>& matrix) const {
        pgo::base::require(static_cast<std::size_t>(matrix.rows()) == full_dofs() &&
                               static_cast<std::size_t>(matrix.cols()) == full_dofs(),
                           "full matrix size does not match DOF map");
    }
};

} // namespace pgo::dof
