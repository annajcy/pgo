#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/math/backend.hpp"
#include "pgo/storage/array_view.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace pgo::dof {

template <pgo::math::RealScalar T>
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
            if (boundary.is_fixed(full_dof)) {
                m_fixed_values[pgo::math::dense_index(full_dof)] = boundary.value(full_dof);
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
        return fixed_value(full_dof);
    }

    [[nodiscard]] T fixed_value(const std::size_t full_dof) const {
        pgo::base::require(full_dof < full_dofs(), "full DOF index is out of range");
        pgo::base::require(!is_free(full_dof), "full DOF is not fixed");
        return m_fixed_values[pgo::math::dense_index(full_dof)];
    }

    [[nodiscard]] pgo::math::DVec<T> unpack_solution(const pgo::math::DVec<T>& free_u) const {
        require_free_vector_size(free_u);

        pgo::math::DVec<T> full_u{pgo::math::dense_index(full_dofs())};
        for (std::size_t full_dof = 0; full_dof < full_dofs(); ++full_dof) {
            if (m_full_to_free[full_dof] == npos) {
                full_u[pgo::math::dense_index(full_dof)] = m_fixed_values[pgo::math::dense_index(full_dof)];
            } else {
                full_u[pgo::math::dense_index(full_dof)] = free_u[pgo::math::dense_index(m_full_to_free[full_dof])];
            }
        }
        return full_u;
    }

    [[nodiscard]] pgo::math::DVec<T> pack_rhs(const pgo::math::DVec<T>& full_f,
                                              const pgo::math::SparseMat<T>& full_mat) const {
        require_full_vector_size(full_f);
        require_full_matrix_size(full_mat);

        // Extract f_free
        pgo::math::DVec<T> reduced_f{pgo::math::dense_index(free_dofs())};
        for (std::size_t free_dof = 0; free_dof < m_free_to_full.size(); ++free_dof) {
            reduced_f[pgo::math::dense_index(free_dof)] = full_f[pgo::math::dense_index(m_free_to_full[free_dof])];
        }

        // Subtract K_fc * g: for each entry where row is free and col is fixed
        for (pgo::math::DenseIndex k = 0; k < full_mat.outerSize(); ++k) {
            for (typename pgo::math::SparseMat<T>::InnerIterator it(full_mat, k); it; ++it) {
                const auto row = static_cast<std::size_t>(it.row());
                const auto col = static_cast<std::size_t>(it.col());

                if (m_full_to_free[row] != npos && m_full_to_free[col] == npos) {
                    reduced_f[pgo::math::dense_index(m_full_to_free[row])] -=
                        it.value() * m_fixed_values[pgo::math::dense_index(col)];
                }
            }
        }

        return reduced_f;
    }

    [[nodiscard]] pgo::math::SparseMat<T> pack_matrix(const pgo::math::SparseMat<T>& full_mat) const {
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

                triplets.emplace_back(
                    pgo::math::dense_index(m_full_to_free[full_row]),
                    pgo::math::dense_index(m_full_to_free[full_col]),
                    it.value());
            }
        }

        const auto n = pgo::math::dense_index(free_dofs());
        pgo::math::SparseMat<T> free_mat(n, n);
        free_mat.setFromTriplets(triplets.begin(), triplets.end());
        return free_mat;
    }

    [[nodiscard]] pgo::math::SparseMat<T> unpack_matrix(const pgo::math::SparseMat<T>& free_mat) const {
        require_free_matrix_size(free_mat);

        const auto num_fixed = full_dofs() - free_dofs();
        std::vector<pgo::math::Triplet<T>> triplets;
        triplets.reserve(static_cast<std::size_t>(free_mat.nonZeros()) + num_fixed);

        // Map free×free entries back to full positions
        for (pgo::math::DenseIndex k = 0; k < free_mat.outerSize(); ++k) {
            for (typename pgo::math::SparseMat<T>::InnerIterator it(free_mat, k); it; ++it) {
                const auto free_row = static_cast<std::size_t>(it.row());
                const auto free_col = static_cast<std::size_t>(it.col());

                triplets.emplace_back(
                    pgo::math::dense_index(m_free_to_full[free_row]),
                    pgo::math::dense_index(m_free_to_full[free_col]),
                    it.value());
            }
        }

        // Fixed DOFs get identity on the diagonal (elimination convention)
        for (std::size_t full_dof = 0; full_dof < full_dofs(); ++full_dof) {
            if (m_full_to_free[full_dof] == npos) {
                triplets.emplace_back(
                    pgo::math::dense_index(full_dof),
                    pgo::math::dense_index(full_dof),
                    T{1});
            }
        }

        const auto n = pgo::math::dense_index(full_dofs());
        pgo::math::SparseMat<T> full_mat(n, n);
        full_mat.setFromTriplets(triplets.begin(), triplets.end());
        return full_mat;
    }

    template <DirichletBoundaryLike<T> Boundary>
    void validate_boundary_snapshot(const Boundary& boundary) const {
        for (std::size_t full_dof = 0; full_dof < full_dofs(); ++full_dof) {
            const bool map_fixed = m_full_to_free[full_dof] == npos;
            const bool boundary_fixed = boundary.is_fixed(full_dof);
            pgo::base::require(map_fixed == boundary_fixed, "boundary fixed/free state does not match DOF map");

            if (map_fixed) {
                pgo::base::require(m_fixed_values[pgo::math::dense_index(full_dof)] == boundary.value(full_dof),
                                   "boundary fixed value does not match DOF map snapshot");
            }
        }
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
        pgo::base::require(static_cast<std::size_t>(matrix.rows()) == full_dofs()
                               && static_cast<std::size_t>(matrix.cols()) == full_dofs(),
                           "full matrix size does not match DOF map");
    }

    void require_free_matrix_size(const pgo::math::SparseMat<T>& matrix) const {
        pgo::base::require(static_cast<std::size_t>(matrix.rows()) == free_dofs()
                               && static_cast<std::size_t>(matrix.cols()) == free_dofs(),
                           "free matrix size does not match DOF map");
    }
};

} // namespace pgo::dof
