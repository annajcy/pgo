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
public:
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    ReducedDofMap(const std::size_t full_dofs, const DirichletBoundary<T>& boundary) : m_full_to_free(full_dofs, npos) {
        m_free_to_full.reserve(full_dofs);
        for (std::size_t full_dof = 0; full_dof < full_dofs; ++full_dof) {
            if (!boundary.is_fixed(full_dof)) {
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

    [[nodiscard]] bool is_free(const std::size_t full_dof) const {
        pgo::base::require(full_dof < full_dofs(), "full DOF index is out of range");
        return m_full_to_free[full_dof] != npos;
    }

    [[nodiscard]] pgo::math::DVec<T> pack_displacement(const pgo::math::DVec<T>& full_u) const {
        require_full_vector_size(full_u);

        pgo::math::DVec<T> free_u{static_cast<Eigen::Index>(free_dofs())};
        for (std::size_t free_dof = 0; free_dof < m_free_to_full.size(); ++free_dof) {
            free_u[static_cast<Eigen::Index>(free_dof)] = full_u[static_cast<Eigen::Index>(m_free_to_full[free_dof])];
        }
        return free_u;
    }

    [[nodiscard]] pgo::math::DVec<T> unpack_displacement(const pgo::math::DVec<T>& free_u,
                                                         const DirichletBoundary<T>& boundary) const {
        require_free_vector_size(free_u);

        pgo::math::DVec<T> full_u{static_cast<Eigen::Index>(full_dofs())};
        for (std::size_t full_dof = 0; full_dof < full_dofs(); ++full_dof) {
            if (m_full_to_free[full_dof] == npos) {
                full_u[static_cast<Eigen::Index>(full_dof)] = boundary.value(full_dof);
            } else {
                full_u[static_cast<Eigen::Index>(full_dof)] =
                    free_u[static_cast<Eigen::Index>(m_full_to_free[full_dof])];
            }
        }
        return full_u;
    }

    [[nodiscard]] pgo::math::DVec<T> reduce_vector(const pgo::math::DVec<T>& full_v) const {
        require_full_vector_size(full_v);

        pgo::math::DVec<T> reduced_v{static_cast<Eigen::Index>(free_dofs())};
        for (std::size_t free_dof = 0; free_dof < m_free_to_full.size(); ++free_dof) {
            reduced_v[static_cast<Eigen::Index>(free_dof)] =
                full_v[static_cast<Eigen::Index>(m_free_to_full[free_dof])];
        }
        return reduced_v;
    }

    [[nodiscard]] pgo::math::SparseMat<T> reduce_sparse_mat(const pgo::math::SparseMat<T>& full_a) const {
        pgo::base::require(static_cast<std::size_t>(full_a.rows()) == full_dofs(),
                           "full matrix row count does not match DOF map");
        pgo::base::require(static_cast<std::size_t>(full_a.cols()) == full_dofs(),
                           "full matrix column count does not match DOF map");

        std::vector<pgo::math::Triplet<T>> triplets;
        triplets.reserve(static_cast<std::size_t>(full_a.nonZeros()));

        for (int outer = 0; outer < full_a.outerSize(); ++outer) {
            for (typename pgo::math::SparseMat<T>::InnerIterator entry{full_a, outer}; entry; ++entry) {
                const auto row = static_cast<std::size_t>(entry.row());
                const auto col = static_cast<std::size_t>(entry.col());
                if (m_full_to_free[row] != npos && m_full_to_free[col] != npos) {
                    triplets.emplace_back(static_cast<int>(m_full_to_free[row]), static_cast<int>(m_full_to_free[col]),
                                          entry.value());
                }
            }
        }

        pgo::math::SparseMat<T> reduced_a{static_cast<Eigen::Index>(free_dofs()),
                                          static_cast<Eigen::Index>(free_dofs())};
        reduced_a.setFromTriplets(triplets.begin(), triplets.end());
        return reduced_a;
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

    std::vector<std::size_t> m_free_to_full;
    std::vector<std::size_t> m_full_to_free;
};

} // namespace pgo::dof
