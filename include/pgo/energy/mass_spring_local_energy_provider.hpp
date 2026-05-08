#pragma once

#include "pgo/assembly/local_matrix.hpp"
#include "pgo/base/assert.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/math/types.hpp"
#include "pgo/storage/array_view.hpp"
#include "pgo/storage/host_buffer.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace pgo::energy {

template <typename T, int Dim>
struct MassSpringLocalData {
    pgo::math::Vec<T, Dim> rest_i{};
    pgo::math::Vec<T, Dim> rest_j{};
    T stiffness{};
};

template <typename T, int Dim>
class MassSpringLocalEnergyModel {
public:
    [[nodiscard]] static std::size_t local_dof_count(const MassSpringLocalData<T, Dim>& local_data) {
        validate_data(local_data);
        return 2 * static_cast<std::size_t>(Dim);
    }

    [[nodiscard]] static T value(const MassSpringLocalData<T, Dim>& local_data,
                                 const pgo::assembly::LocalVector<T>& local_u) {
        validate_data(local_data);
        require_local_u_size(local_u);

        const auto rest_d = local_data.rest_i - local_data.rest_j;
        const auto current_d = current_edge_vector(local_data, local_u);
        const auto rest_length = rest_d.norm();
        const auto current_length = current_d.norm();
        const auto safe_length = current_length > kMinLength ? current_length : kMinLength;
        const auto stretch = safe_length - rest_length;

        return T{0.5} * local_data.stiffness * stretch * stretch;
    }

    static void gradient(const MassSpringLocalData<T, Dim>& local_data, const pgo::assembly::LocalVector<T>& local_u,
                         pgo::assembly::LocalVector<T>& local_g) {
        validate_data(local_data);
        require_local_u_size(local_u);

        const auto rest_d = local_data.rest_i - local_data.rest_j;
        const auto current_d = current_edge_vector(local_data, local_u);
        const auto rest_length = rest_d.norm();
        const auto current_length = current_d.norm();
        const auto safe_length = current_length > kMinLength ? current_length : kMinLength;
        const auto stretch = safe_length - rest_length;

        pgo::math::Vec<T, Dim> direction{};
        if (current_length > kMinLength) {
            direction = current_d / current_length;
        } else {
            direction.setZero();
        }

        const pgo::math::Vec<T, Dim> grad_d = local_data.stiffness * stretch * direction;

        local_g.resize(pgo::math::dense_index(local_dof_count(local_data)));
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            local_g[pgo::math::dense_index(component)] = grad_d[pgo::math::dense_index(component)];
            local_g[pgo::math::dense_index(static_cast<std::size_t>(Dim) + component)] =
                -grad_d[pgo::math::dense_index(component)];
        }
    }

    static void hessian(const MassSpringLocalData<T, Dim>& local_data, const pgo::assembly::LocalVector<T>& local_u,
                        pgo::assembly::LocalMatrix<T>& local_H) {
        T local_value{};
        pgo::assembly::LocalVector<T> local_g;
        value_gradient_hessian(local_data, local_u, local_value, local_g, local_H);
    }

    static void value_gradient_hessian(const MassSpringLocalData<T, Dim>& local_data,
                                       const pgo::assembly::LocalVector<T>& local_u, T& value,
                                       pgo::assembly::LocalVector<T>& local_g, pgo::assembly::LocalMatrix<T>& local_H) {
        validate_data(local_data);
        require_local_u_size(local_u);

        const auto rest_d = local_data.rest_i - local_data.rest_j;
        const auto current_d = current_edge_vector(local_data, local_u);
        const auto rest_length = rest_d.norm();
        const auto current_length = current_d.norm();
        const auto safe_length = current_length > kMinLength ? current_length : kMinLength;
        const auto stretch = safe_length - rest_length;

        value = T{0.5} * local_data.stiffness * stretch * stretch;

        pgo::math::Vec<T, Dim> direction{};
        if (current_length > kMinLength) {
            direction = current_d / current_length;
        } else {
            direction.setZero();
        }

        const pgo::math::Vec<T, Dim> grad_d = local_data.stiffness * stretch * direction;

        pgo::math::Mat<T, Dim, Dim> H_d{};
        if (current_length > kMinLength) {
            const pgo::math::Mat<T, Dim, Dim> nnT = direction * direction.transpose();
            const pgo::math::Mat<T, Dim, Dim> I = pgo::math::Mat<T, Dim, Dim>::Identity();
            H_d = local_data.stiffness * (nnT + (T{1} - rest_length / current_length) * (I - nnT));
        } else {
            H_d = local_data.stiffness * pgo::math::Mat<T, Dim, Dim>::Identity();
        }

        local_g.resize(pgo::math::dense_index(local_dof_count(local_data)));
        local_H.resize(pgo::math::dense_index(local_dof_count(local_data)),
                       pgo::math::dense_index(local_dof_count(local_data)));
        local_H.setZero();

        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            local_g[pgo::math::dense_index(component)] = grad_d[pgo::math::dense_index(component)];
            local_g[pgo::math::dense_index(static_cast<std::size_t>(Dim) + component)] =
                -grad_d[pgo::math::dense_index(component)];
        }

        for (std::size_t row = 0; row < static_cast<std::size_t>(Dim); ++row) {
            for (std::size_t col = 0; col < static_cast<std::size_t>(Dim); ++col) {
                const auto value_H = H_d(pgo::math::dense_index(row), pgo::math::dense_index(col));
                local_H(pgo::math::dense_index(row), pgo::math::dense_index(col)) = value_H;
                local_H(pgo::math::dense_index(row), pgo::math::dense_index(static_cast<std::size_t>(Dim) + col)) =
                    -value_H;
                local_H(pgo::math::dense_index(static_cast<std::size_t>(Dim) + row), pgo::math::dense_index(col)) =
                    -value_H;
                local_H(pgo::math::dense_index(static_cast<std::size_t>(Dim) + row),
                        pgo::math::dense_index(static_cast<std::size_t>(Dim) + col)) = value_H;
            }
        }
    }

private:
    // Numerical safety threshold to prevent division by zero when current edge length collapses.
    // sqrt(epsilon) gives ~1.49e-8 for double, ~3.45e-4 for float.
    static constexpr T kMinLength = std::is_same_v<T, float> ? T{3.4526698e-4} : T{1.4901161193847656e-8};

    static void validate_data(const MassSpringLocalData<T, Dim>& local_data) {
        static_assert(Dim > 0);
        pgo::base::require(local_data.stiffness >= T{0}, "mass-spring stiffness must be non-negative");
        pgo::base::require((local_data.rest_i - local_data.rest_j).norm() > kMinLength,
                           "mass-spring rest edge length is too small");
    }

    static void require_local_u_size(const pgo::assembly::LocalVector<T>& local_u) {
        pgo::base::require(static_cast<std::size_t>(local_u.size()) == 2 * static_cast<std::size_t>(Dim),
                           "mass-spring local displacement size does not match local DOF count");
    }

    [[nodiscard]] static pgo::math::Vec<T, Dim> current_edge_vector(const MassSpringLocalData<T, Dim>& local_data,
                                                                    const pgo::assembly::LocalVector<T>& local_u) {
        pgo::math::Vec<T, Dim> x_i = local_data.rest_i;
        pgo::math::Vec<T, Dim> x_j = local_data.rest_j;
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            x_i[pgo::math::dense_index(component)] += local_u[pgo::math::dense_index(component)];
            x_j[pgo::math::dense_index(component)] +=
                local_u[pgo::math::dense_index(static_cast<std::size_t>(Dim) + component)];
        }

        return x_i - x_j;
    }
};

template <typename T, int Dim>
class MassSpringLocalEnergyProvider {
    using Model = MassSpringLocalEnergyModel<T, Dim>;

    const pgo::geometry::RestMesh<T, Dim>* m_mesh;
    pgo::storage::HostBuffer<T> m_stiffnesses;

public:
    explicit MassSpringLocalEnergyProvider(const pgo::geometry::RestMesh<T, Dim>& mesh, const T stiffness)
        : m_mesh{&mesh}, m_stiffnesses(mesh.num_edges(), stiffness) {
        pgo::base::require(stiffness >= T{0}, "mass-spring stiffness must be non-negative");
        validate_all_edges();
    }

    explicit MassSpringLocalEnergyProvider(const pgo::geometry::RestMesh<T, Dim>& mesh,
                                           const pgo::storage::ConstArrayView<T> stiffnesses)
        : m_mesh{&mesh}, m_stiffnesses(stiffnesses.begin(), stiffnesses.end()) {
        pgo::base::require(stiffnesses.size() == mesh.num_edges(), "mass-spring stiffness count must match edge count");
        validate_all_edges();
    }

    [[nodiscard]] std::size_t local_count() const {
        return m_mesh->num_edges();
    }

    [[nodiscard]] std::size_t max_local_dofs() const {
        return 2 * static_cast<std::size_t>(Dim);
    }

    void local_dofs(const std::size_t edge_id, std::vector<std::size_t>& dofs) const {
        pgo::base::require(edge_id < local_count(), "spring edge index is out of range");

        dofs.resize(max_local_dofs());
        const auto vertex_i = static_cast<std::size_t>(m_mesh->edge_vertex(edge_id, 0));
        const auto vertex_j = static_cast<std::size_t>(m_mesh->edge_vertex(edge_id, 1));
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            dofs[component] = vertex_i * static_cast<std::size_t>(Dim) + component;
            dofs[static_cast<std::size_t>(Dim) + component] = vertex_j * static_cast<std::size_t>(Dim) + component;
        }
    }

    [[nodiscard]] T local_value(const std::size_t edge_id, const pgo::math::DVec<T>& full_u) const {
        return Model::value(local_data(edge_id), gather_local_u(edge_id, full_u));
    }

    void local_gradient(const std::size_t edge_id, const pgo::math::DVec<T>& full_u,
                        pgo::assembly::LocalVector<T>& local_g) const {
        Model::gradient(local_data(edge_id), gather_local_u(edge_id, full_u), local_g);
    }

    void local_hessian(const std::size_t edge_id, const pgo::math::DVec<T>& full_u,
                       pgo::assembly::LocalMatrix<T>& local_H) const {
        Model::hessian(local_data(edge_id), gather_local_u(edge_id, full_u), local_H);
    }

    void local_value_gradient_hessian(const std::size_t edge_id, const pgo::math::DVec<T>& full_u, T& value,
                                      pgo::assembly::LocalVector<T>& local_g,
                                      pgo::assembly::LocalMatrix<T>& local_H) const {
        Model::value_gradient_hessian(local_data(edge_id), gather_local_u(edge_id, full_u), value, local_g, local_H);
    }

private:
    void require_edge_id_in_range(const std::size_t edge_id) const {
        pgo::base::require(edge_id < local_count(), "spring edge index is out of range");
    }

    void require_full_u_size(const pgo::math::DVec<T>& full_u) const {
        pgo::base::require(static_cast<std::size_t>(full_u.size()) ==
                               m_mesh->num_vertices() * static_cast<std::size_t>(Dim),
                           "displacement vector size does not match mass-spring mesh");
    }

    void validate_all_edges() const {
        static_assert(Dim > 0);
        for (std::size_t edge_id = 0; edge_id < m_mesh->num_edges(); ++edge_id) {
            static_cast<void>(Model::local_dof_count(local_data(edge_id)));
        }
    }

    [[nodiscard]] MassSpringLocalData<T, Dim> local_data(const std::size_t edge_id) const {
        require_edge_id_in_range(edge_id);
        const auto vertex_i = static_cast<std::size_t>(m_mesh->edge_vertex(edge_id, 0));
        const auto vertex_j = static_cast<std::size_t>(m_mesh->edge_vertex(edge_id, 1));
        return MassSpringLocalData<T, Dim>{
            .rest_i = m_mesh->rest_position(vertex_i),
            .rest_j = m_mesh->rest_position(vertex_j),
            .stiffness = m_stiffnesses[edge_id],
        };
    }

    [[nodiscard]] pgo::assembly::LocalVector<T> gather_local_u(const std::size_t edge_id,
                                                               const pgo::math::DVec<T>& full_u) const {
        require_edge_id_in_range(edge_id);
        require_full_u_size(full_u);

        const auto vertex_i = static_cast<std::size_t>(m_mesh->edge_vertex(edge_id, 0));
        const auto vertex_j = static_cast<std::size_t>(m_mesh->edge_vertex(edge_id, 1));

        pgo::assembly::LocalVector<T> local_u{pgo::math::dense_index(max_local_dofs())};
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            local_u[pgo::math::dense_index(component)] =
                full_u[pgo::math::dense_index(vertex_i * static_cast<std::size_t>(Dim) + component)];
            local_u[pgo::math::dense_index(static_cast<std::size_t>(Dim) + component)] =
                full_u[pgo::math::dense_index(vertex_j * static_cast<std::size_t>(Dim) + component)];
        }

        return local_u;
    }
};

} // namespace pgo::energy
