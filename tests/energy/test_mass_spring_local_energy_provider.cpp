#include "pgo/assembly/cpu_assembler.hpp"
#include "pgo/energy/mass_spring_local_energy_provider.hpp"
#include "pgo/math/finite_difference.hpp"

#include <array>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace pgo::energy::test {

namespace {

[[nodiscard]] pgo::geometry::RestMesh<double, 2> make_single_spring_mesh() {
    return pgo::geometry::RestMesh<double, 2>{{0.0, 0.0, 1.0, 0.0}, {0, 1}};
}

[[nodiscard]] pgo::geometry::RestMesh<double, 2> make_two_spring_mesh() {
    return pgo::geometry::RestMesh<double, 2>{{0.0, 0.0, 1.0, 0.0, 2.0, 0.0}, {0, 1, 1, 2}};
}

[[nodiscard]] pgo::math::DVec<double> local_to_full_single_spring(const pgo::math::DVec<double>& local_u) {
    pgo::math::DVec<double> full_u{4};
    full_u = local_u;
    return full_u;
}

void expect_dense_near(const pgo::math::DMat<double>& expected, const pgo::math::DMat<double>& actual,
                       const double tolerance) {
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    for (pgo::math::DenseIndex row = 0; row < expected.rows(); ++row) {
        for (pgo::math::DenseIndex col = 0; col < expected.cols(); ++col) {
            EXPECT_NEAR(expected(row, col), actual(row, col), tolerance) << "at (" << row << ", " << col << ")";
        }
    }
}

} // namespace

static_assert(pgo::energy::LocalEnergyProvider<pgo::energy::MassSpringLocalEnergyProvider<double, 2>, double>);
static_assert(pgo::energy::FusedLocalEnergyProvider<pgo::energy::MassSpringLocalEnergyProvider<double, 2>, double>);
static_assert(pgo::energy::LocalEnergyModel<pgo::energy::MassSpringLocalEnergyModel<double, 2>, double,
                                            pgo::energy::MassSpringLocalData<double, 2>>);
static_assert(pgo::energy::FusedLocalEnergyModel<pgo::energy::MassSpringLocalEnergyModel<double, 2>, double,
                                                 pgo::energy::MassSpringLocalData<double, 2>>);

TEST(MassSpringLocalEnergyModel, RestStateIsZeroAndRejectsInvalidData) {
    using Model = pgo::energy::MassSpringLocalEnergyModel<double, 2>;

    pgo::energy::MassSpringLocalData<double, 2> data{};
    data.rest_i << 0.0, 0.0;
    data.rest_j << 1.0, 0.0;
    data.stiffness = 10.0;
    data.min_length = 1e-8;

    pgo::assembly::LocalVector<double> local_u{4};
    local_u.setZero();

    pgo::assembly::LocalVector<double> gradient;
    EXPECT_EQ(4, Model::local_dof_count(data));
    EXPECT_DOUBLE_EQ(0.0, Model::value(data, local_u));
    Model::gradient(data, local_u, gradient);
    EXPECT_NEAR(0.0, gradient.norm(), 1e-14);

    data.rest_j << 0.0, 0.0;
    EXPECT_THROW(static_cast<void>(Model::value(data, local_u)), std::runtime_error);

    data.rest_j << 1.0, 0.0;
    data.stiffness = -1.0;
    EXPECT_THROW(static_cast<void>(Model::value(data, local_u)), std::runtime_error);

    data.stiffness = 10.0;
    data.min_length = 0.0;
    EXPECT_THROW(static_cast<void>(Model::value(data, local_u)), std::runtime_error);

    data.min_length = 1e-8;
    pgo::assembly::LocalVector<double> wrong_size_local_u{3};
    wrong_size_local_u.setZero();
    EXPECT_THROW(static_cast<void>(Model::value(data, wrong_size_local_u)), std::runtime_error);
}

TEST(MassSpringLocalEnergyModel, LocalDerivativesMatchFiniteDifferencesForStretchAndCompression) {
    using Model = pgo::energy::MassSpringLocalEnergyModel<double, 2>;

    pgo::energy::MassSpringLocalData<double, 2> data{};
    data.rest_i << 0.0, 0.0;
    data.rest_j << 1.0, 0.0;
    data.stiffness = 7.0;
    data.min_length = 1e-8;

    const std::vector<pgo::math::DVec<double>> cases = [] {
        std::vector<pgo::math::DVec<double>> values;
        pgo::math::DVec<double> stretched{4};
        stretched << 0.1, 0.2, 0.35, -0.15;
        values.push_back(stretched);
        pgo::math::DVec<double> compressed{4};
        compressed << 0.2, -0.1, -0.25, 0.3;
        values.push_back(compressed);
        return values;
    }();

    for (const auto& local_u : cases) {
        pgo::assembly::LocalVector<double> analytic_gradient;
        pgo::assembly::LocalMatrix<double> analytic_hessian;
        Model::gradient(data, local_u, analytic_gradient);
        Model::hessian(data, local_u, analytic_hessian);

        const auto fd_gradient = pgo::math::finite_difference_gradient(
            [&](const pgo::math::DVec<double>& value_local_u) { return Model::value(data, value_local_u); }, local_u,
            1e-6);
        const auto fd_hessian = pgo::math::finite_difference_hessian_from_gradient(
            [&](const pgo::math::DVec<double>& gradient_local_u) {
                pgo::assembly::LocalVector<double> local_gradient;
                Model::gradient(data, gradient_local_u, local_gradient);
                return local_gradient;
            },
            local_u, 1e-6);

        ASSERT_EQ(fd_gradient.size(), analytic_gradient.size());
        for (pgo::math::DenseIndex i = 0; i < fd_gradient.size(); ++i) {
            EXPECT_NEAR(fd_gradient[i], analytic_gradient[i], 1e-7);
        }
        expect_dense_near(fd_hessian, analytic_hessian, 1e-6);
    }
}

TEST(MassSpringLocalEnergyProvider, ExposesLocalSpringApiAndRestStateIsZero) {
    const auto mesh = make_single_spring_mesh();
    const pgo::energy::MassSpringLocalEnergyProvider<double, 2> energy{mesh, 10.0};

    EXPECT_EQ(1, energy.local_count());
    EXPECT_EQ(4, energy.max_local_dofs());

    std::vector<std::size_t> dofs;
    energy.local_dofs(0, dofs);
    EXPECT_EQ((std::vector<std::size_t>{0, 1, 2, 3}), dofs);

    pgo::math::DVec<double> u{4};
    u.setZero();
    pgo::assembly::LocalVector<double> gradient;

    EXPECT_DOUBLE_EQ(0.0, energy.local_value(0, u));
    energy.local_gradient(0, u, gradient);
    EXPECT_NEAR(0.0, gradient.norm(), 1e-14);

    u[2] = 0.25;
    EXPECT_GT(energy.local_value(0, u), 0.0);
}

TEST(MassSpringLocalEnergyProvider, RejectsZeroLengthRestSpring) {
    const pgo::geometry::RestMesh<double, 2> mesh{{0.0, 0.0, 0.0, 0.0}, {0, 1}};
    EXPECT_THROW((pgo::energy::MassSpringLocalEnergyProvider<double, 2>{mesh, 1.0}), std::runtime_error);
    EXPECT_THROW((pgo::energy::MassSpringLocalEnergyProvider<double, 2>{make_single_spring_mesh(), -1.0}),
                 std::runtime_error);
}

TEST(MassSpringLocalEnergyProvider, SupportsPerEdgeStiffness) {
    const auto mesh = make_two_spring_mesh();
    const std::array<double, 2> stiffnesses{2.0, 8.0};
    const pgo::energy::MassSpringLocalEnergyProvider<double, 2> energy{
        mesh, pgo::storage::ConstArrayView<double>{stiffnesses}};

    pgo::math::DVec<double> u{6};
    u << 0.0, 0.0, 0.2, 0.0, -0.3, 0.0;

    const pgo::energy::MassSpringLocalEnergyProvider<double, 2> edge_0_uniform{mesh, stiffnesses[0]};
    const pgo::energy::MassSpringLocalEnergyProvider<double, 2> edge_1_uniform{mesh, stiffnesses[1]};

    EXPECT_DOUBLE_EQ(edge_0_uniform.local_value(0, u), energy.local_value(0, u));
    EXPECT_DOUBLE_EQ(edge_1_uniform.local_value(1, u), energy.local_value(1, u));
    EXPECT_NE(edge_0_uniform.local_value(1, u), energy.local_value(1, u));
}

TEST(MassSpringLocalEnergyProvider, RejectsInvalidPerEdgeStiffness) {
    const auto mesh = make_two_spring_mesh();
    const std::array<double, 1> wrong_count{1.0};
    const std::array<double, 2> negative{1.0, -1.0};

    EXPECT_THROW((pgo::energy::MassSpringLocalEnergyProvider<double, 2>{
                     mesh, pgo::storage::ConstArrayView<double>{wrong_count}}),
                 std::runtime_error);
    EXPECT_THROW(
        (pgo::energy::MassSpringLocalEnergyProvider<double, 2>{mesh, pgo::storage::ConstArrayView<double>{negative}}),
        std::runtime_error);
}

TEST(MassSpringLocalEnergyProvider, LocalDerivativesMatchFiniteDifferencesForStretchAndCompression) {
    const auto mesh = make_single_spring_mesh();
    const pgo::energy::MassSpringLocalEnergyProvider<double, 2> energy{mesh, 7.0};

    const std::vector<pgo::math::DVec<double>> cases = [] {
        std::vector<pgo::math::DVec<double>> values;
        pgo::math::DVec<double> stretched{4};
        stretched << 0.1, 0.2, 0.35, -0.15;
        values.push_back(stretched);
        pgo::math::DVec<double> compressed{4};
        compressed << 0.2, -0.1, -0.25, 0.3;
        values.push_back(compressed);
        return values;
    }();

    for (const auto& local_u : cases) {
        const auto full_u = local_to_full_single_spring(local_u);

        pgo::assembly::LocalVector<double> analytic_gradient;
        pgo::assembly::LocalMatrix<double> analytic_hessian;
        energy.local_gradient(0, full_u, analytic_gradient);
        energy.local_hessian(0, full_u, analytic_hessian);

        const auto fd_gradient = pgo::math::finite_difference_gradient(
            [&](const pgo::math::DVec<double>& value_local_u) {
                return energy.local_value(0, local_to_full_single_spring(value_local_u));
            },
            local_u, 1e-6);

        const auto fd_hessian = pgo::math::finite_difference_hessian_from_gradient(
            [&](const pgo::math::DVec<double>& gradient_local_u) {
                pgo::assembly::LocalVector<double> local_gradient;
                energy.local_gradient(0, local_to_full_single_spring(gradient_local_u), local_gradient);
                return local_gradient;
            },
            local_u, 1e-6);

        ASSERT_EQ(fd_gradient.size(), analytic_gradient.size());
        for (pgo::math::DenseIndex i = 0; i < fd_gradient.size(); ++i) {
            EXPECT_NEAR(fd_gradient[i], analytic_gradient[i], 1e-7);
        }
        expect_dense_near(fd_hessian, analytic_hessian, 1e-6);
    }
}

TEST(MassSpringLocalEnergyProvider, GlobalAssemblyDerivativesMatchFiniteDifferences) {
    const auto mesh = make_two_spring_mesh();
    const pgo::energy::MassSpringLocalEnergyProvider<double, 2> energy{mesh, 5.0};
    pgo::math::DVec<double> u{6};
    u << 0.1, 0.2, 0.35, -0.15, -0.2, 0.25;

    pgo::math::DVec<double> analytic_gradient;
    pgo::math::SparseMat<double> analytic_sparse_hessian;
    pgo::assembly::assemble_gradient<double>(energy, u, analytic_gradient);
    pgo::assembly::assemble_hessian<double>(energy, u, analytic_sparse_hessian);

    const auto fd_gradient = pgo::math::finite_difference_gradient(
        [&](const pgo::math::DVec<double>& value_u) { return pgo::assembly::assemble_value<double>(energy, value_u); },
        u, 1e-6);
    const auto fd_hessian = pgo::math::finite_difference_hessian_from_gradient(
        [&](const pgo::math::DVec<double>& gradient_u) {
            pgo::math::DVec<double> gradient;
            pgo::assembly::assemble_gradient<double>(energy, gradient_u, gradient);
            return gradient;
        },
        u, 1e-6);
    const pgo::math::DMat<double> analytic_hessian = pgo::math::DMat<double>{analytic_sparse_hessian};

    for (pgo::math::DenseIndex i = 0; i < fd_gradient.size(); ++i) {
        EXPECT_NEAR(fd_gradient[i], analytic_gradient[i], 1e-7);
    }
    expect_dense_near(fd_hessian, analytic_hessian, 1e-6);

    pgo::assembly::LocalVector<double> spring_0_gradient;
    pgo::assembly::LocalVector<double> spring_1_gradient;
    energy.local_gradient(0, u, spring_0_gradient);
    energy.local_gradient(1, u, spring_1_gradient);
    EXPECT_NEAR(spring_0_gradient[2] + spring_1_gradient[0], analytic_gradient[2], 1e-14);
    EXPECT_NEAR(spring_0_gradient[3] + spring_1_gradient[1], analytic_gradient[3], 1e-14);
}

} // namespace pgo::energy::test
