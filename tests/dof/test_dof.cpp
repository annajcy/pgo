#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/dof/displacement.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/dof/reduced_dof_map.hpp"

#include <array>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace pgo::dof::test {

TEST(DofLayout, MapsVertexComponentsToFullDofs) {
    const pgo::dof::DofLayout<3> layout{4};

    EXPECT_EQ(4, layout.num_vertices());
    EXPECT_EQ(12, layout.num_dofs());
    EXPECT_EQ(7, layout.index(2, 1));
}

TEST(Displacement, StoresFullVectorAndReturnsVertexDisplacement) {
    pgo::dof::Displacement<double, 3> displacement{2};
    displacement.vector() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    const auto vertex_displacement = displacement.at(1);

    EXPECT_EQ(6, displacement.vector().size());
    EXPECT_DOUBLE_EQ(4.0, vertex_displacement[0]);
    EXPECT_DOUBLE_EQ(5.0, vertex_displacement[1]);
    EXPECT_DOUBLE_EQ(6.0, vertex_displacement[2]);
}

TEST(DirichletBoundary, FixesAndPrescribesScalarDofs) {
    pgo::dof::DirichletBoundary<double> boundary{};

    boundary.fix_dof(2);
    boundary.prescribe_dof(5, 1.25);

    EXPECT_TRUE(boundary.is_fixed(2));
    EXPECT_TRUE(boundary.is_fixed(5));
    EXPECT_DOUBLE_EQ(0.0, boundary.value(2));
    EXPECT_DOUBLE_EQ(1.25, boundary.value(5));
}

TEST(DirichletBoundary, SupportsVertexAndListHelpers) {
    const pgo::dof::DofLayout<3> layout{5};
    pgo::dof::DirichletBoundary<double> boundary{};

    boundary.fix_vertex(layout, 1);
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(1, 0)));
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(1, 1)));
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(1, 2)));

    pgo::math::Vec<double, 3> prescribed{};
    prescribed << 1.0, 2.0, 3.0;
    boundary.prescribe_vertex(layout, 2, prescribed);
    EXPECT_DOUBLE_EQ(2.0, boundary.value(layout.index(2, 1)));

    const std::array<std::size_t, 2> fixed_vertices{0, 4};
    boundary.fix_vertices(layout, fixed_vertices);
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(4, 2)));

    pgo::math::Vec<double, 3> shared_value{};
    shared_value << 4.0, 5.0, 6.0;
    boundary.prescribe_vertices(layout, fixed_vertices, shared_value);
    EXPECT_DOUBLE_EQ(4.0, boundary.value(layout.index(0, 0)));
    EXPECT_DOUBLE_EQ(6.0, boundary.value(layout.index(4, 2)));
}

TEST(DirichletBoundary, SupportsPerVertexFlatPrescribedValues) {
    const pgo::dof::DofLayout<2> layout{3};
    pgo::dof::DirichletBoundary<double> boundary{};
    const std::array<std::size_t, 2> vertices{0, 2};
    const std::array<double, 4> values{1.0, 2.0, 3.0, 4.0};

    boundary.prescribe_vertices_by_list(layout, vertices, values);

    EXPECT_DOUBLE_EQ(1.0, boundary.value(layout.index(0, 0)));
    EXPECT_DOUBLE_EQ(2.0, boundary.value(layout.index(0, 1)));
    EXPECT_DOUBLE_EQ(3.0, boundary.value(layout.index(2, 0)));
    EXPECT_DOUBLE_EQ(4.0, boundary.value(layout.index(2, 1)));

    const std::array<double, 3> wrong_length_values{1.0, 2.0, 3.0};
    EXPECT_THROW(boundary.prescribe_vertices_by_list(layout, vertices, wrong_length_values), std::runtime_error);
}

TEST(ReducedDofMap, PacksAndUnpacksDisplacements) {
    pgo::dof::DirichletBoundary<double> boundary{};
    boundary.prescribe_dof(1, 10.0);
    boundary.prescribe_dof(4, 20.0);
    const pgo::dof::ReducedDofMap<double> map{6, boundary};

    pgo::math::DVec<double> full_u{6};
    full_u << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;

    const auto free_u = map.pack_displacement(full_u);
    EXPECT_EQ(4, free_u.size());
    EXPECT_DOUBLE_EQ(0.0, free_u[0]);
    EXPECT_DOUBLE_EQ(2.0, free_u[1]);
    EXPECT_DOUBLE_EQ(3.0, free_u[2]);
    EXPECT_DOUBLE_EQ(5.0, free_u[3]);

    pgo::math::DVec<double> replacement_free_u{4};
    replacement_free_u << 9.0, 8.0, 7.0, 6.0;

    const auto unpacked_u = map.unpack_displacement(replacement_free_u, boundary);
    EXPECT_DOUBLE_EQ(9.0, unpacked_u[0]);
    EXPECT_DOUBLE_EQ(10.0, unpacked_u[1]);
    EXPECT_DOUBLE_EQ(8.0, unpacked_u[2]);
    EXPECT_DOUBLE_EQ(7.0, unpacked_u[3]);
    EXPECT_DOUBLE_EQ(20.0, unpacked_u[4]);
    EXPECT_DOUBLE_EQ(6.0, unpacked_u[5]);
}

TEST(ReducedDofMap, ReducesVectorsAndSparseMatrices) {
    pgo::dof::DirichletBoundary<double> boundary{};
    boundary.fix_dof(1);
    const pgo::dof::ReducedDofMap<double> map{4, boundary};

    pgo::math::DVec<double> full_vector{4};
    full_vector << 10.0, 11.0, 12.0, 13.0;
    const auto reduced_vector = map.reduce_vector(full_vector);
    EXPECT_EQ(3, reduced_vector.size());
    EXPECT_DOUBLE_EQ(10.0, reduced_vector[0]);
    EXPECT_DOUBLE_EQ(12.0, reduced_vector[1]);
    EXPECT_DOUBLE_EQ(13.0, reduced_vector[2]);

    pgo::math::SparseMat<double> full_matrix{4, 4};
    std::vector<pgo::math::Triplet<double>> triplets{
        {0, 0, 1.0}, {0, 1, 99.0}, {2, 0, 2.0}, {2, 3, 3.0}, {1, 2, 88.0}, {3, 3, 4.0},
    };
    full_matrix.setFromTriplets(triplets.begin(), triplets.end());

    const auto reduced_matrix = map.reduce_sparse_mat(full_matrix);
    EXPECT_EQ(3, reduced_matrix.rows());
    EXPECT_EQ(3, reduced_matrix.cols());
    EXPECT_DOUBLE_EQ(1.0, reduced_matrix.coeff(0, 0));
    EXPECT_DOUBLE_EQ(2.0, reduced_matrix.coeff(1, 0));
    EXPECT_DOUBLE_EQ(3.0, reduced_matrix.coeff(1, 2));
    EXPECT_DOUBLE_EQ(4.0, reduced_matrix.coeff(2, 2));
    EXPECT_DOUBLE_EQ(0.0, reduced_matrix.coeff(0, 1));
}

} // namespace pgo::dof::test
