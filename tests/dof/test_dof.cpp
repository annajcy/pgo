#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/dof/displacement.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/dof/reduced_dof_map.hpp"

#include <Eigen/SparseLU>
#include <array>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <vector>

namespace pgo::dof::test {

static_assert(pgo::dof::DirichletBoundaryLike<pgo::dof::DirichletBoundary<double>, double>);
static_assert(pgo::dof::DirichletBoundaryLike<pgo::dof::ReducedDofMap<double>, double>);

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
    boundary.prescribe_dof(5, 2.5);

    EXPECT_TRUE(boundary.is_fixed(2));
    EXPECT_TRUE(boundary.is_fixed(5));
    EXPECT_DOUBLE_EQ(0.0, boundary.value(2));
    EXPECT_DOUBLE_EQ(2.5, boundary.value(5));
    ASSERT_TRUE(boundary.fixed_value(5).has_value());
    EXPECT_DOUBLE_EQ(2.5, *boundary.fixed_value(5));
    EXPECT_EQ(std::nullopt, boundary.fixed_value(0));
    EXPECT_THROW(static_cast<void>(boundary.value(0)), std::runtime_error);
}

TEST(DirichletBoundary, SupportsVertexAndListHelpers) {
    const pgo::dof::DofLayout<3> layout{5};
    pgo::dof::DirichletBoundary<double> boundary{};

    pgo::dof::fix_vertex(boundary, layout, 1);
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(1, 0)));
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(1, 1)));
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(1, 2)));

    pgo::math::Vec<double, 3> prescribed{};
    prescribed << 1.0, 2.0, 3.0;
    pgo::dof::prescribe_vertex(boundary, layout, 2, prescribed);
    EXPECT_DOUBLE_EQ(2.0, boundary.value(layout.index(2, 1)));

    const std::array<std::size_t, 2> fixed_vertices{0, 4};
    pgo::dof::fix_vertices(boundary, layout, fixed_vertices);
    EXPECT_DOUBLE_EQ(0.0, boundary.value(layout.index(4, 2)));

    pgo::math::Vec<double, 3> shared_value{};
    shared_value << 4.0, 5.0, 6.0;
    pgo::dof::prescribe_vertices(boundary, layout, fixed_vertices, shared_value);
    EXPECT_DOUBLE_EQ(4.0, boundary.value(layout.index(0, 0)));
    EXPECT_DOUBLE_EQ(6.0, boundary.value(layout.index(4, 2)));
}

TEST(DirichletBoundary, SupportsPerVertexFlatPrescribedValues) {
    const pgo::dof::DofLayout<2> layout{3};
    pgo::dof::DirichletBoundary<double> boundary{};
    const std::array<std::size_t, 2> vertices{0, 2};
    const std::array<double, 4> values{1.0, 2.0, 3.0, 4.0};

    pgo::dof::prescribe_vertices_by_list(boundary, layout, vertices, pgo::storage::ConstArrayView<double>{values});

    EXPECT_DOUBLE_EQ(1.0, boundary.value(layout.index(0, 0)));
    EXPECT_DOUBLE_EQ(2.0, boundary.value(layout.index(0, 1)));
    EXPECT_DOUBLE_EQ(3.0, boundary.value(layout.index(2, 0)));
    EXPECT_DOUBLE_EQ(4.0, boundary.value(layout.index(2, 1)));

    const std::array<double, 3> wrong_length_values{1.0, 2.0, 3.0};
    EXPECT_THROW(pgo::dof::prescribe_vertices_by_list(boundary, layout, vertices,
                                                      pgo::storage::ConstArrayView<double>{wrong_length_values}),
                 std::runtime_error);
}

TEST(ReducedDofMap, SatisfiesDirichletBoundaryLikeAndSupportsVertexHelpers) {
    const pgo::dof::DofLayout<2> layout{3};
    pgo::dof::DirichletBoundary<double> boundary{};

    pgo::math::Vec<double, 2> value{};
    value << 3.0, 4.0;
    pgo::dof::prescribe_vertex(boundary, layout, 1, value);

    const pgo::dof::ReducedDofMap<double> map{layout.num_dofs(), boundary};

    // DirichletBoundaryLike and convenience query interface
    EXPECT_TRUE(map.is_fixed(layout.index(1, 0)));
    EXPECT_TRUE(map.is_fixed(layout.index(1, 1)));
    EXPECT_FALSE(map.is_fixed(layout.index(0, 0)));
    ASSERT_TRUE(map.fixed_value(layout.index(1, 0)).has_value());
    ASSERT_TRUE(map.fixed_value(layout.index(1, 1)).has_value());
    EXPECT_EQ(std::nullopt, map.fixed_value(layout.index(0, 0)));
    EXPECT_DOUBLE_EQ(3.0, map.value(layout.index(1, 0)));
    EXPECT_DOUBLE_EQ(4.0, map.value(layout.index(1, 1)));

    // Mapping interface
    EXPECT_EQ(4, map.free_dofs());
    EXPECT_FALSE(map.is_free(layout.index(1, 0)));
    EXPECT_FALSE(map.is_free(layout.index(1, 1)));
}

TEST(ReducedDofMap, ScattersSolutionsAndDirectionsWithDifferentFixedDofSemantics) {
    pgo::dof::DirichletBoundary<double> boundary{};
    boundary.prescribe_dof(1, 10.0);
    boundary.prescribe_dof(4, 20.0);
    const pgo::dof::ReducedDofMap<double> map{6, boundary};
    EXPECT_EQ(0, map.full_dof(0));
    EXPECT_EQ(2, map.full_dof(1));
    EXPECT_EQ(3, map.full_dof(2));
    EXPECT_EQ(5, map.full_dof(3));
    EXPECT_EQ(1, map.free_dof(2));
    EXPECT_THROW(static_cast<void>(map.full_dof(4)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(map.free_dof(1)), std::runtime_error);
    ASSERT_TRUE(map.fixed_value(1).has_value());
    ASSERT_TRUE(map.fixed_value(4).has_value());
    EXPECT_DOUBLE_EQ(10.0, *map.fixed_value(1));
    EXPECT_DOUBLE_EQ(20.0, *map.fixed_value(4));
    EXPECT_EQ(std::nullopt, map.fixed_value(0));
    EXPECT_THROW(static_cast<void>(map.value(0)), std::runtime_error);

    pgo::math::DVec<double> free_values{4};
    free_values << 9.0, 8.0, 7.0, 6.0;

    boundary.prescribe_dof(1, 100.0);
    boundary.prescribe_dof(4, 200.0);

    const auto solution = map.scatter_solution(free_values);
    EXPECT_DOUBLE_EQ(9.0, solution[0]);
    EXPECT_DOUBLE_EQ(10.0, solution[1]);
    EXPECT_DOUBLE_EQ(8.0, solution[2]);
    EXPECT_DOUBLE_EQ(7.0, solution[3]);
    EXPECT_DOUBLE_EQ(20.0, solution[4]);
    EXPECT_DOUBLE_EQ(6.0, solution[5]);

    const auto direction = map.scatter_direction(free_values);
    EXPECT_DOUBLE_EQ(9.0, direction[0]);
    EXPECT_DOUBLE_EQ(0.0, direction[1]);
    EXPECT_DOUBLE_EQ(8.0, direction[2]);
    EXPECT_DOUBLE_EQ(7.0, direction[3]);
    EXPECT_DOUBLE_EQ(0.0, direction[4]);
    EXPECT_DOUBLE_EQ(6.0, direction[5]);
}

TEST(ReducedDofMap, RestrictsVectorsAndSparseMatricesToFreeDofs) {
    // 6 full DOFs, fix DOFs 1 and 4 → 4 free DOFs (0, 2, 3, 5)
    pgo::dof::DirichletBoundary<double> boundary{};
    boundary.fix_dof(1);
    boundary.fix_dof(4);
    const pgo::dof::ReducedDofMap<double> map{6, boundary};

    // Build a 6×6 sparse matrix with entries at known positions
    std::vector<pgo::math::Triplet<double>> full_triplets;
    full_triplets.emplace_back(0, 0, 1.0); // free×free → kept
    full_triplets.emplace_back(0, 2, 2.0); // free×free → kept
    full_triplets.emplace_back(1, 0, 3.0); // fixed×free → dropped
    full_triplets.emplace_back(2, 1, 4.0); // free×fixed → dropped
    full_triplets.emplace_back(3, 5, 5.0); // free×free → kept
    full_triplets.emplace_back(5, 3, 6.0); // free×free → kept
    full_triplets.emplace_back(4, 4, 7.0); // fixed×fixed → dropped

    pgo::math::SparseMat<double> full_mat(6, 6);
    full_mat.setFromTriplets(full_triplets.begin(), full_triplets.end());

    pgo::math::DVec<double> full_vector{6};
    full_vector << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;
    const auto free_vector = map.restrict_vector_to_free(full_vector);
    EXPECT_EQ(4, free_vector.size());
    EXPECT_DOUBLE_EQ(10.0, free_vector[0]);
    EXPECT_DOUBLE_EQ(30.0, free_vector[1]);
    EXPECT_DOUBLE_EQ(40.0, free_vector[2]);
    EXPECT_DOUBLE_EQ(60.0, free_vector[3]);

    // Restrict: 6×6 → 4×4, free DOFs map: 0→0, 2→1, 3→2, 5→3
    const auto free_mat = map.restrict_matrix_to_free(full_mat);
    EXPECT_EQ(4, free_mat.rows());
    EXPECT_EQ(4, free_mat.cols());
    EXPECT_DOUBLE_EQ(1.0, free_mat.coeff(0, 0)); // (0,0)→(0,0)
    EXPECT_DOUBLE_EQ(2.0, free_mat.coeff(0, 1)); // (0,2)→(0,1)
    EXPECT_DOUBLE_EQ(5.0, free_mat.coeff(2, 3)); // (3,5)→(2,3)
    EXPECT_DOUBLE_EQ(6.0, free_mat.coeff(3, 2)); // (5,3)→(3,2)
    EXPECT_DOUBLE_EQ(0.0, free_mat.coeff(1, 0)); // not present
}

TEST(ReducedDofMap, EliminatesRhsForDirichletUsingNonZeroPrescribedValues) {
    // 4 full DOFs, fix DOF 1 = 3.0 and DOF 3 = 5.0 → 2 free DOFs (0, 2)
    //
    //     K = [2 1 0 0]    f = [1]
    //         [1 3 1 0]        [2]
    //         [0 1 4 1]        [3]
    //         [0 0 1 5]        [4]
    //
    // Partition: free = {0, 2}, fixed = {1, 3}
    //
    //     K_ff = [2 0]    K_fc = [1 0]    g = [3.0]
    //            [0 4]           [1 1]        [5.0]
    //
    //     f_reduced = f_free - K_fc * g = [1 - 3]  = [-2]
    //                                     [3 - 8]    [-5]

    pgo::dof::DirichletBoundary<double> boundary{};
    boundary.prescribe_dof(1, 3.0);
    boundary.prescribe_dof(3, 5.0);
    const pgo::dof::ReducedDofMap<double> map{4, boundary};

    std::vector<pgo::math::Triplet<double>> triplets;
    triplets.emplace_back(0, 0, 2.0);
    triplets.emplace_back(0, 1, 1.0);
    triplets.emplace_back(1, 0, 1.0);
    triplets.emplace_back(1, 1, 3.0);
    triplets.emplace_back(1, 2, 1.0);
    triplets.emplace_back(2, 1, 1.0);
    triplets.emplace_back(2, 2, 4.0);
    triplets.emplace_back(2, 3, 1.0);
    triplets.emplace_back(3, 2, 1.0);
    triplets.emplace_back(3, 3, 5.0);

    pgo::math::SparseMat<double> K(4, 4);
    K.setFromTriplets(triplets.begin(), triplets.end());

    pgo::math::DVec<double> f{4};
    f << 1.0, 2.0, 3.0, 4.0;

    // eliminate_rhs_for_dirichlet should give f_free - K_fc * g
    const auto reduced_f = map.eliminate_rhs_for_dirichlet(f, K);
    EXPECT_EQ(2, reduced_f.size());
    EXPECT_DOUBLE_EQ(-2.0, reduced_f[0]); // 1 - 1*3 - 0*5 = -2
    EXPECT_DOUBLE_EQ(-5.0, reduced_f[1]); // 3 - 1*3 - 1*5 = -5
}

TEST(ReducedDofMap, EndToEndEliminationSolvesCorrectly) {
    // Same 4×4 SPD system as above. Solve via elimination and verify K*u = f
    // for the free DOF equations.
    //
    // Expected solution: u = [-1, 3, -1.25, 5]

    pgo::dof::DirichletBoundary<double> boundary{};
    boundary.prescribe_dof(1, 3.0);
    boundary.prescribe_dof(3, 5.0);
    const pgo::dof::ReducedDofMap<double> map{4, boundary};

    std::vector<pgo::math::Triplet<double>> triplets;
    triplets.emplace_back(0, 0, 2.0);
    triplets.emplace_back(0, 1, 1.0);
    triplets.emplace_back(1, 0, 1.0);
    triplets.emplace_back(1, 1, 3.0);
    triplets.emplace_back(1, 2, 1.0);
    triplets.emplace_back(2, 1, 1.0);
    triplets.emplace_back(2, 2, 4.0);
    triplets.emplace_back(2, 3, 1.0);
    triplets.emplace_back(3, 2, 1.0);
    triplets.emplace_back(3, 3, 5.0);

    pgo::math::SparseMat<double> K(4, 4);
    K.setFromTriplets(triplets.begin(), triplets.end());

    pgo::math::DVec<double> f{4};
    f << 1.0, 2.0, 3.0, 4.0;

    // Step 1: Eliminate — restrict matrix and eliminate RHS
    const auto K_ff = map.restrict_matrix_to_free(K);
    const auto f_reduced = map.eliminate_rhs_for_dirichlet(f, K);

    // Step 2: Solve K_ff * u_free = f_reduced
    //   K_ff = [2 0]  f_reduced = [-2]  → u_free = [-1, -1.25]
    //          [0 4]               [-5]
    Eigen::SparseLU<pgo::math::SparseMat<double>> solver;
    solver.compute(K_ff);
    ASSERT_EQ(Eigen::Success, solver.info());
    const pgo::math::DVec<double> u_free = solver.solve(f_reduced);

    EXPECT_DOUBLE_EQ(-1.0, u_free[0]);
    EXPECT_DOUBLE_EQ(-1.25, u_free[1]);

    // Step 3: Scatter — reconstruct full displacement solution
    const auto u_full = map.scatter_solution(u_free);
    EXPECT_EQ(4, u_full.size());
    EXPECT_DOUBLE_EQ(-1.0, u_full[0]);  // free DOF 0
    EXPECT_DOUBLE_EQ(3.0, u_full[1]);   // fixed DOF 1 = prescribed
    EXPECT_DOUBLE_EQ(-1.25, u_full[2]); // free DOF 2
    EXPECT_DOUBLE_EQ(5.0, u_full[3]);   // fixed DOF 3 = prescribed

    // Step 4: Verify K * u = f for the free DOF rows
    const pgo::math::DVec<double> residual = K * u_full - f;
    // Free DOF rows should have zero residual
    EXPECT_NEAR(0.0, residual[0], 1e-14); // row 0 (free)
    EXPECT_NEAR(0.0, residual[2], 1e-14); // row 2 (free)
    // Fixed DOF rows give reaction forces (not necessarily zero)
}

} // namespace pgo::dof::test
