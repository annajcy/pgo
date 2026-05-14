#include "pgo/core/geometry/rest_mesh.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace pgo::geometry::test {

TEST(RestMesh, StoresFlatRestPositionsAndTopology) {
    pgo::geometry::RestMesh<double, 3> mesh{
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
        {0, 1, 1, 2},
        {0, 1, 2},
    };

    EXPECT_EQ(3, mesh.num_vertices());
    EXPECT_EQ(2, mesh.num_edges());
    EXPECT_EQ(1, mesh.num_faces());
    EXPECT_EQ(4, mesh.edge_indices().size());
    EXPECT_EQ(3, mesh.face_indices().size());

    EXPECT_EQ(1, mesh.edge_vertex(0, 1));
    EXPECT_EQ(2, mesh.edge_vertex(1, 1));
    EXPECT_EQ(2, mesh.face_vertex(0, 2));

    const auto rest_position = mesh.rest_position(2);
    EXPECT_DOUBLE_EQ(0.0, rest_position[0]);
    EXPECT_DOUBLE_EQ(1.0, rest_position[1]);
    EXPECT_DOUBLE_EQ(0.0, rest_position[2]);
}

TEST(RestMesh, RejectsInvalidFlatBufferLengths) {
    EXPECT_THROW((pgo::geometry::RestMesh<double, 3>{{0.0, 1.0}, {}, {}}), std::runtime_error);
    EXPECT_THROW((pgo::geometry::RestMesh<double, 3>{{0.0, 0.0, 0.0}, {0}, {}}), std::runtime_error);
    EXPECT_THROW((pgo::geometry::RestMesh<double, 3>{{0.0, 0.0, 0.0}, {}, {0, 1}}), std::runtime_error);
}

} // namespace pgo::geometry::test
