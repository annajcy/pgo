#include "pgo/io/obj_writer.hpp"
#include "pgo/io/obj_reader.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace pgo::io::test {

namespace {

[[nodiscard]] std::filesystem::path test_output_dir() {
    auto dir = std::filesystem::temp_directory_path() / "pgo_obj_io_tests";
    std::filesystem::create_directories(dir);
    return dir;
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
    std::ifstream input{path};
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

} // namespace

TEST(obj_reader, ReadsTriangularMeshAndExtractsUniqueEdges) {
    const auto path = test_output_dir() / "quad.obj";
    {
        std::ofstream output{path};
        output << "v 0 0 0\n";
        output << "v 1 0 0\n";
        output << "v 1 1 0\n";
        output << "v 0 1 0\n";
        output << "f 1/10/20 2/11/21 3/12/22\n";
        output << "f 1//20 3//22 4//23\n";
    }

    const auto mesh = pgo::io::read_obj_rest_mesh_3d(path);

    EXPECT_EQ(4, mesh.num_vertices());
    EXPECT_EQ(2, mesh.num_faces());
    EXPECT_EQ(5, mesh.num_edges());
    EXPECT_EQ(6, mesh.face_indices().size());
    EXPECT_EQ(10, mesh.edge_indices().size());
    EXPECT_EQ(0, mesh.face_vertex(0, 0));
    EXPECT_EQ(1, mesh.face_vertex(0, 1));
    EXPECT_EQ(2, mesh.face_vertex(0, 2));
}

TEST(obj_writer3d, WritesDisplacedLineFrameWhenMeshHasNoFaces) {
    pgo::storage::HostBuffer<double> positions{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edges{0, 1};
    const pgo::geometry::RestMesh<double, 3> mesh{std::move(positions), std::move(edges)};

    pgo::math::DVec<double> displacement{6};
    displacement << 0.0, 0.0, 0.0, 0.5, 0.25, -0.5;

    const auto dir = test_output_dir() / "frames";
    std::filesystem::remove_all(dir);
    pgo::io::ObjWriter3d writer{dir};
    const auto path = writer.write_frame(mesh, displacement);

    const auto text = read_text(path);
    EXPECT_NE(std::string::npos, text.find("v 1.5 0.25 -0.5"));
    EXPECT_NE(std::string::npos, text.find("l 1 2"));
}

TEST(obj_writer3d, WritesTriangleFaceRecords) {
    pgo::storage::HostBuffer<double> positions{
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0};
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> faces{0, 1, 2};
    const pgo::geometry::RestMesh<double, 3> mesh{std::move(positions), {}, std::move(faces)};

    pgo::math::DVec<double> displacement{9};
    displacement.setZero();

    const auto dir = test_output_dir() / "frames_tri";
    std::filesystem::remove_all(dir);
    pgo::io::ObjWriter3d writer{dir};
    const auto path = writer.write_frame(mesh, displacement);

    const auto text = read_text(path);
    EXPECT_NE(std::string::npos, text.find("f 1 2 3"));
}

TEST(obj_writer3d, VertexPositionsEqualRestPlusDisplacement) {
    pgo::storage::HostBuffer<double> positions{
        0.0, 0.0, 0.0,
        2.0, 0.0, 0.0};
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edges{0, 1};
    const pgo::geometry::RestMesh<double, 3> mesh{std::move(positions), std::move(edges)};

    pgo::math::DVec<double> displacement{6};
    displacement << 0.5, 0.25, 0.125, 0.25, 0.5, 0.75;

    const auto dir = test_output_dir() / "frames_vpos";
    std::filesystem::remove_all(dir);
    pgo::io::ObjWriter3d writer{dir};
    const auto path = writer.write_frame(mesh, displacement);

    const auto text = read_text(path);
    EXPECT_NE(std::string::npos, text.find("v 0.5 0.25 0.125"));
    EXPECT_NE(std::string::npos, text.find("v 2.25 0.5 0.75"));
}

TEST(obj_writer3d, ThrowsOnDisplacementSizeMismatch) {
    pgo::storage::HostBuffer<double> positions{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edges{0, 1};
    const pgo::geometry::RestMesh<double, 3> mesh{std::move(positions), std::move(edges)};

    pgo::math::DVec<double> displacement{4}; // wrong: needs 6
    displacement.setZero();

    const auto dir = test_output_dir() / "frames_err";
    std::filesystem::remove_all(dir);
    pgo::io::ObjWriter3d writer{dir};
    EXPECT_THROW(std::ignore = writer.write_frame(mesh, displacement), std::exception);
}

} // namespace pgo::io::test
