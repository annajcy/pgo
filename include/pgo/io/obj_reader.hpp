#pragma once

#include "pgo/geometry/rest_mesh.hpp"

#include <filesystem>

namespace pgo::io {

[[nodiscard]] pgo::geometry::RestMesh<double, 3> read_obj_rest_mesh_3d(const std::filesystem::path& path);

} // namespace pgo::io
