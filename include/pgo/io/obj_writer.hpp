#pragma once

#include "pgo/core/geometry/rest_mesh.hpp"
#include "pgo/core/math/types.hpp"

#include <filesystem>
#include <string>

namespace pgo::io {

class ObjWriter3d {
    std::filesystem::path m_output_dir;
    std::size_t m_next_frame = 0;

public:
    explicit ObjWriter3d(std::filesystem::path output_dir);

    [[nodiscard]] std::filesystem::path write_frame(const pgo::geometry::RestMesh<double, 3>& mesh,
                                                    const pgo::math::DVec<double>& displacement);

private:
    [[nodiscard]] static std::string frame_name(std::size_t frame);
};

} // namespace pgo::io
