#pragma once

#if defined(PGO_ENABLE_ALEMBIC)

#include "pgo/core/geometry/rest_mesh.hpp"
#include "pgo/core/math/types.hpp"

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Imath/ImathVec.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace pgo::io {

class AbcWriter3d {
    std::unique_ptr<Alembic::Abc::OArchive> m_archive;
    std::unique_ptr<Alembic::AbcGeom::OPolyMesh> m_mesh_obj;
    Alembic::AbcGeom::OPolyMeshSchema m_schema;
    std::vector<std::int32_t> m_face_indices;
    std::vector<std::int32_t> m_face_counts;
    Alembic::Abc::Int32ArraySample m_face_indices_sample;
    Alembic::Abc::Int32ArraySample m_face_counts_sample;

public:
    AbcWriter3d(const std::filesystem::path& output_path, double fps,
                std::span<const std::uint32_t> face_indices,
                std::span<const int> face_counts);

    void write_frame(const pgo::geometry::RestMesh<double, 3>& mesh,
                     const pgo::math::DVec<double>& displacement);

    void write_frame(std::span<const float> positions_x3_per_vertex);

private:
    void write_sample(std::span<const float> xyz);
};

}  // namespace pgo::io

#endif  // PGO_ENABLE_ALEMBIC
