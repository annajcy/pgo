#pragma once

#if defined(PGO_ENABLE_ALEMBIC)

#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/math/types.hpp"

#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Imath/ImathVec.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace pgo::io {

template <typename T, int Dim>
class AbcWriter {
    static_assert(Dim > 0);

    std::unique_ptr<Alembic::Abc::OArchive> m_archive;
    std::unique_ptr<Alembic::AbcGeom::OPolyMesh> m_mesh_obj;
    Alembic::AbcGeom::OPolyMeshSchema m_schema;
    std::vector<std::int32_t> m_face_indices;
    std::vector<std::int32_t> m_face_counts;
    Alembic::Abc::Int32ArraySample m_face_indices_sample;
    Alembic::Abc::Int32ArraySample m_face_counts_sample;

public:
    AbcWriter(const std::filesystem::path& output_path, double fps,
              std::span<const std::uint32_t> face_indices,
              std::span<const int> face_counts)
        : m_face_indices(face_indices.begin(), face_indices.end())
        , m_face_counts(face_counts.begin(), face_counts.end())
        , m_face_indices_sample(m_face_indices.data(), m_face_indices.size())
        , m_face_counts_sample(m_face_counts.data(), m_face_counts.size())
    {
        namespace Abc = Alembic::Abc;
        namespace AbcGeom = Alembic::AbcGeom;

        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        m_archive = std::make_unique<Abc::OArchive>(
            Alembic::AbcCoreOgawa::WriteArchive(), output_path.string());

        const AbcGeom::TimeSampling ts{1.0 / fps, 0.0};
        const auto ts_index = m_archive->addTimeSampling(ts);

        m_mesh_obj = std::make_unique<AbcGeom::OPolyMesh>(
            m_archive->getTop(), "mesh", ts_index);
        m_schema = m_mesh_obj->getSchema();
        m_schema.setTimeSampling(ts_index);
    }

    void write_frame(const pgo::geometry::RestMesh<T, Dim>& mesh,
                     const pgo::math::DVec<T>& displacement)
    {
        const auto nv = mesh.num_vertices();
        std::vector<float> xyz;
        xyz.reserve(nv * 3);
        for (std::size_t v = 0; v < nv; ++v) {
            for (std::size_t c = 0; c < 3; ++c) {
                T val{};
                if (c < static_cast<std::size_t>(Dim)) {
                    const auto offset = v * static_cast<std::size_t>(Dim) + c;
                    val = mesh.rest_positions()[offset] + displacement[pgo::math::dense_index(offset)];
                }
                xyz.push_back(static_cast<float>(val));
            }
        }
        write_sample(xyz);
    }

    void write_frame(std::span<const float> positions_x3_per_vertex)
    {
        write_sample(positions_x3_per_vertex);
    }

private:
    void write_sample(std::span<const float> xyz)
    {
        namespace AbcGeom = Alembic::AbcGeom;

        const auto nv = xyz.size() / 3;
        std::vector<Imath::V3f> points;
        points.reserve(nv);
        for (std::size_t i = 0; i < xyz.size(); i += 3) {
            points.emplace_back(xyz[i], xyz[i + 1], xyz[i + 2]);
        }

        AbcGeom::OPolyMeshSchema::Sample sample{
            AbcGeom::P3fArraySample{points.data(), points.size()},
            m_face_indices_sample, m_face_counts_sample};
        m_schema.set(sample);
    }
};

}  // namespace pgo::io

#endif  // PGO_ENABLE_ALEMBIC
