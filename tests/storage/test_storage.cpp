#include <gtest/gtest.h>

#include <span>
#include <type_traits>
#include <vector>

#include "pgo/core/storage/array_view.hpp"
#include "pgo/core/storage/host_buffer.hpp"

namespace pgo::storage::test {

static_assert(std::is_same_v<pgo::storage::ArrayView<double>, std::span<double>>);
static_assert(std::is_same_v<pgo::storage::ConstArrayView<double>, std::span<const double>>);
static_assert(std::is_same_v<pgo::storage::HostBuffer<double>, std::vector<double>>);

TEST(StoragePrimitive, ArrayViewReferencesHostBufferStorage)
{
    pgo::storage::HostBuffer<double> buffer{1.0, 2.0, 3.0};
    pgo::storage::ArrayView<double> view{buffer};

    view[1] = 4.0;

    EXPECT_EQ(3, view.size());
    EXPECT_DOUBLE_EQ(4.0, buffer[1]);
}

TEST(StoragePrimitive, ConstArrayViewReadsHostBufferStorage)
{
    const pgo::storage::HostBuffer<double> buffer{1.0, 2.0, 3.0};
    pgo::storage::ConstArrayView<double> view{buffer};

    EXPECT_EQ(3, view.size());
    EXPECT_DOUBLE_EQ(3.0, view[2]);
}

} // namespace pgo::storage::test
