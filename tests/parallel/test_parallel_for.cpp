#include <pgo/parallel/parallel_for.hpp>
#include <pgo/parallel/runtime.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

class ParallelFixture : public ::testing::Test {
protected:
    void SetUp() override {
        pgo::parallel::set_thread_count(0);
    }
    void TearDown() override {
        pgo::parallel::set_thread_count(0);
    }
};

using ParallelFor = ParallelFixture;
using ParallelRuntime = ParallelFixture;

TEST_F(ParallelFor, EmptyRangeDoesNothing) {
    int count = 0;
    pgo::parallel::parallel_for(5, 5, [&](int) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST_F(ParallelFor, VisitsEachIndexOnce) {
    std::vector<int> visits(128, 0);
    pgo::parallel::set_thread_count(1);

    pgo::parallel::parallel_for(0, static_cast<int>(visits.size()), [&](int index) {
        visits[static_cast<std::size_t>(index)] += 1;
    });

    for (int visit : visits) {
        EXPECT_EQ(visit, 1);
    }
}

TEST_F(ParallelRuntime, RejectsNegativeThreadCount) {
    EXPECT_THROW(pgo::parallel::set_thread_count(-1), std::invalid_argument);
}

TEST_F(ParallelRuntime, StoresConfiguredThreadCount) {
    pgo::parallel::set_thread_count(0);
    EXPECT_EQ(pgo::parallel::configured_thread_count(), 0);

    pgo::parallel::set_thread_count(1);
    EXPECT_EQ(pgo::parallel::configured_thread_count(), 1);
}

TEST_F(ParallelRuntime, ReportsCompiledBackend) {
#if defined(PGO_ENABLE_TBB)
    EXPECT_TRUE(pgo::parallel::is_tbb_enabled());
#else
    EXPECT_FALSE(pgo::parallel::is_tbb_enabled());
#endif
}

TEST_F(ParallelFor, DefaultThreadCountProducesCorrectReduction) {
    pgo::parallel::set_thread_count(0);

    std::vector<int> values(4096, 0);
    pgo::parallel::parallel_for(0, static_cast<int>(values.size()), [&](int index) {
        values[static_cast<std::size_t>(index)] = index % 7;
    });

    const int sum = std::accumulate(values.begin(), values.end(), 0);
    int expected = 0;
    for (int index = 0; index < static_cast<int>(values.size()); ++index) {
        expected += index % 7;
    }
    EXPECT_EQ(sum, expected);
}

} // namespace
