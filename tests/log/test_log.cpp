#include "pgo/log/null_sink.hpp"
#include "pgo/log/registry.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

namespace pgo::log::test {

class CaptureSink final : public Sink {
public:
    struct Entry {
        Level level;
        std::string name;
        std::string message;
    };

    void log(Level level, std::string_view name, std::string_view message) override {
        entries.push_back({level, std::string{name}, std::string{message}});
    }

    std::vector<Entry> entries;
};

TEST(LogLevel, NamesAndThresholdsMatchExpectedOrdering) {
    EXPECT_EQ("trace", level_name(Level::trace));
    EXPECT_EQ("debug", level_name(Level::debug));
    EXPECT_EQ("info", level_name(Level::info));
    EXPECT_EQ("warn", level_name(Level::warn));
    EXPECT_EQ("error", level_name(Level::error));
    EXPECT_EQ("off", level_name(Level::off));

    EXPECT_TRUE(should_log(Level::warn, Level::info));
    EXPECT_FALSE(should_log(Level::debug, Level::info));
    EXPECT_FALSE(should_log(Level::error, Level::off));
}

TEST(Registry, GetReturnsNamedLoggerAndForwardsMessagesToSink) {
    auto sink = std::make_shared<CaptureSink>();
    Registry registry{sink};

    const Logger logger = registry.get("pgo.test", Level::trace);
    EXPECT_EQ("pgo.test", logger.name());

    logger.info("hello");

    ASSERT_EQ(1, sink->entries.size());
    EXPECT_EQ(Level::info, sink->entries[0].level);
    EXPECT_EQ("pgo.test", sink->entries[0].name);
    EXPECT_EQ("hello", sink->entries[0].message);
}

TEST(Registry, LevelFiltersMessagesBasedOnCallSiteLevel) {
    auto sink = std::make_shared<CaptureSink>();
    Registry registry{sink};
    const Logger logger = registry.get("pgo.filter", Level::warn);

    logger.info("hidden");
    logger.warn("shown");
    logger.error("also shown");

    ASSERT_EQ(2, sink->entries.size());
    EXPECT_EQ(Level::warn, sink->entries[0].level);
    EXPECT_EQ(Level::error, sink->entries[1].level);
}

TEST(Registry, RootLoggerNameIsPgo) {
    auto sink = std::make_shared<CaptureSink>();
    Registry registry{sink};

    registry.root(Level::trace).debug("root message");

    ASSERT_EQ(1, sink->entries.size());
    EXPECT_EQ("pgo", sink->entries[0].name);
}

// Convenience free functions (set_sink, get, info, ...) are trivial wrappers
// around default_registry(). Their forwarding logic is already covered by the
// Registry member-function tests above; this test only verifies that the
// singleton is accessible and the free functions don't crash.
TEST(Registry, DefaultRegistryConvenienceApiIsAccessible) {
    EXPECT_NO_THROW(info("smoke test"));
}

TEST(NullSink, DropsMessagesWithoutThrowing) {
    Registry registry{std::make_shared<NullSink>()};
    EXPECT_NO_THROW(registry.get("pgo.null", Level::trace).error("ignored"));
}

} // namespace pgo::log::test
