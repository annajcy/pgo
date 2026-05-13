#if defined(PGO_ENABLE_SPDLOG)

#    include "pgo/log/registry.hpp"
#    include "pgo/log/spdlog_sink.hpp"

#    include <gtest/gtest.h>

namespace pgo::log::test {

TEST(SpdlogSink, MakeDefaultSpdlogSinkReturnsNonNull) {
    auto sink = make_default_spdlog_sink();
    EXPECT_NE(sink, nullptr);
}

TEST(SpdlogSink, LogMessagesAreForwardedWithoutCrash) {
    auto sink = make_default_spdlog_sink();
    Registry registry{sink};

    const Logger logger = registry.get("pgo.spdlog", Level::trace);
    EXPECT_NO_THROW(logger.info("hello from spdlog"));
    EXPECT_NO_THROW(logger.warn("warning from spdlog"));
    EXPECT_NO_THROW(logger.error("error from spdlog"));
}

} // namespace pgo::log::test

#endif
