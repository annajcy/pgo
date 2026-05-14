#include "pgo/log/spdlog_sink.hpp"

#if defined(PGO_ENABLE_SPDLOG)

#    include <spdlog/sinks/basic_file_sink.h>
#    include <spdlog/sinks/rotating_file_sink.h>
#    include <spdlog/sinks/stdout_color_sinks.h>
#    include <spdlog/spdlog.h>

#    include <string>
#    include <utility>

namespace pgo::log {

namespace {

[[nodiscard]] spdlog::level::level_enum to_spdlog_level(Level level) {
    switch (level) {
    case Level::trace:
        return spdlog::level::trace;
    case Level::debug:
        return spdlog::level::debug;
    case Level::info:
        return spdlog::level::info;
    case Level::warn:
        return spdlog::level::warn;
    case Level::error:
        return spdlog::level::err;
    case Level::off:
        return spdlog::level::off;
    }
    return spdlog::level::info;
}

} // namespace

SpdlogSink::SpdlogSink(std::shared_ptr<spdlog::logger> logger)
    : m_logger{std::move(logger)} {}

void SpdlogSink::log(Level level, std::string_view logger_name, std::string_view message) {
    m_logger->log(to_spdlog_level(level), "[{}] {}", logger_name, message);
}

std::shared_ptr<Sink> make_default_spdlog_sink() {
    return std::make_shared<SpdlogSink>(spdlog::stderr_color_mt("pgo"));
}

std::shared_ptr<Sink> make_spdlog_file_sink(const std::string& path) {
    return std::make_shared<SpdlogSink>(spdlog::basic_logger_mt("pgo", path));
}

std::shared_ptr<Sink> make_spdlog_rotating_file_sink(
    const std::string& path, size_t max_size, size_t max_files) {
    return std::make_shared<SpdlogSink>(
        spdlog::rotating_logger_mt("pgo", path, max_size, max_files));
}

} // namespace pgo::log

#endif
