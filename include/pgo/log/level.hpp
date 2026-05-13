#pragma once

#include <string_view>

namespace pgo::log {

enum class Level {
    trace,
    debug,
    info,
    warn,
    error,
    off,
};

static_assert(static_cast<int>(Level::trace) < static_cast<int>(Level::error),
              "Level enum must be ordered from least to most severe");

[[nodiscard]] constexpr std::string_view level_name(Level level) {
    switch (level) {
    case Level::trace:
        return "trace";
    case Level::debug:
        return "debug";
    case Level::info:
        return "info";
    case Level::warn:
        return "warn";
    case Level::error:
        return "error";
    case Level::off:
        return "off";
    }
    return "unknown";
}

[[nodiscard]] constexpr bool should_log(Level message_level, Level threshold) {
    return threshold != Level::off && static_cast<int>(message_level) >= static_cast<int>(threshold);
}

} // namespace pgo::log
