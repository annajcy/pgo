#pragma once

#include "pgo/log/level.hpp"
#include "pgo/log/sink.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace pgo::log {

class Logger {
public:
    Logger(std::shared_ptr<Sink> sink, Level threshold, std::string name)
        : m_sink{std::move(sink)},
          m_threshold{threshold},
          m_name{std::move(name)} {}

    [[nodiscard]] std::string_view name() const {
        return m_name;
    }

    void set_name(std::string name) { m_name = std::move(name); }
    void set_threshold(Level threshold) { m_threshold = threshold; }

    void log(Level level, std::string_view message) const {
        if (!m_sink) {
            return;
        }
        if (should_log(level, m_threshold)) {
            m_sink->log(level, m_name, message);
        }
    }

    void trace(std::string_view message) const { log(Level::trace, message); }
    void debug(std::string_view message) const { log(Level::debug, message); }
    void info(std::string_view message) const { log(Level::info, message); }
    void warn(std::string_view message) const { log(Level::warn, message); }
    void error(std::string_view message) const { log(Level::error, message); }

private:
    std::shared_ptr<Sink> m_sink;
    Level m_threshold;
    std::string m_name;
};

} // namespace pgo::log
