#pragma once

#include "pgo/log/logger.hpp"

#include <memory>
#include <string_view>

namespace pgo::log {

class Registry {
    std::shared_ptr<Sink> m_sink;
    Level m_threshold;
    
public:
    Registry();
    explicit Registry(std::shared_ptr<Sink> sink, Level threshold = Level::info);

    // with default level (inherits Registry's m_threshold)
    [[nodiscard]] Logger root() const;
    [[nodiscard]] Logger get(std::string_view name) const;

    // with explicit level override
    [[nodiscard]] Logger root(Level level) const;
    [[nodiscard]] Logger get(std::string_view name, Level level) const;

    void set_sink(std::shared_ptr<Sink> sink);
    void set_level(Level threshold);
};

Registry& default_registry();

// with default level
[[nodiscard]] Logger root();
[[nodiscard]] Logger get(std::string_view name);

// with explicit level override
[[nodiscard]] Logger root(Level level);
[[nodiscard]] Logger get(std::string_view name, Level level);

void set_sink(std::shared_ptr<Sink> sink);
void set_level(Level level);

void trace(std::string_view message);
void debug(std::string_view message);
void info(std::string_view message);
void warn(std::string_view message);
void error(std::string_view message);

} // namespace pgo::log
