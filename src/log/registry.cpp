#include "pgo/log/registry.hpp"
#include "pgo/log/stderr_sink.hpp"

#include <utility>

namespace pgo::log {

Registry::Registry()
    : Registry{std::make_shared<StderrSink>(), Level::info} {}

Registry::Registry(std::shared_ptr<Sink> sink, Level threshold)
    : m_sink{std::move(sink)},
      m_threshold{threshold} {}

Logger Registry::root() const {
    return get("pgo");
}

Logger Registry::get(std::string_view name) const {
    return Logger{m_sink, m_threshold, std::string{name}};
}

Logger Registry::root(Level level) const {
    return get("pgo", level);
}

Logger Registry::get(std::string_view name, Level level) const {
    return Logger{m_sink, level, std::string{name}};
}

void Registry::set_sink(std::shared_ptr<Sink> sink) {
    m_sink = std::move(sink);
}

void Registry::set_level(Level threshold) {
    m_threshold = threshold;
}

Registry& default_registry() {
    static Registry registry;
    return registry;
}

Logger root() {
    return default_registry().root();
}

Logger get(std::string_view name) {
    return default_registry().get(name);
}

Logger root(Level level) {
    return default_registry().root(level);
}

Logger get(std::string_view name, Level level) {
    return default_registry().get(name, level);
}

void set_sink(std::shared_ptr<Sink> sink) {
    default_registry().set_sink(std::move(sink));
}

void set_level(Level level) {
    default_registry().set_level(level);
}

void trace(std::string_view message) { root(Level::trace).trace(message); }
void debug(std::string_view message) { root(Level::debug).debug(message); }
void info(std::string_view message) { root(Level::info).info(message); }
void warn(std::string_view message) { root(Level::warn).warn(message); }
void error(std::string_view message) { root(Level::error).error(message); }

} // namespace pgo::log
