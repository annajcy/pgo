#include "pgo/log/stdio_sink.hpp"
#include "pgo/log/level.hpp"

#include <iostream>

namespace pgo::log {

StdIOSink::StdIOSink()
    : m_use_file{false} {}

StdIOSink::StdIOSink(const std::string& path)
    : m_file{path}, m_use_file{true} {}

void StdIOSink::log(Level level, std::string_view logger_name, std::string_view message) {
    std::scoped_lock lock{m_mutex};
    if (m_use_file) {
        m_file << '[' << level_name(level) << "] [" << logger_name << "] " << message << '\n';
    } else {
        auto& out = (level == Level::warn || level == Level::error) ? std::cerr : std::cout;
        out << '[' << level_name(level) << "] [" << logger_name << "] " << message << '\n';
    }
}

} // namespace pgo::log
