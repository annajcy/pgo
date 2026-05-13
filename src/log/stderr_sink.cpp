#include "pgo/log/stderr_sink.hpp"

#include <iostream>

namespace pgo::log {

void StderrSink::log(Level level, std::string_view logger_name, std::string_view message) {
    std::scoped_lock lock{m_mutex};
    std::cerr << '[' << level_name(level) << "] [" << logger_name << "] " << message << '\n';
}

} // namespace pgo::log
