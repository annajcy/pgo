#pragma once

#include "pgo/log/sink.hpp"

#include <mutex>

namespace pgo::log {

class StderrSink final : public Sink {
    std::mutex m_mutex;
public:
    void log(Level level, std::string_view logger_name, std::string_view message) override;
};

} // namespace pgo::log
