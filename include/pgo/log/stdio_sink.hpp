#pragma once

#include "pgo/log/sink.hpp"

#include <fstream>
#include <mutex>
#include <string>

namespace pgo::log {

class StdIOSink final : public Sink {
    std::mutex m_mutex;
    std::ofstream m_file;
    bool m_use_file;

public:
    StdIOSink();
    explicit StdIOSink(const std::string& path);
    void log(Level level, std::string_view logger_name, std::string_view message) override;
};

} // namespace pgo::log
