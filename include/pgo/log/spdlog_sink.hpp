#pragma once

#if defined(PGO_ENABLE_SPDLOG)

#    include "pgo/log/sink.hpp"

#    include <memory>

namespace spdlog {
class logger;
}

namespace pgo::log {

class SpdlogSink final : public Sink {
public:
    explicit SpdlogSink(std::shared_ptr<spdlog::logger> logger);

    void log(Level level, std::string_view logger_name, std::string_view message) override;

private:
    std::shared_ptr<spdlog::logger> m_logger;
};

std::shared_ptr<Sink> make_default_spdlog_sink();
std::shared_ptr<Sink> make_spdlog_file_sink(const std::string& path);
std::shared_ptr<Sink> make_spdlog_rotating_file_sink(
    const std::string& path, size_t max_size, size_t max_files);

} // namespace pgo::log

#endif
