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

} // namespace pgo::log

#endif
