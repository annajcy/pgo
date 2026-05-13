#pragma once

#include "pgo/log/sink.hpp"

namespace pgo::log {

class NullSink final : public Sink {
public:
    void log(Level, std::string_view, std::string_view) override {}
};

} // namespace pgo::log
