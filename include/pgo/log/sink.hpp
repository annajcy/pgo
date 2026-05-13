#pragma once

#include "pgo/log/level.hpp"

#include <string_view>

namespace pgo::log {

class Sink {
public:
    virtual ~Sink() = default;
    virtual void log(Level level, std::string_view logger_name, std::string_view message) = 0;
};

} // namespace pgo::log
