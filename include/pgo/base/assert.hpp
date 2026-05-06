#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace pgo::base {

inline void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

} // namespace pgo::base
