#include "pgo/base/assert.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace pgo::base::test {

TEST(BaseRequire, DoesNothingWhenConditionIsTrue) {
    EXPECT_NO_THROW(pgo::base::require(true, "should not throw"));
}

TEST(BaseRequire, ThrowsRuntimeErrorWithMessageWhenConditionIsFalse) {
    try {
        pgo::base::require(false, "missing rest mesh position");
        FAIL() << "Expected pgo::base::require to throw";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string{"missing rest mesh position"}, error.what());
    }
}

} // namespace pgo::base::test
