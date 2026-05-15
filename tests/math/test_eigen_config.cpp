#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace pgo::math::test {

namespace {

constexpr bool kPgoBackendNone = false
#if defined(PGO_EIGEN_ACCELERATION_NONE)
                                 || true
#endif
    ;

constexpr bool kPgoBackendMkl = false
#if defined(PGO_EIGEN_ACCELERATION_MKL)
                                || true
#endif
    ;

constexpr bool kPgoBackendAccelerate = false
#if defined(PGO_EIGEN_ACCELERATION_ACCELERATE)
                                       || true
#endif
    ;

constexpr bool kEigenUsesMkl = false
#if defined(EIGEN_USE_MKL_ALL)
                               || true
#endif
    ;

constexpr bool kEigenUsesBlas = false
#if defined(EIGEN_USE_BLAS)
                                || true
#endif
    ;

} // namespace

TEST(EigenConfig, DenseMatrixMultiplyWorks) {
    Eigen::Matrix2d a;
    a << 1.0, 2.0, 3.0, 4.0;

    const Eigen::Matrix2d b = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d c = a * b;

    EXPECT_DOUBLE_EQ(c(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 4.0);
}

TEST(EigenConfig, DeclaresExactlyOneAccelerationBackend) {
    constexpr int backend_count = 0
#if defined(PGO_EIGEN_ACCELERATION_NONE)
                                  + 1
#endif
#if defined(PGO_EIGEN_ACCELERATION_MKL)
                                  + 1
#endif
#if defined(PGO_EIGEN_ACCELERATION_ACCELERATE)
                                  + 1
#endif
        ;

    EXPECT_EQ(backend_count, 1);
}

TEST(EigenConfig, BackendDefinitionsMatchEigenDefinitions) {
    if constexpr (kPgoBackendNone) {
        EXPECT_FALSE(kEigenUsesMkl);
        EXPECT_FALSE(kEigenUsesBlas);
    } else if constexpr (kPgoBackendMkl) {
        EXPECT_TRUE(kEigenUsesMkl);
    } else if constexpr (kPgoBackendAccelerate) {
        EXPECT_FALSE(kEigenUsesMkl);
        EXPECT_TRUE(kEigenUsesBlas);
    } else {
        FAIL() << "No pgo Eigen acceleration backend was declared";
    }
}

TEST(EigenConfig, EnabledAutoBackendMatchesPlatformConvention) {
    if constexpr (!kPgoBackendNone) {
#if defined(__APPLE__)
        EXPECT_TRUE(kPgoBackendAccelerate);
        EXPECT_FALSE(kPgoBackendMkl);
#elif defined(_WIN32) || defined(__unix__)
        EXPECT_TRUE(kPgoBackendMkl);
        EXPECT_FALSE(kPgoBackendAccelerate);
#endif
    }
}

TEST(EigenConfig, EigenInternalParallelismIsDisabled) {
#if defined(EIGEN_DONT_PARALLELIZE)
    SUCCEED();
#else
    FAIL() << "EIGEN_DONT_PARALLELIZE should be defined by pgo::eigen_config";
#endif
}

TEST(EigenConfig, EigenAlignmentOverrideIsOptional) {
#if defined(EIGEN_MAX_ALIGN_BYTES)
    EXPECT_GT(EIGEN_MAX_ALIGN_BYTES, 0);
#else
    SUCCEED() << "default configuration keeps Eigen/platform alignment policy";
#endif
}

TEST(EigenConfig, MklNoDirectCallIsOptInAndScopedToMklBackend) {
#if defined(EIGEN_MKL_NO_DIRECT_CALL)
    #if defined(PGO_EIGEN_ACCELERATION_MKL)
        SUCCEED() << "MKL backend explicitly opted into EIGEN_MKL_NO_DIRECT_CALL";
    #else
        FAIL() << "EIGEN_MKL_NO_DIRECT_CALL must not leak outside the MKL backend";
    #endif
#else
    SUCCEED() << "default configuration keeps Eigen direct-MKL-call path enabled";
#endif
}

} // namespace pgo::math::test
