#include "pgo/core/assembly/cpu_assembler.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace pgo::assembly::test {

class ToyEdgeEnergy {
public:
    [[nodiscard]] std::size_t local_count() const {
        return 1;
    }

    void local_dofs(const std::size_t, std::vector<std::size_t>& dofs) const {
        dofs = {0, 1};
    }

    [[nodiscard]] double local_value(const std::size_t, const pgo::math::DVec<double>& u) const {
        return 0.5 * (u[0] - u[1]) * (u[0] - u[1]);
    }

    void local_gradient(const std::size_t, const pgo::math::DVec<double>& u,
                        pgo::assembly::LocalVector<double>& g) const {
        g.resize(2);
        g[0] = u[0] - u[1];
        g[1] = u[1] - u[0];
    }

    void local_hessian(const std::size_t, const pgo::math::DVec<double>&, pgo::assembly::LocalMatrix<double>& H) const {
        H.resize(2, 2);
        H << 1.0, -1.0, -1.0, 1.0;
    }
};

class SharedDofEnergy {
public:
    [[nodiscard]] std::size_t local_count() const {
        return 2;
    }

    void local_dofs(const std::size_t local_id, std::vector<std::size_t>& dofs) const {
        if (local_id == 0) {
            dofs = {0, 1, 2, 3};
        } else {
            dofs = {2, 3, 4, 5};
        }
    }

    [[nodiscard]] double local_value(const std::size_t, const pgo::math::DVec<double>&) const {
        return 0.0;
    }

    void local_gradient(const std::size_t, const pgo::math::DVec<double>&,
                        pgo::assembly::LocalVector<double>& g) const {
        g.resize(4);
        g.setOnes();
    }

    void local_hessian(const std::size_t, const pgo::math::DVec<double>&, pgo::assembly::LocalMatrix<double>& H) const {
        H.resize(4, 4);
        H.setZero();
    }
};

class BadGradientEnergy : public ToyEdgeEnergy {
public:
    void local_gradient(const std::size_t, const pgo::math::DVec<double>&,
                        pgo::assembly::LocalVector<double>& g) const {
        g.resize(1);
        g.setZero();
    }
};

class BadHessianEnergy : public ToyEdgeEnergy {
public:
    void local_hessian(const std::size_t, const pgo::math::DVec<double>&, pgo::assembly::LocalMatrix<double>& H) const {
        H.resize(2, 1);
        H.setZero();
    }
};

class BadDofEnergy : public ToyEdgeEnergy {
public:
    void local_dofs(const std::size_t, std::vector<std::size_t>& dofs) const {
        dofs = {0, 2};
    }
};

TEST(CPUAssembler, AssemblesToyEdgeValueGradientAndHessian) {
    const ToyEdgeEnergy energy{};
    pgo::math::DVec<double> u{2};
    u << 2.0, -1.0;

    EXPECT_DOUBLE_EQ(4.5, pgo::assembly::assemble_value<double>(energy, u));

    pgo::math::DVec<double> gradient;
    pgo::assembly::assemble_gradient<double>(energy, u, gradient);
    ASSERT_EQ(2, gradient.size());
    EXPECT_DOUBLE_EQ(3.0, gradient[0]);
    EXPECT_DOUBLE_EQ(-3.0, gradient[1]);

    pgo::math::SparseMat<double> hessian;
    pgo::assembly::assemble_hessian<double>(energy, u, hessian);
    EXPECT_DOUBLE_EQ(1.0, hessian.coeff(0, 0));
    EXPECT_DOUBLE_EQ(-1.0, hessian.coeff(0, 1));
    EXPECT_DOUBLE_EQ(-1.0, hessian.coeff(1, 0));
    EXPECT_DOUBLE_EQ(1.0, hessian.coeff(1, 1));

    double fused_value = 0.0;
    pgo::math::DVec<double> fused_gradient;
    pgo::math::SparseMat<double> fused_hessian;
    pgo::assembly::assemble_value_gradient_hessian<double>(energy, u, fused_value, fused_gradient, fused_hessian);
    EXPECT_DOUBLE_EQ(4.5, fused_value);
    EXPECT_DOUBLE_EQ(3.0, fused_gradient[0]);
    EXPECT_DOUBLE_EQ(-1.0, fused_hessian.coeff(0, 1));
}

TEST(CPUAssembler, AccumulatesSharedDofGradientContributions) {
    const SharedDofEnergy energy{};
    pgo::math::DVec<double> u{6};
    u.setZero();

    pgo::math::DVec<double> gradient;
    pgo::assembly::assemble_gradient<double>(energy, u, gradient);

    ASSERT_EQ(6, gradient.size());
    EXPECT_DOUBLE_EQ(1.0, gradient[0]);
    EXPECT_DOUBLE_EQ(1.0, gradient[1]);
    EXPECT_DOUBLE_EQ(2.0, gradient[2]);
    EXPECT_DOUBLE_EQ(2.0, gradient[3]);
    EXPECT_DOUBLE_EQ(1.0, gradient[4]);
    EXPECT_DOUBLE_EQ(1.0, gradient[5]);
}

TEST(CPUAssembler, RejectsBadLocalContributions) {
    pgo::math::DVec<double> u{2};
    u.setZero();

    pgo::math::DVec<double> gradient;
    pgo::math::SparseMat<double> hessian;

    EXPECT_THROW(pgo::assembly::assemble_gradient<double>(BadGradientEnergy{}, u, gradient), std::runtime_error);
    EXPECT_THROW(pgo::assembly::assemble_hessian<double>(BadHessianEnergy{}, u, hessian), std::runtime_error);
    EXPECT_THROW(pgo::assembly::assemble_gradient<double>(BadDofEnergy{}, u, gradient), std::runtime_error);
}

} // namespace pgo::assembly::test
