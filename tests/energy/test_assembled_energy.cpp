#include "pgo/assembly/cpu_assembler.hpp"
#include "pgo/energy/assembled_energy.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace pgo::energy::test {

class ToyEdgeEnergyProvider {
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

    void local_hessian(const std::size_t, const pgo::math::DVec<double>&,
                       pgo::assembly::LocalMatrix<double>& H) const {
        H.resize(2, 2);
        H << 1.0, -1.0, -1.0, 1.0;
    }
};

static_assert(pgo::energy::FullEnergy<pgo::energy::AssembledEnergy<double, ToyEdgeEnergyProvider>, double>);

TEST(AssembledEnergy, MatchesCPUAssemblerOutputs) {
    const ToyEdgeEnergyProvider provider{};
    const pgo::energy::AssembledEnergy<double, ToyEdgeEnergyProvider> energy{provider};

    pgo::math::DVec<double> u{2};
    u << 2.0, -1.0;

    EXPECT_DOUBLE_EQ(pgo::assembly::assemble_value<double>(provider, u), energy.value(u));

    pgo::math::DVec<double> expected_gradient;
    pgo::math::DVec<double> gradient;
    pgo::assembly::assemble_gradient<double>(provider, u, expected_gradient);
    energy.gradient(u, gradient);
    ASSERT_EQ(expected_gradient.size(), gradient.size());
    EXPECT_TRUE(expected_gradient.isApprox(gradient));

    pgo::math::SparseMat<double> expected_hessian;
    pgo::math::SparseMat<double> hessian;
    pgo::assembly::assemble_hessian<double>(provider, u, expected_hessian);
    energy.hessian(u, hessian);
    EXPECT_TRUE(expected_hessian.isApprox(hessian));

    double expected_value = 0.0;
    double fused_value = 0.0;
    pgo::math::DVec<double> expected_fused_gradient;
    pgo::math::DVec<double> fused_gradient;
    pgo::math::SparseMat<double> expected_fused_hessian;
    pgo::math::SparseMat<double> fused_hessian;
    pgo::assembly::assemble_value_gradient_hessian<double>(provider, u, expected_value, expected_fused_gradient,
                                                           expected_fused_hessian);
    energy.value_gradient_hessian(u, fused_value, fused_gradient, fused_hessian);

    EXPECT_DOUBLE_EQ(expected_value, fused_value);
    EXPECT_TRUE(expected_fused_gradient.isApprox(fused_gradient));
    EXPECT_TRUE(expected_fused_hessian.isApprox(fused_hessian));
}

} // namespace pgo::energy::test
