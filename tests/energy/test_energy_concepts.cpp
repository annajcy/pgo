#include "pgo/core/energy/energy_concepts.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace pgo::energy::test {

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

static_assert(pgo::energy::LocalEnergyProvider<ToyEdgeEnergy, double>);
static_assert(!pgo::energy::FusedLocalEnergyProvider<ToyEdgeEnergy, double>);

class QuadraticEnergy {
public:
    [[nodiscard]] double value(const pgo::math::DVec<double>& z) const {
        return 0.5 * z.squaredNorm();
    }

    void gradient(const pgo::math::DVec<double>& z, pgo::math::DVec<double>& gradient) const {
        gradient = z;
    }

    void hessian(const pgo::math::DVec<double>& z, pgo::math::SparseMat<double>& hessian) const {
        hessian.resize(z.size(), z.size());
        hessian.setIdentity();
    }

    void value_gradient_hessian(const pgo::math::DVec<double>& z, double& value,
                                pgo::math::DVec<double>& gradient,
                                pgo::math::SparseMat<double>& hessian) const {
        value = this->value(z);
        this->gradient(z, gradient);
        this->hessian(z, hessian);
    }
};

static_assert(pgo::energy::DifferentiableEnergy<QuadraticEnergy, double>);
static_assert(pgo::energy::FullEnergy<QuadraticEnergy, double>);

TEST(EnergyConcept, ToyLocalEnergyExposesLocalContributionSizes) {
    const ToyEdgeEnergy energy{};
    pgo::math::DVec<double> u{2};
    u << 2.0, -1.0;

    std::vector<std::size_t> dofs;
    pgo::assembly::LocalVector<double> gradient;
    pgo::assembly::LocalMatrix<double> hessian;

    energy.local_dofs(0, dofs);
    energy.local_gradient(0, u, gradient);
    energy.local_hessian(0, u, hessian);

    EXPECT_EQ(2, dofs.size());
    EXPECT_EQ(2, gradient.size());
    EXPECT_EQ(2, hessian.rows());
    EXPECT_EQ(2, hessian.cols());
}

} // namespace pgo::energy::test
