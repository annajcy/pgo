#pragma once

#include "pgo/dof/reduced_dof_map.hpp"
#include "pgo/energy/energy_sum.hpp"
#include "pgo/energy/inertial_energy.hpp"
#include "pgo/energy/reduced_energy.hpp"
#include "pgo/integrator/dynamic_state.hpp"
#include "pgo/integrator/time_step_result.hpp"
#include "pgo/solver/newton_solver.hpp"

namespace pgo::integrator {

template <typename T>
class BackwardEuler {
public:
    template <typename PotentialEnergy>
        requires pgo::energy::FullEnergy<PotentialEnergy, T>
    TimeStepResult<T> step(const PotentialEnergy& potential_energy,
                           const pgo::math::DVec<T>& lumped_mass,
                           const pgo::dof::ReducedDofMap<T>& dof_map,
                           DynamicState<T>& state,
                           const T dt,
                           const pgo::solver::NewtonOptions<T>& options = {},
                           const bool commit_on_failure = false) const {

        const pgo::math::DVec<T> u_old = state.u;
        const pgo::math::DVec<T> v_old = state.v;
        const pgo::math::DVec<T> u_hat = u_old + dt * state.v;

        const pgo::energy::LumpedInertialEnergy<T> inertia{lumped_mass, u_hat, dt};
        const pgo::energy::EnergySumView<T, PotentialEnergy, pgo::energy::LumpedInertialEnergy<T>> step_energy{
            potential_energy, inertia};

        const pgo::energy::ReducedEnergyView<T, decltype(step_energy)> reduced{step_energy, dof_map};

        pgo::math::DVec<T> free_u = dof_map.restrict_vector_to_free(u_old);

        const pgo::solver::SolverResult<T> solver_result =
            pgo::solver::solve_newton(reduced, free_u, options);

        const bool converged = solver_result.status == pgo::solver::SolverStatus::converged;
        if (converged || commit_on_failure) {
            state.u = dof_map.scatter_solution(free_u);
            state.v = (state.u - u_old) / dt;
            state.a = (state.v - v_old) / dt;
        }

        return {solver_result.status, solver_result.iterations,
                solver_result.final_value, solver_result.final_gradient_norm, dt};
    }
};

} // namespace pgo::integrator
