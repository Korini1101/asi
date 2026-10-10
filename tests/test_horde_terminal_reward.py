"""GVFSpec.terminal_reward must enter every Horde TD target.

Sutton et al. (2011) define the per-step GVF cumulant as
``C_t = c_t + z (1 - gamma_t)``. The one-step target is therefore
``c + gamma * V(s') + (1 - gamma) * z``. Default ``z = 0`` is unchanged.
"""

import chex
import jax.numpy as jnp
import jax.random as jr

from alberta_framework import (
    DemonType,
    GVFSpec,
    HordeLearner,
    HordeSpec,
    create_horde_spec,
)
from alberta_framework.core.independent_demon_horde import IndependentDemonHorde
from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    OffPolicyHordeLearner,
)


def _spec_with_z() -> HordeSpec:
    return create_horde_spec(
        [
            GVFSpec(
                name="z",
                demon_type=DemonType.PREDICTION,
                gamma=0.5,
                lamda=0.0,
                cumulant_index=0,
                terminal_reward=2.0,
            )
        ]
    )


def test_shared_horde_target_includes_terminal_reward() -> None:
    horde = HordeLearner(horde_spec=_spec_with_z(), hidden_sizes=(), sparsity=0.0)
    state = horde.init(3, jr.key(0))
    obs = jnp.ones(3)
    next_obs = jnp.ones(3) * 0.5
    cumulant = jnp.array([1.0])
    v_next = horde.predict(state, next_obs)

    result = horde.update(state, obs, cumulant, next_obs)
    chex.assert_trees_all_close(
        result.td_targets, cumulant + 0.5 * v_next + 0.5 * 2.0, atol=1e-6
    )

    terminal = horde.update_with_discounts(
        state, obs, cumulant, next_obs, jnp.array([0.0])
    )
    chex.assert_trees_all_close(terminal.td_targets, cumulant + 2.0, atol=1e-6)


def test_independent_demon_target_includes_terminal_reward() -> None:
    horde = IndependentDemonHorde(horde_spec=_spec_with_z(), hidden_sizes=(), sparsity=0.0)
    state = horde.init(3, jr.key(1))
    obs = jnp.ones(3)
    next_obs = jnp.ones(3) * 0.5
    cumulant = jnp.array([1.0])
    v_next = horde.predict(state, next_obs)

    result = horde.update(state, obs, cumulant, next_obs)
    chex.assert_trees_all_close(
        result.td_targets, cumulant + 0.5 * v_next + 0.5 * 2.0, atol=1e-6
    )


def test_off_policy_targets_include_terminal_reward() -> None:
    spec = _spec_with_z()
    obs = jnp.ones(3)
    next_obs = jnp.ones(3) * 0.5
    cumulant = jnp.array([1.0])
    rho = jnp.array([1.0])

    learner = OffPolicyHordeLearner(spec, hidden_sizes=(), ratio_clip=2.0)
    state = learner.init(3, jr.key(2))
    v_next = learner.predict(state, next_obs)
    result = learner.update_with_ratios(state, obs, cumulant, next_obs, rho)
    chex.assert_trees_all_close(
        result.td_targets, cumulant + 0.5 * v_next + 0.5 * 2.0, atol=1e-6
    )
    terminal = learner.update_with_ratios_and_discounts(
        state, obs, cumulant, next_obs, rho, jnp.array([0.0])
    )
    chex.assert_trees_all_close(terminal.td_targets, cumulant + 2.0, atol=1e-6)

    shared = NonlinearSharedGTDHordeLearner(
        spec, hidden_size=4, primary_step_size=0.002, secondary_step_size=1e-5
    )
    shared_state = shared.init(3, jr.key(3))
    shared_terminal = shared.update_with_ratios_and_discounts(
        shared_state, obs, cumulant, next_obs, rho, jnp.array([0.0])
    )
    chex.assert_trees_all_close(shared_terminal.td_targets, cumulant + 2.0, atol=1e-6)
