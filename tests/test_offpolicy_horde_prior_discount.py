"""OffPolicyHordeLearner head-trace decay uses the prior call's discount.

``OffPolicyHordeLearner.update_with_ratios_and_discounts`` must decay head
eligibility traces with the PRIOR call's discount (the discount into the
state being processed), while the current call's discount only controls the
TD bootstrap. Decaying with the current discount wipes the episode trace on
a terminal transition (discount 0) before the terminal TD error credits it.

Setup: one linear demon, LMS step_size 1, rho 1, lamda 0.9.
Step 1: obs [1, 0], cumulant 1, discount 0.9 (non-terminal).
Step 2: obs [0, 1], cumulant 2, discount 0.0 (terminal).
Head weights start at zero, so step-1 TD error is 1 and step-2 TD error is
2 - 1 (bias) = 1. Prior-discount-correct step-2 head weights are
[1 + 0.9*0.9, 1] = [1.81, 1]; decaying with the current discount gives [1, 1].
"""

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPState,
    migrate_legacy_multi_head_mlp_state,
)
from alberta_framework.core.off_policy_horde import OffPolicyHordeLearner
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    MLPParams,
    create_horde_spec,
)

pytestmark = pytest.mark.unit


def _learner() -> OffPolicyHordeLearner:
    spec = create_horde_spec(
        [
            GVFSpec(
                name="d",
                demon_type=DemonType.PREDICTION,
                gamma=0.9,
                lamda=0.9,
                cumulant_index=0,
            )
        ]
    )
    return OffPolicyHordeLearner(
        spec,
        hidden_sizes=(),
        step_size=1.0,
        sparsity=0.0,
        ratio_clip=10.0,
        trace_ratio_clip=10.0,
    )


def test_terminal_transition_credits_prior_step_trace() -> None:
    learner = _learner()
    state = learner.init(2, jr.key(0))
    # Zero the head so TD errors are exact by construction.
    state = state.replace(
        head_params=MLPParams(
            weights=(jnp.zeros((1, 2), dtype=jnp.float32),),
            biases=(jnp.zeros((1,), dtype=jnp.float32),),
        )
    )

    step1 = learner.update_with_ratios_and_discounts(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([0.9], dtype=jnp.float32),
    )
    assert bool(step1.update_applied)
    np.testing.assert_allclose(
        np.asarray(step1.state.head_params.weights[0]), [[1.0, 0.0]], atol=1e-6
    )

    step2 = learner.update_with_ratios_and_discounts(
        step1.state,
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array([2.0], dtype=jnp.float32),
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
    )
    assert bool(step2.update_applied)
    # Terminal TD error (1.0) must credit the decayed step-1 trace (0.81),
    # not only the current gradient.
    np.testing.assert_allclose(
        np.asarray(step2.state.head_params.weights[0]), [[1.81, 1.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(step2.state.head_params.biases[0]), [2.81], atol=1e-6
    )


def _legacy_dict_without_new_fields() -> dict:
    """Build a pre-``previous_discounts`` legacy state mapping."""
    state = _learner().init(2, jr.key(0))
    fields = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(MultiHeadMLPState)
    }
    del fields["step_words"]
    del fields["previous_discounts"]
    return fields


def test_legacy_state_without_previous_discounts_migrates() -> None:
    migrated = migrate_legacy_multi_head_mlp_state(
        _legacy_dict_without_new_fields()
    )
    assert migrated.previous_discounts is None
    np.testing.assert_array_equal(
        np.asarray(migrated.step_words), np.array([0, 0], dtype=np.uint32)
    )


def test_legacy_state_with_unknown_field_still_rejected() -> None:
    fields = _legacy_dict_without_new_fields()
    fields["bogus_field"] = 0
    with pytest.raises(ValueError, match="not exact"):
        migrate_legacy_multi_head_mlp_state(fields)
