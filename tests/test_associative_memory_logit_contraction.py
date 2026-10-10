"""Scaled row logits must not depend on compiled FMA contraction.

``logit_scale * row`` is recomputed inside the cross-entropy max-subtraction
fusion and contracted into an FMA, so on an uncentered formulation the
difference at the maximum is the rounding error of that product instead of
exactly zero. That shifts the per-feature loss, and therefore the stored
utility, by a value the finiteness gate cannot see: the committed state is
finite and the update is accepted while holding a different number.

Coverage is two-layered because the failure is compilation-target specific.
The large-row cases discriminate only on a backend whose compiler contracts
the fused multiply (measured on arm64 macOS); the shift-invariance guards at
the end are deterministic consequences of exact arithmetic and discriminate
through ordinary rounding on x86 as well.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
)

pytestmark = pytest.mark.unit


def _config(**overrides: object) -> AssociativeMemoryConfig:
    values: dict[str, object] = {
        "vocab_size": 2,
        "block_size": 2,
        "suffix_length": 2,
        "feature_family": "position_token",
        "max_features": 4,
    }
    values.update(overrides)
    return AssociativeMemoryConfig(**values)  # type: ignore[arg-type]


def _two_writes(config: AssociativeMemoryConfig) -> tuple[np.ndarray, list[bool]]:
    learner = AssociativeMemoryLearner(config)
    state = learner.init()
    context = jnp.array([0, 1], dtype=jnp.int32)
    applied: list[bool] = []
    for _ in range(2):
        result = learner.update(state, context, jnp.asarray(0, dtype=jnp.int32))
        state = result.state
        applied.append(bool(result.update_applied))
    return np.asarray(state.utility, dtype=np.float64), applied


@pytest.mark.parametrize("logit_scale", [0.3, 0.7, 0.9, 0.99, 3.0])
def test_large_row_logits_keep_the_exact_utility(logit_scale: float) -> None:
    """A repeated identical write leaves every feature utility at zero."""
    config = _config(logit_scale=logit_scale, write_lr=1.0e12 / logit_scale)

    utility, applied = _two_writes(config)

    # Both rows predict the written label exactly, so every per-feature loss
    # equals the aggregate loss and no utility moves off zero.
    assert applied == [True, True]
    np.testing.assert_array_equal(utility, np.zeros_like(utility))


@pytest.mark.parametrize("logit_scale", [0.25, 0.5, 1.0, 4.0])
def test_power_of_two_scales_are_unchanged(logit_scale: float) -> None:
    config = _config(logit_scale=logit_scale, write_lr=1.0e12 / logit_scale)

    utility, applied = _two_writes(config)

    assert applied == [True, True]
    np.testing.assert_array_equal(utility, np.zeros_like(utility))


def test_compiled_and_eager_agree_on_the_committed_state() -> None:
    config = _config(logit_scale=0.7, write_lr=1.0e12 / 0.7)

    compiled, compiled_applied = _two_writes(config)
    with jax.disable_jit():
        eager, eager_applied = _two_writes(config)

    assert compiled_applied == eager_applied == [True, True]
    np.testing.assert_array_equal(compiled, eager)


def test_compiled_matches_eager_at_the_default_scale() -> None:
    """The ordinary power-of-two path keeps compiled and eager identical."""
    config = _config(logit_scale=4.0, write_lr=1.0)
    learner = AssociativeMemoryLearner(config)
    contexts = (
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([1, 0], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    labels = (0, 1, 0)

    def run() -> tuple[np.ndarray, np.ndarray]:
        state = learner.init()
        for context, label in zip(contexts, labels, strict=True):
            result = learner.update(state, context, jnp.asarray(label, dtype=jnp.int32))
            assert bool(result.update_applied)
            state = result.state
        return (
            np.asarray(state.utility, dtype=np.float64),
            np.asarray(state.values, dtype=np.float64),
        )

    compiled_utility, compiled_values = run()
    with jax.disable_jit():
        eager_utility, eager_values = run()

    np.testing.assert_array_equal(compiled_utility, eager_utility)
    np.testing.assert_array_equal(compiled_values, eager_values)
    assert np.all(np.isfinite(compiled_values))

def test_update_utilities_are_shift_invariant() -> None:
    """Opposite exact row shifts that cancel in aggregate leave identical bits.

    This guard is deterministic and platform-independent where the large-row
    cases above are not: it does not rely on the compiler contracting an FMA.
    The two row banks differ by exact opposite shifts of 0.125, so their sums
    and their max-centered representations are bit-identical, and cross
    entropy's shift invariance requires bit-identical committed utilities.
    The uncentered formulation breaks this through ordinary rounding of
    ``logit_scale * row``. Constants due to the review on the first head.
    """
    config = AssociativeMemoryConfig(
        vocab_size=3,
        block_size=2,
        suffix_length=2,
        feature_family="position_token",
        max_features=4,
        logit_scale=0.3,
        normalize_by_weight=True,
        utility_lr=1.0,
    )
    learner = AssociativeMemoryLearner(config)
    context = jnp.array([0, 1], dtype=jnp.int32)
    rows_a = jnp.array(
        [
            [0.27219006, 0.8811424, 1.8304602],
            [-0.62953675, 1.602477, -1.0598103],
        ],
        dtype=jnp.float32,
    )
    rows_b = jnp.array(
        [
            [0.39719006, 1.0061424, 1.9554602],
            [-0.75453675, 1.477477, -1.1848103],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(rows_a.sum(axis=0)).view(np.uint32),
        np.asarray(rows_b.sum(axis=0)).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(rows_a - rows_a.max(axis=-1, keepdims=True)).view(np.uint32),
        np.asarray(rows_b - rows_b.max(axis=-1, keepdims=True)).view(np.uint32),
    )

    def committed_utility(rows: jnp.ndarray) -> np.ndarray:
        state = learner.init()
        state = learner.update(state, context, jnp.asarray(0, dtype=jnp.int32)).state
        padded = jnp.zeros((4, 3), dtype=jnp.float32).at[:2].set(rows)
        state = state.replace(
            values=padded,
            prior=jnp.zeros_like(state.prior),
            utility=jnp.zeros_like(state.utility),
        )
        result = learner.update(state, context, jnp.asarray(1, dtype=jnp.int32))
        assert bool(result.update_applied)
        return np.asarray(result.state.utility)[:2]

    np.testing.assert_array_equal(
        committed_utility(rows_a).view(np.uint32),
        committed_utility(rows_b).view(np.uint32),
    )


def test_weighted_feature_loss_is_shift_invariant() -> None:
    """A uniform exact shift of one row must not move the loss by an ulp."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=3,
            block_size=2,
            suffix_length=2,
            feature_family="position_token",
            max_features=4,
            logit_scale=0.3,
            normalize_by_weight=True,
        )
    )
    row = jnp.array([-0.9721598, 0.76766425, 0.2550581], dtype=jnp.float32)
    weights = jnp.ones(1, dtype=jnp.float32)
    label = jnp.asarray(1, dtype=jnp.int32)

    base = learner._weighted_feature_loss(row[None, :], weights, label)
    shifted = learner._weighted_feature_loss((row + 1.0)[None, :], weights, label)

    assert np.asarray(base).view(np.uint32) == np.asarray(shifted).view(np.uint32)

