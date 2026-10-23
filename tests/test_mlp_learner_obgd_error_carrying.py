"""ObGD error-carrying bound: MLPLearner with a delta-returning optimizer.

Published ObGD (Elsayed et al. 2024) bounds error-free steps with
``M = alpha * kappa * max(|delta|, 1) * ||z||_1``. When the proposed steps
already carry the prediction error (``alpha * delta * z`` -- the case for
delta-returning optimizers such as Adam), the published-equivalent bound in
terms of those steps is ``kappa * sum|steps| / min(|delta|, 1)``, i.e. the
total applied L1 step is capped at ``min(|delta|, 1) / kappa``. Feeding a
pseudo-error of ``1.0`` to ``ObGDBounding.bound`` lets small-error updates
overshoot the published guarantee by up to ``1 / |delta|``.
"""

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.core.learners import MLPLearner
from alberta_framework.core.optimizers import ObGDBounding


def _total_applied_l1(old_state, new_state) -> float:
    total = 0.0
    for old_w, new_w in zip(old_state.params.weights, new_state.params.weights):
        total += float(jnp.sum(jnp.abs(new_w - old_w)))
    for old_b, new_b in zip(old_state.params.biases, new_state.params.biases):
        total += float(jnp.sum(jnp.abs(new_b - old_b)))
    return total


def test_obgd_bound_uses_error_carrying_form_with_delta_optimizer():
    kappa = 2.0
    learner = MLPLearner(
        hidden_sizes=(2,),
        optimizer=Adam(step_size=0.1),
        bounder=ObGDBounding(kappa=kappa),
        gamma=0.0,
        lamda=0.0,
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(1, jr.key(0))
    observation = jnp.array([10.0], dtype=jnp.float32)

    prediction = float(jnp.squeeze(learner.predict(state, observation)))
    # Small error: |delta| = 0.1, well under 1.0 so the published
    # error-carrying cap min(|delta|, 1) / kappa = 0.05 bites.
    target = jnp.array([prediction + 0.1], dtype=jnp.float32)

    result = learner.update(state, observation, target)
    assert bool(result.update_applied), "update must be applied for the bound check"

    error = float(jnp.abs(jnp.squeeze(result.error)))
    assert error == jnp.float32(0.1).item() or abs(error - 0.1) < 1e-6, (
        f"expected |delta| ~= 0.1, got {error}"
    )
    applied = _total_applied_l1(state, result.state)
    published_cap = min(error, 1.0) / kappa
    print(f"|delta|={error}, applied L1 step={applied:.6f}, published cap={published_cap:.6f}")
    assert applied <= published_cap * (1.0 + 1e-5), (
        f"ObGD overshoot: applied L1 step {applied:.6f} exceeds the published "
        f"error-carrying cap {published_cap:.6f}"
    )
