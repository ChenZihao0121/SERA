"""SERA: frozen historical success estimates and exact-budget rollout allocation.

The core is independent of Ray and inference backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class SERAAllocation:
    raw_success_prob: np.ndarray
    allocation_success_prob: np.ndarray
    ideal_rollouts: np.ndarray
    continuous_rollouts: np.ndarray
    rollouts: np.ndarray
    all_failure_prob: np.ndarray
    coverage_prob: np.ndarray
    centered_failure_prob: np.ndarray
    fidelity_prob: np.ndarray
    prompt_weights: np.ndarray
    waterline: float
    ideal_waterline: float
    rho_hat: float
    kappa_hat: float
    rollout_count_normalized: bool
    rho_normalized: bool
    strict_equalization_continuous_feasible: bool
    strict_equalization_integer_feasible: bool
    lower_bound_violation_fraction: float
    upper_bound_violation_fraction: float
    integer_distance_max: float
    all_failure_spread: float
    all_failure_log_spread: float
    centered_failure_spread: float
    centered_failure_log_spread: float
    active_prompt_mask: np.ndarray | None = None
    activation_score: np.ndarray | None = None
    activation_realized_signal: np.ndarray | None = None
    activation_threshold: float = float("nan")
    recoverability_selected_k: int = -1
    recoverability_k_min: int = -1
    recoverability_k_max: int = -1
    recoverability_candidate_count: int = 0
    recoverability_reference_rollouts: float = float("nan")
    recoverability_expected_mixed_group_count: float = float("nan")


class HistoricalSuccessEstimator:
    """Frozen-per-epoch discounted Beta estimator for sparse prompt keys.

    The state stores evidence *around* a fixed Beta prior. At the next epoch
    boundary, old evidence is multiplied by ``discount`` and the just-finished
    epoch's outcomes are added once::

        alpha_q <- prior_alpha + discount * (alpha_q - prior_alpha) + K_q
        beta_q  <- prior_beta  + discount * (beta_q  - prior_beta)  + N_q-K_q

    Allocations within an epoch read only the frozen posterior mean. Current-
    epoch outcomes remain pending, so there is no within-epoch information
    leakage. This implements the historical estimates in Appendix C.1.
    """

    def __init__(
        self,
        *,
        prior_alpha: float = 0.5,
        prior_beta: float = 0.5,
        discount: float = 0.75,
    ) -> None:
        self.prior_alpha = _validate_positive_real(
            "historical_prior_alpha", prior_alpha
        )
        self.prior_beta = _validate_positive_real(
            "historical_prior_beta", prior_beta
        )
        if isinstance(discount, bool) or not isinstance(discount, Real):
            raise ValueError("historical_discount must lie in [0, 1]")
        discount = float(discount)
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("historical_discount must lie in [0, 1]")
        self.discount = discount
        self.active_epoch = -1
        self._decayed_successes: dict[str, float] = {}
        self._decayed_failures: dict[str, float] = {}
        self._pending_successes: dict[str, int] = {}
        self._pending_failures: dict[str, int] = {}
        self._frozen_estimates: dict[str, float] = {}
        self._frozen_effective_counts: dict[str, float] = {}

    def start_epoch(self) -> int:
        """Promote pending outcomes and freeze the next posterior snapshot."""
        if self.active_epoch >= 0:
            prompt_keys = (
                set(self._decayed_successes)
                | set(self._decayed_failures)
                | set(self._pending_successes)
                | set(self._pending_failures)
            )
            decayed_successes: dict[str, float] = {}
            decayed_failures: dict[str, float] = {}
            frozen_estimates: dict[str, float] = {}
            frozen_effective_counts: dict[str, float] = {}
            for key in prompt_keys:
                successes = (
                    self.discount * self._decayed_successes.get(key, 0.0)
                    + self._pending_successes.get(key, 0)
                )
                failures = (
                    self.discount * self._decayed_failures.get(key, 0.0)
                    + self._pending_failures.get(key, 0)
                )
                effective_count = successes + failures
                posterior_mean = (
                    self.prior_alpha + successes
                ) / (
                    self.prior_alpha
                    + self.prior_beta
                    + effective_count
                )
                decayed_successes[key] = float(successes)
                decayed_failures[key] = float(failures)
                frozen_estimates[key] = float(posterior_mean)
                frozen_effective_counts[key] = float(effective_count)
            self._decayed_successes = decayed_successes
            self._decayed_failures = decayed_failures
            self._frozen_estimates = frozen_estimates
            self._frozen_effective_counts = frozen_effective_counts
            self._pending_successes = {}
            self._pending_failures = {}
        self.active_epoch += 1
        return self.active_epoch

    def update(self, prompt_keys: np.ndarray, outcomes: np.ndarray) -> None:
        """Stage current-epoch binary outcomes for the next epoch."""
        if self.active_epoch < 0:
            raise RuntimeError("start_epoch must be called before historical success-probability estimator updates")
        keys = np.asarray(prompt_keys, dtype=object)
        values = np.asarray(outcomes)
        if keys.ndim != 1 or values.ndim != 1 or keys.size != values.size:
            raise ValueError(
                "prompt_keys and outcomes must be aligned one-dimensional arrays"
            )
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("historical success-probability estimator outcomes must be binary")
        for prompt_key, outcome in zip(keys, values):
            key = str(prompt_key)
            if int(outcome) == 1:
                self._pending_successes[key] = (
                    self._pending_successes.get(key, 0) + 1
                )
            else:
                self._pending_failures[key] = (
                    self._pending_failures.get(key, 0) + 1
                )

    def lookup(
        self,
        prompt_keys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return frozen posterior means and discounted effective counts."""
        keys = np.asarray(prompt_keys, dtype=object)
        if keys.ndim != 1:
            raise ValueError("prompt_keys must be one-dimensional")
        estimates = np.array(
            [self._frozen_estimates.get(str(key), np.nan) for key in keys],
            dtype=np.float64,
        )
        effective_counts = np.array(
            [
                self._frozen_effective_counts.get(str(key), 0.0)
                for key in keys
            ],
            dtype=np.float64,
        )
        return estimates.copy(), estimates, effective_counts

    def state_dict(self) -> dict:
        """Return frozen and pending state for exact checkpoint resumption."""
        return {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "discount": self.discount,
            "active_epoch": self.active_epoch,
            "decayed_successes": dict(self._decayed_successes),
            "decayed_failures": dict(self._decayed_failures),
            "pending_successes": dict(self._pending_successes),
            "pending_failures": dict(self._pending_failures),
            "frozen_estimates": dict(self._frozen_estimates),
            "frozen_effective_counts": dict(self._frozen_effective_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state while rejecting a changed prior or discount."""
        for name in ("prior_alpha", "prior_beta", "discount"):
            if not np.isclose(
                float(state[name]),
                float(getattr(self, name)),
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError(f"historical success-probability estimator state {name} does not match")

        def _float_dict(name: str) -> dict[str, float]:
            return {
                str(key): float(value)
                for key, value in state.get(name, {}).items()
            }

        def _int_dict(name: str) -> dict[str, int]:
            values = {
                str(key): int(value)
                for key, value in state.get(name, {}).items()
            }
            if any(value < 0 for value in values.values()):
                raise ValueError(f"historical success-probability estimator state {name} contains a negative count")
            return values

        self.active_epoch = int(state.get("active_epoch", -1))
        self._decayed_successes = _float_dict("decayed_successes")
        self._decayed_failures = _float_dict("decayed_failures")
        self._pending_successes = _int_dict("pending_successes")
        self._pending_failures = _int_dict("pending_failures")
        self._frozen_estimates = _float_dict("frozen_estimates")
        self._frozen_effective_counts = _float_dict(
            "frozen_effective_counts"
        )
        if any(
            value < 0.0
            for values in (
                self._decayed_successes,
                self._decayed_failures,
                self._frozen_effective_counts,
            )
            for value in values.values()
        ):
            raise ValueError("historical success-probability estimator state contains negative evidence")


# Compatibility for older public launchers and imports. Both names use the
# same frozen discounted-Beta estimator and checkpoint representation.
AdaEMAPQEstimator = HistoricalSuccessEstimator


def _validate_positive_int(name: str, value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_positive_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def ragged_repeat_indices(repeats: np.ndarray) -> np.ndarray:
    """Return source-row indices for an interleaved ragged repeat."""
    repeats = np.asarray(repeats)
    if repeats.ndim != 1 or not np.issubdtype(repeats.dtype, np.integer):
        raise ValueError("repeats must be a one-dimensional integer array")
    if np.any(repeats < 0):
        raise ValueError("repeats must be non-negative")
    return np.repeat(np.arange(repeats.size, dtype=np.int64), repeats.astype(np.int64))


def _continuous_waterline(
    success_prob: np.ndarray,
    budget: int,
    n_min: int,
    n_max: int,
    *,
    max_iterations: int = 100,
) -> tuple[float, float, np.ndarray]:
    """Solve the bounded continuous equal-fidelity allocation.

    Centered MaxRL scales the per-prompt population gradient by
    ``kappa_q(N_q) = 1 - (1 - p_q)**(N_q - 1)``. Equalizing that coefficient
    therefore equalizes ``(1-p_q)**(N_q-1)`` and gives the unconstrained law

    ``N_q = 1 + tau / -log(1-p_q)``.

    The bisection is performed in log-``tau`` space so very small ImageNet
    probabilities remain representable.
    """
    failure_rate = -np.log1p(-success_prob)
    if not np.all(np.isfinite(failure_rate)) or np.any(failure_rate <= 0.0):
        raise ValueError("success_prob produced invalid failure rates")

    num_prompts = int(success_prob.size)
    minimum_budget = num_prompts * n_min
    maximum_budget = num_prompts * n_max
    if budget < minimum_budget or budget > maximum_budget:
        raise ValueError("continuous rollout budget is infeasible under bounds")
    if budget == minimum_budget:
        continuous = np.full(success_prob.shape, float(n_min), dtype=np.float64)
        centered_failure = np.exp(-failure_rate * (n_min - 1))
        return (
            float(np.max(centered_failure)),
            float(np.min(1.0 - centered_failure)),
            continuous,
        )

    effective_min = n_min - 1
    effective_max = n_max - 1
    if effective_max <= 0:
        raise ValueError("positive residual budget requires n_max >= 2")
    if effective_min > 0:
        tau_low = float(effective_min * np.min(failure_rate))
    else:
        tau_low = float(np.nextafter(0.0, 1.0))
    tau_high = float(effective_max * np.max(failure_rate))
    log_tau_low = float(np.log(tau_low))
    log_tau_high = float(np.log(tau_high))

    for _ in range(max_iterations):
        log_tau = 0.5 * (log_tau_low + log_tau_high)
        tau = float(np.exp(log_tau))
        continuous = np.clip(
            1.0 + tau / failure_rate,
            n_min,
            n_max,
        )
        if float(np.sum(continuous)) > budget:
            log_tau_high = log_tau
        else:
            log_tau_low = log_tau

    tau = float(np.exp(log_tau_low))
    continuous = np.clip(1.0 + tau / failure_rate, n_min, n_max)
    centered_failure_waterline = float(np.exp(-tau))
    target_fidelity = float(-np.expm1(-tau))
    return centered_failure_waterline, target_fidelity, continuous


def _centered_continuous_rounding_rollouts(
    success_prob: np.ndarray,
    budget: int,
    n_min: int,
    n_max: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Round the centered waterline by marginal squared-fidelity error."""
    waterline, target_fidelity, continuous = _continuous_waterline(
        success_prob,
        budget,
        n_min,
        n_max,
    )
    log_failure = np.log1p(-success_prob)
    rollouts = np.floor(continuous).astype(np.int64)
    rollouts = np.clip(rollouts, n_min, n_max)
    remaining = int(budget - np.sum(rollouts))
    if remaining < 0:
        raise RuntimeError(
            f"continuous rounding exceeded rollout budget by {-remaining}"
        )
    if remaining:
        candidates = np.flatnonzero(rollouts < n_max)
        if remaining > candidates.size:
            raise RuntimeError(
                "not enough prompts below n_max to complete integer rounding"
            )
        candidate_log_failure = log_failure[candidates]
        current_fidelity = -np.expm1(
            candidate_log_failure * (rollouts[candidates] - 1)
        )
        next_fidelity = -np.expm1(
            candidate_log_failure * rollouts[candidates]
        )
        current_error = (current_fidelity - target_fidelity) ** 2
        next_error = (next_fidelity - target_fidelity) ** 2
        marginal_error = next_error - current_error
        order = np.lexsort((candidates, marginal_error))
        rollouts[candidates[order[:remaining]]] += 1
    return waterline, continuous, rollouts


def _harmonic_weights(success_prob: np.ndarray) -> np.ndarray:
    """Return normalized inverse ``-log(1-p)`` weights without overflow."""
    failure_rate = -np.log1p(-success_prob)
    if not np.all(np.isfinite(failure_rate)) or np.any(failure_rate <= 0.0):
        raise ValueError("success_prob produced invalid failure rates")

    log_weights = -np.log(failure_rate)
    log_weights -= np.max(log_weights)
    scaled_weights = np.exp(log_weights)
    weight_sum = float(np.sum(scaled_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise RuntimeError("harmonic allocation produced invalid normalized weights")
    return scaled_weights / weight_sum


def _mixed_group_probability(
    success_prob: np.ndarray,
    rollouts: np.ndarray,
) -> np.ndarray:
    """Return P(0 < successes < n) for aligned probabilities/counts."""
    probability = np.asarray(success_prob, dtype=np.float64)
    counts = np.asarray(rollouts, dtype=np.int64)
    mixed = np.zeros(probability.shape, dtype=np.float64)
    positive = counts > 0
    mixed[positive] = (
        1.0
        - np.power(1.0 - probability[positive], counts[positive])
        - np.power(probability[positive], counts[positive])
    )
    np.clip(mixed, 0.0, 1.0, out=mixed)
    return mixed


def _dynamic_active_equalized_rollouts(
    success_prob: np.ndarray,
    budget: int,
    n_min: int,
    n_max: int,
    *,
    activation_threshold: float,
    inactive_floor: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Activate prompts with useful mixed-signal potential, then equalize.

    The activation score is ``u_q(N_max)``, the maximum mixed-group signal
    attainable under the per-prompt cap. Among individually eligible prompts,
    binary search returns the largest passing prefix it encounters whose
    *actual allocated* rollout counts satisfy the threshold. Integer rounding
    can break monotonicity, so this need not be the largest passing prefix
    over all sizes. Active
    prompts use the historical continuous floor-and-refill rule, with only its
    closed-form Nq target corrected to centered-MaxRL fidelity. Inactive prompts
    receive either zero or ``n_min`` *total* rollouts.  In floor mode those
    samples are the first-stage samples for the batch; they are not added
    again as a continuation lower bound.

    Because the total budget remains exactly ``B * N0``, a threshold can
    select too few prompts to absorb it under ``n_max``. In that case the
    highest-scoring prompts are force-activated until the allocation is
    feasible, even if their realized signal is below threshold. This keeps the
    fixed-budget constraint explicit rather than silently violating it.
    """
    if isinstance(activation_threshold, bool) or not isinstance(
        activation_threshold, Real
    ):
        raise ValueError("activation_useful_threshold must be a real number")
    activation_threshold = float(activation_threshold)
    if (
        not np.isfinite(activation_threshold)
        or activation_threshold < 0.0
        or activation_threshold > 1.0
    ):
        raise ValueError("activation_useful_threshold must lie in [0, 1]")
    if inactive_floor not in {0, n_min}:
        raise ValueError("inactive_floor must be either zero or n_min")

    num_prompts = int(success_prob.size)
    max_counts = np.full(num_prompts, n_max, dtype=np.int64)
    activation_score = _mixed_group_probability(success_prob, max_counts)
    ranked = np.argsort(-activation_score, kind="stable")
    eligible_count = int(np.sum(activation_score >= activation_threshold))

    base_budget = num_prompts * inactive_floor
    residual_budget = budget - base_budget
    incremental_capacity = n_max - inactive_floor
    if residual_budget < 0:
        raise ValueError("inactive rollout floor exceeds the exact rollout budget")
    if incremental_capacity == 0:
        if residual_budget != 0:
            raise ValueError("dynamic activation has no capacity for residual budget")
        required_active = 0
    else:
        required_active = int(
            np.ceil(residual_budget / float(incremental_capacity))
        )
    if required_active > num_prompts:
        raise ValueError(
            "dynamic activation cannot absorb the exact budget under n_max"
        )
    def allocate_active_count(
        active_count: int,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        active = np.zeros(num_prompts, dtype=bool)
        active[ranked[:active_count]] = True
        rollouts = np.full(num_prompts, inactive_floor, dtype=np.int64)
        continuous = np.full(
            num_prompts,
            float(inactive_floor),
            dtype=np.float64,
        )
        if active_count:
            # Every prompt has already consumed ``inactive_floor`` samples.
            # Only the residual budget is allocated in the continuation.  The
            # waterline solver works with total counts, so add the existing
            # floor for active prompts exactly once when forming its budget.
            continuation_budget = budget - num_prompts * inactive_floor
            active_budget = (
                active_count * inactive_floor + continuation_budget
            )
            active_waterline, active_continuous, active_rollouts = (
                _centered_continuous_rounding_rollouts(
                    success_prob[active],
                    active_budget,
                    n_min,
                    n_max,
                )
            )
            continuous[active] = active_continuous
            rollouts[active] = active_rollouts
            waterline = active_waterline
        else:
            if int(np.sum(rollouts)) != budget:
                raise RuntimeError("dynamic activation selected no feasible prompts")
            all_failure = np.exp(np.log1p(-success_prob) * continuous)
            waterline = float(np.mean(all_failure))
        realized_signal = _mixed_group_probability(success_prob, rollouts)
        return rollouts, continuous, waterline, active, realized_signal

    # Search for the largest useful active curriculum. With prompts ordered by
    # attainable signal, adding prompts reduces the per-active-prompt budget;
    # binary search therefore avoids rerunning the waterline solver O(B) times.
    # Integer rounding can create equality-boundary noise, handled by a small
    # numerical tolerance.
    upper_active = max(required_active, eligible_count)
    selected_active = required_active
    cached: dict[
        int,
        tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray],
    ] = {}

    def evaluate(active_count: int):
        if active_count not in cached:
            cached[active_count] = allocate_active_count(active_count)
        return cached[active_count]

    if eligible_count >= required_active:
        low = required_active
        high = eligible_count
        best_useful = None
        while low <= high:
            middle = (low + high) // 2
            candidate = evaluate(middle)
            candidate_active = candidate[3]
            candidate_signal = candidate[4]
            useful = bool(
                np.all(
                    candidate_signal[candidate_active]
                    >= activation_threshold - 1e-12
                )
            )
            if useful:
                best_useful = middle
                low = middle + 1
            else:
                high = middle - 1
        if best_useful is not None:
            selected_active = best_useful
    else:
        selected_active = upper_active

    rollouts, continuous, waterline, active, realized_signal = evaluate(
        selected_active
    )

    if int(np.sum(rollouts)) != budget:
        raise RuntimeError("dynamic activation missed the exact rollout budget")
    return (
        rollouts,
        continuous,
        waterline,
        active,
        activation_score,
        realized_signal,
    )


def _strict_equalization_diagnostics(
    success_prob: np.ndarray,
    budget: int,
    n_min: int,
    n_max: int,
) -> tuple[np.ndarray, float, bool, bool, float, float, float]:
    """Describe whether strict equal-fidelity rollouts are representable."""
    effective_budget = float(budget - success_prob.size)
    ideal_rollouts = 1.0 + effective_budget * _harmonic_weights(success_prob)
    ideal_log_failure = np.log1p(-success_prob) * (ideal_rollouts - 1.0)
    ideal_waterline = float(np.exp(np.mean(ideal_log_failure)))

    tolerance = 1e-9
    lower_violations = ideal_rollouts < n_min - tolerance
    upper_violations = ideal_rollouts > n_max + tolerance
    continuous_feasible = not bool(
        np.any(lower_violations) or np.any(upper_violations)
    )
    integer_distance = np.abs(ideal_rollouts - np.rint(ideal_rollouts))
    integer_distance_max = float(np.max(integer_distance))
    integer_feasible = bool(
        continuous_feasible and integer_distance_max <= tolerance
    )
    return (
        ideal_rollouts,
        ideal_waterline,
        continuous_feasible,
        integer_feasible,
        float(np.mean(lower_violations)),
        float(np.mean(upper_violations)),
        integer_distance_max,
    )


def _validate_rollout_bounds(
    baseline_n: int,
    n_min: int,
    n_max: int,
) -> tuple[int, int, int]:
    baseline_n = _validate_positive_int("baseline_n", baseline_n)
    n_min = _validate_positive_int("n_min", n_min)
    n_max = _validate_positive_int("n_max", n_max)
    if not n_min <= baseline_n <= n_max:
        raise ValueError(
            f"expected n_min <= baseline_n <= n_max, got {n_min}, {baseline_n}, {n_max}"
        )
    return baseline_n, n_min, n_max


def allocate_rollouts_from_probabilities(
    success_prob: np.ndarray,
    *,
    baseline_n: int,
    n_min: int,
    n_max: int,
    allocation_mode: str = "dynamic_active_equalized",
    raw_success_prob: np.ndarray | None = None,
    normalize_by_rollout_count: bool = True,
    normalize_by_rho: bool = False,
    activation_useful_threshold: float = 0.05,
    batch_divisor: int = 1,
) -> SERAAllocation:
    """Allocate an exact ``len(success_prob) * baseline_n`` rollout budget.

    SERA selects a useful active set, solves the bounded centered-fidelity
    waterline, and completes integer counts by marginal squared-fidelity error.
    ``fixed`` and ``bias_equalized`` retain the homogeneous and all-active
    benchmark paths. Probabilities must be strictly between zero and one.
    """
    baseline_n, n_min, n_max = _validate_rollout_bounds(
        baseline_n,
        n_min,
        n_max,
    )
    batch_divisor = _validate_positive_int("batch_divisor", batch_divisor)
    if not isinstance(normalize_by_rollout_count, (bool, np.bool_)):
        raise ValueError("normalize_by_rollout_count must be a boolean")
    normalize_by_rollout_count = bool(normalize_by_rollout_count)
    if not isinstance(normalize_by_rho, (bool, np.bool_)):
        raise ValueError("normalize_by_rho must be a boolean")
    normalize_by_rho = bool(normalize_by_rho)
    allocation_prob = np.asarray(success_prob, dtype=np.float64)
    if allocation_prob.ndim != 1 or allocation_prob.size == 0:
        raise ValueError("success_prob must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(allocation_prob)) or np.any(
        (allocation_prob <= 0.0) | (allocation_prob >= 1.0)
    ):
        raise ValueError("success_prob must contain finite values strictly between 0 and 1")

    if raw_success_prob is None:
        raw_prob = allocation_prob.copy()
    else:
        raw_prob = np.asarray(raw_success_prob, dtype=np.float64)
        if raw_prob.shape != allocation_prob.shape:
            raise ValueError("raw_success_prob must have the same shape as success_prob")
        if not np.all(np.isfinite(raw_prob)) or np.any(
            (raw_prob < 0.0) | (raw_prob > 1.0)
        ):
            raise ValueError("raw_success_prob must contain finite values in [0, 1]")

    num_prompts = raw_prob.size
    budget = num_prompts * baseline_n

    log_failure = np.log1p(-allocation_prob)
    (
        ideal_rollouts,
        ideal_waterline,
        strict_continuous_feasible,
        strict_integer_feasible,
        lower_bound_violation_fraction,
        upper_bound_violation_fraction,
        integer_distance_max,
    ) = _strict_equalization_diagnostics(
        allocation_prob,
        budget,
        n_min,
        n_max,
    )
    active_prompt_mask = None
    activation_score = None
    activation_realized_signal = None
    resolved_activation_threshold = float("nan")
    recoverability_selected_k = -1
    recoverability_k_min = -1
    recoverability_k_max = -1
    recoverability_candidate_count = 0
    recoverability_reference_rollouts = float("nan")
    recoverability_expected_mixed_group_count = float("nan")

    if allocation_mode == "fixed":
        continuous = np.full(num_prompts, float(baseline_n), dtype=np.float64)
        rollouts = np.full(num_prompts, baseline_n, dtype=np.int64)
        all_failure = np.exp(log_failure * rollouts)
        waterline = float(np.mean(all_failure))
    elif allocation_mode == "dynamic_active_equalized":
        (
            rollouts,
            continuous,
            waterline,
            active_prompt_mask,
            activation_score,
            activation_realized_signal,
        ) = _dynamic_active_equalized_rollouts(
            allocation_prob,
            budget,
            n_min,
            n_max,
            activation_threshold=activation_useful_threshold,
            inactive_floor=0,
        )
        resolved_activation_threshold = float(activation_useful_threshold)
        all_failure = np.exp(log_failure * rollouts)
    elif allocation_mode == "bias_equalized":
        waterline, continuous, rollouts = _centered_continuous_rounding_rollouts(
            allocation_prob, budget, n_min, n_max,
        )
        all_failure = np.exp(log_failure * rollouts)
    else:
        raise ValueError(f"unsupported SERA allocation_mode={allocation_mode!r}")

    used_budget = int(np.sum(rollouts))
    if allocation_mode == "dynamic_active_equalized":
        invalid_positive = (rollouts > 0) & (rollouts < n_min)
        if np.any(invalid_positive) or np.any(rollouts > n_max):
            raise RuntimeError(
                "discard allocator returned a positive count outside bounds"
            )
        if used_budget != budget:
            raise RuntimeError(
                f"discard allocator returned budget {used_budget}, expected {budget}"
            )
    else:
        if np.any(rollouts < n_min) or np.any(rollouts > n_max):
            raise RuntimeError(
                "allocator returned a rollout count outside configured bounds"
            )
        if used_budget != budget:
            raise RuntimeError(
                f"allocator returned budget {used_budget}, expected {budget}"
            )

    # ``1 - (1 - p)**n`` loses all precision for oracle probabilities far
    # below machine epsilon.  The expm1 form remains accurate in the
    # low-probability ImageNet cold start.
    coverage = -np.expm1(log_failure * rollouts)
    positive_rollouts = rollouts > 0
    centered_exponent = np.maximum(rollouts - 1, 0)
    centered_failure = np.ones(num_prompts, dtype=np.float64)
    centered_failure[positive_rollouts] = np.exp(
        log_failure[positive_rollouts]
        * centered_exponent[positive_rollouts]
    )
    fidelity = np.zeros(num_prompts, dtype=np.float64)
    fidelity[positive_rollouts] = 1.0 - centered_failure[positive_rollouts]
    if active_prompt_mask is None:
        objective_prompts = positive_rollouts
        rho_prompts = np.ones(num_prompts, dtype=bool)
    else:
        objective_prompts = positive_rollouts & active_prompt_mask
        rho_prompts = active_prompt_mask
    if np.any(rho_prompts):
        rho_hat = float(np.mean(coverage[rho_prompts]))
        if not np.isfinite(rho_hat) or rho_hat <= 0.0:
            raise RuntimeError(f"invalid active-set rho_hat={rho_hat}")
    else:
        # Degenerate N0 == inactive_floor configuration: there is no active
        # policy objective, so keep a neutral common scale and zero weights.
        rho_hat = 1.0
    if np.any(objective_prompts):
        kappa_hat = float(np.mean(fidelity[objective_prompts]))
        if not np.isfinite(kappa_hat) or kappa_hat < 0.0:
            raise RuntimeError(f"invalid active-set kappa_hat={kappa_hat}")
    else:
        kappa_hat = 1.0
    prompt_weights = np.zeros(num_prompts, dtype=np.float64)
    if normalize_by_rollout_count:
        prompt_weights[objective_prompts] = (
            float(baseline_n) / rollouts[objective_prompts].astype(np.float64)
        )
    else:
        # Ablation: allocation now also determines each prompt's aggregate
        # objective weight. Active samples receive unit weight; inactive
        # prompts remain masked with zero weight.
        prompt_weights[objective_prompts] = 1.0
    rho_normalized = normalize_by_rho and allocation_mode != "fixed"
    if rho_normalized:
        prompt_weights[objective_prompts] /= rho_hat

    all_failure_log = log_failure * rollouts
    all_failure_spread = float(np.max(all_failure) - np.min(all_failure))
    all_failure_log_spread = float(
        np.max(all_failure_log) - np.min(all_failure_log)
    )
    if np.any(objective_prompts):
        objective_centered_failure = centered_failure[objective_prompts]
        objective_centered_log = (
            log_failure[objective_prompts]
            * centered_exponent[objective_prompts]
        )
        centered_failure_spread = float(
            np.max(objective_centered_failure)
            - np.min(objective_centered_failure)
        )
        centered_failure_log_spread = float(
            np.max(objective_centered_log) - np.min(objective_centered_log)
        )
    else:
        centered_failure_spread = float("nan")
        centered_failure_log_spread = float("nan")

    return SERAAllocation(
        raw_success_prob=raw_prob,
        allocation_success_prob=allocation_prob,
        ideal_rollouts=ideal_rollouts,
        continuous_rollouts=continuous,
        rollouts=rollouts,
        all_failure_prob=all_failure,
        coverage_prob=coverage,
        centered_failure_prob=centered_failure,
        fidelity_prob=fidelity,
        prompt_weights=prompt_weights,
        waterline=waterline,
        ideal_waterline=ideal_waterline,
        rho_hat=rho_hat,
        kappa_hat=kappa_hat,
        rollout_count_normalized=normalize_by_rollout_count,
        rho_normalized=rho_normalized,
        strict_equalization_continuous_feasible=strict_continuous_feasible,
        strict_equalization_integer_feasible=strict_integer_feasible,
        lower_bound_violation_fraction=lower_bound_violation_fraction,
        upper_bound_violation_fraction=upper_bound_violation_fraction,
        integer_distance_max=integer_distance_max,
        all_failure_spread=all_failure_spread,
        all_failure_log_spread=all_failure_log_spread,
        centered_failure_spread=centered_failure_spread,
        centered_failure_log_spread=centered_failure_log_spread,
        active_prompt_mask=active_prompt_mask,
        activation_score=activation_score,
        activation_realized_signal=activation_realized_signal,
        activation_threshold=resolved_activation_threshold,
        recoverability_selected_k=recoverability_selected_k,
        recoverability_k_min=recoverability_k_min,
        recoverability_k_max=recoverability_k_max,
        recoverability_candidate_count=recoverability_candidate_count,
        recoverability_reference_rollouts=recoverability_reference_rollouts,
        recoverability_expected_mixed_group_count=(
            recoverability_expected_mixed_group_count
        ),
    )


def uniform_initial_allocation(
    num_prompts: int,
    *,
    baseline_n: int,
    n_min: int,
    n_max: int,
    normalize_by_rollout_count: bool = True,
) -> SERAAllocation:
    """Return the homogeneous cold-start allocation before epoch estimates exist."""
    num_prompts = _validate_positive_int("num_prompts", num_prompts)
    baseline_n, _, _ = _validate_rollout_bounds(baseline_n, n_min, n_max)
    if not isinstance(normalize_by_rollout_count, (bool, np.bool_)):
        raise ValueError("normalize_by_rollout_count must be a boolean")
    normalize_by_rollout_count = bool(normalize_by_rollout_count)
    unknown = np.full(num_prompts, np.nan, dtype=np.float64)
    rollouts = np.full(num_prompts, baseline_n, dtype=np.int64)
    return SERAAllocation(
        raw_success_prob=unknown.copy(),
        allocation_success_prob=unknown.copy(),
        ideal_rollouts=unknown.copy(),
        continuous_rollouts=rollouts.astype(np.float64),
        rollouts=rollouts,
        all_failure_prob=unknown.copy(),
        coverage_prob=unknown.copy(),
        centered_failure_prob=unknown.copy(),
        fidelity_prob=unknown.copy(),
        prompt_weights=np.ones(num_prompts, dtype=np.float64),
        waterline=float("nan"),
        ideal_waterline=float("nan"),
        rho_hat=1.0,
        kappa_hat=1.0,
        rollout_count_normalized=normalize_by_rollout_count,
        rho_normalized=False,
        strict_equalization_continuous_feasible=False,
        strict_equalization_integer_feasible=False,
        lower_bound_violation_fraction=float("nan"),
        upper_bound_violation_fraction=float("nan"),
        integer_distance_max=float("nan"),
        all_failure_spread=float("nan"),
        all_failure_log_spread=float("nan"),
        centered_failure_spread=float("nan"),
        centered_failure_log_spread=float("nan"),
    )


# Backward-compatible name used by earlier public recipes.
fixed_rollout_allocation_without_estimates = uniform_initial_allocation
