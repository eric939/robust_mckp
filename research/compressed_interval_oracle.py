#!/usr/bin/env python3
"""Compressed group-envelope oracle for robust-MCKP threshold intervals.

For fixed multiplier ``lambda``, the baseline-slack Lagrangian expression

    lambda * C(theta) + sum_i max_j(v_ij - lambda*c_ij(theta))

simplifies exactly to

    sum_i max_j(v_ij + lambda*s_ij(theta)) - lambda*Gamma*theta.

Within a group and between two option-deviation breakpoints, all unsaturated
options have the same slope ``lambda`` and all saturated options are constant.
The group envelope is therefore the maximum of one line and one constant.  We
accumulate those pieces with range differences, avoiding the dense
``groups x thresholds x options`` tensor used by the original research oracle.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.optimize as opt

from robust_mckp import PricingInstance
from robust_mckp.exact_bnb import build_full_theta_candidates
from research.novelty_go_no_go import IntervalBound


TOL = 1e-9


@dataclass(frozen=True)
class OracleProfile:
    groups: int
    options: int
    thresholds: int
    local_segments: int
    preprocessing_seconds: float


class CompressedThetaIntervalOracle:
    """Evaluate the same Lagrangian function in O(B+K log B) time."""

    def __init__(self, instance: PricingInstance):
        start = time.perf_counter()
        self.instance = instance
        self.thetas = np.asarray(build_full_theta_candidates(instance), dtype=float)
        self.values = [
            np.asarray([option.value for option in group], dtype=float)
            for group in instance.items
        ]
        self.margins = [
            np.asarray([option.margin for option in group], dtype=float)
            for group in instance.items
        ]
        self.deviations = [
            np.abs(np.asarray([option.uncertainty for option in group], dtype=float))
            for group in instance.items
        ]
        max_options = max(len(values) for values in self.values)
        n_groups = len(self.values)
        self._sorted_values = np.full((n_groups, max_options), -np.inf, dtype=float)
        self._sorted_margins = np.zeros((n_groups, max_options), dtype=float)
        self._sorted_deviations = np.zeros((n_groups, max_options), dtype=float)
        self._sorted_eligible = np.zeros((n_groups, max_options), dtype=bool)
        self._group_sizes = np.asarray([len(values) for values in self.values], dtype=int)
        segment_lo: list[int] = []
        segment_hi: list[int] = []
        segment_group: list[int] = []
        segment_last_saturated: list[int] = []
        local_segments = 0
        for group_index, (values, margins, deviations) in enumerate(
            zip(self.values, self.margins, self.deviations)
        ):
            order = np.argsort(deviations, kind="stable")
            sorted_values = values[order]
            sorted_margins = margins[order]
            sorted_deviations = deviations[order]
            size = len(values)
            self._sorted_values[group_index, :size] = sorted_values
            self._sorted_margins[group_index, :size] = sorted_margins
            self._sorted_deviations[group_index, :size] = sorted_deviations
            self._sorted_eligible[group_index, :size] = True
            positions = np.searchsorted(self.thetas, sorted_deviations, side="left")
            positions = np.clip(positions, 0, len(self.thetas) - 1)
            starts = np.unique(np.concatenate([np.array([0], dtype=int), positions]))
            local_segments += len(starts)
            for segment_index, lo_raw in enumerate(starts):
                lo = int(lo_raw)
                hi = (
                    int(starts[segment_index + 1]) - 1
                    if segment_index + 1 < len(starts)
                    else len(self.thetas) - 1
                )
                if lo > hi:
                    continue
                segment_lo.append(lo)
                segment_hi.append(hi)
                segment_group.append(group_index)
                segment_last_saturated.append(
                    int(
                        np.searchsorted(
                            sorted_deviations,
                            float(self.thetas[lo]),
                            side="right",
                        )
                        - 1
                    )
                )
        self._segment_lo = np.asarray(segment_lo, dtype=int)
        self._segment_hi = np.asarray(segment_hi, dtype=int)
        self._segment_group = np.asarray(segment_group, dtype=int)
        self._segment_last_saturated = np.asarray(
            segment_last_saturated, dtype=int
        )

        value_array = np.concatenate(self.values)
        deviation_array = np.concatenate(self.deviations)
        value_scale = max(
            1.0,
            float(np.ptp(value_array)),
            float(np.max(np.abs(value_array))),
        )
        # Only within-menu tradeoffs matter for a group-separable multiplier.
        # Forming all global pairwise differences would be quadratic in the
        # number of options and dominated preprocessing on large instances.
        within_group_differences = np.concatenate(
            [
                np.abs(margins[:, None] - margins[None, :]).ravel()
                for margins in self.margins
            ]
        )
        positive = within_group_differences[within_group_differences > TOL]
        cost_scale = float(np.median(positive)) if positive.size else 0.0
        if cost_scale <= TOL:
            positive_deviations = deviation_array[deviation_array > TOL]
            cost_scale = (
                float(np.median(positive_deviations))
                if positive_deviations.size
                else 1.0
            )
        self.lambda_scale = max(1e-6, value_scale / max(cost_scale, 1e-6))
        self.multiplier_grid = np.concatenate(
            [np.array([0.0]), self.lambda_scale * np.geomspace(1e-4, 1e4, 25)]
        )
        self._cacheable_lambdas = {float(value) for value in self.multiplier_grid}
        self._fixed_value_cache: dict[float, np.ndarray] = {}

        zero_values = [np.zeros_like(values) for values in self.values]
        self.capacities = self._envelope_values(1.0, zero_values)
        self.preprocessing_seconds = time.perf_counter() - start
        self.profile = OracleProfile(
            groups=instance.n_items,
            options=sum(len(group) for group in instance.items),
            thresholds=len(self.thetas),
            local_segments=local_segments,
            preprocessing_seconds=self.preprocessing_seconds,
        )

    def _envelope_values(
        self,
        lambda_value: float,
        objective_values: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        lam = max(0.0, float(lambda_value))
        values_by_group = self.values if objective_values is None else objective_values
        if lam == 0.0:
            constant = sum(float(np.max(values)) for values in values_by_group)
            return np.full(len(self.thetas), constant, dtype=float)

        zero_objective = objective_values is not None
        base = (
            np.where(self._sorted_eligible, 0.0, -np.inf)
            if zero_objective
            else self._sorted_values
        )
        saturated_scores = base + lam * self._sorted_margins
        prefix_max = np.maximum.accumulate(saturated_scores, axis=1)
        active_scores = base + lam * (
            self._sorted_margins - self._sorted_deviations
        )
        suffix_max = np.maximum.accumulate(active_scores[:, ::-1], axis=1)[:, ::-1]
        constants = np.full(len(self._segment_lo), -np.inf, dtype=float)
        has_constant = self._segment_last_saturated >= 0
        constants[has_constant] = prefix_max[
            self._segment_group[has_constant],
            self._segment_last_saturated[has_constant],
        ]
        first_active = self._segment_last_saturated + 1
        has_line = first_active < self._group_sizes[self._segment_group]
        line_intercepts = np.full(len(self._segment_lo), -np.inf, dtype=float)
        line_intercepts[has_line] = suffix_max[
            self._segment_group[has_line], first_active[has_line]
        ]
        finite_constant = np.isfinite(constants)
        finite_line = np.isfinite(line_intercepts)
        first_line = self._segment_lo.copy()
        both = finite_constant & finite_line
        first_line[both] = np.searchsorted(
            self.thetas,
            (constants[both] - line_intercepts[both]) / lam,
            side="left",
        )
        first_line[finite_constant & ~finite_line] = self._segment_hi[
            finite_constant & ~finite_line
        ] + 1
        first_line = np.minimum(
            np.maximum(first_line, self._segment_lo), self._segment_hi + 1
        )

        intercept_diff = np.zeros(len(self.thetas) + 1, dtype=float)
        slope_diff = np.zeros(len(self.thetas) + 1, dtype=float)
        constant_piece = finite_constant & (self._segment_lo < first_line)
        np.add.at(
            intercept_diff,
            self._segment_lo[constant_piece],
            constants[constant_piece],
        )
        np.add.at(
            intercept_diff,
            first_line[constant_piece],
            -constants[constant_piece],
        )
        line_piece = finite_line & (first_line <= self._segment_hi)
        np.add.at(
            intercept_diff,
            first_line[line_piece],
            line_intercepts[line_piece],
        )
        np.add.at(
            intercept_diff,
            self._segment_hi[line_piece] + 1,
            -line_intercepts[line_piece],
        )
        np.add.at(slope_diff, first_line[line_piece], lam)
        np.add.at(slope_diff, self._segment_hi[line_piece] + 1, -lam)
        intercept = np.cumsum(intercept_diff[:-1])
        slope = np.cumsum(slope_diff[:-1])
        return intercept + slope * self.thetas - lam * float(self.instance.gamma) * self.thetas

    def values_at_lambda(self, lambda_value: float, lo: int, hi: int) -> np.ndarray:
        lam = max(0.0, float(lambda_value))
        if lam in self._cacheable_lambdas:
            if lam not in self._fixed_value_cache:
                full = self._envelope_values(lam)
                full[self.capacities < -TOL] = -np.inf
                self._fixed_value_cache[lam] = full
            return self._fixed_value_cache[lam][lo : hi + 1].copy()
        result = self._envelope_values(lam)[lo : hi + 1].copy()
        result[self.capacities[lo : hi + 1] < -TOL] = -np.inf
        return result

    def bound(self, lo: int, hi: int, local_search: bool = True) -> IntervalBound:
        start = time.perf_counter()
        evaluations: list[tuple[float, float]] = []

        def evaluate(lambda_value: float) -> float:
            lam = max(0.0, float(lambda_value))
            value = float(np.max(self.values_at_lambda(lam, lo, hi)))
            evaluations.append((lam, value))
            return value

        grid = self.multiplier_grid
        grid_values = np.asarray([evaluate(lam) for lam in grid], dtype=float)
        best_index = int(np.argmin(grid_values))
        if local_search and 0 < best_index < len(grid) - 1:
            result = opt.minimize_scalar(
                evaluate,
                bounds=(float(grid[best_index - 1]), float(grid[best_index + 1])),
                method="bounded",
                options={"xatol": 1e-9, "maxiter": 80},
            )
            if math.isfinite(float(result.x)):
                evaluate(float(result.x))
        finite = [(lam, value) for lam, value in evaluations if math.isfinite(value)]
        if not finite:
            return IntervalBound(
                float("-inf"), 0.0, len(evaluations), time.perf_counter() - start
            )
        best_lambda, best_value = min(finite, key=lambda pair: pair[1])
        # The mathematical oracle is exact in real arithmetic.  In binary64,
        # move the evaluated value one representable number upward so that the
        # returned value does not lose the upper-bound direction merely when
        # it is stored.  The paper separately states the solver feasibility
        # tolerance; this is not presented as a proof of interval arithmetic.
        padded = np.nextafter(best_value, math.inf)
        return IntervalBound(
            float(padded),
            float(best_lambda),
            len(evaluations),
            time.perf_counter() - start,
        )


__all__ = ["CompressedThetaIntervalOracle", "OracleProfile"]
