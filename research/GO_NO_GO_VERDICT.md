# Go/no-go verdict for a new robust-MCKP contribution

Date: 19 July 2026

## Executive verdict

**No-go on rewriting the paper around either the tested interval branch-and-bound or the tested group-conflict cuts.** Neither prototype currently produces the material end-to-end advantage required for a top-journal methodological contribution. The interval bound is fast and unusually tight, but the resulting exact method is slower than the existing threshold solver and has the same certification rate. Exhaustive group conflicts can be strong on small arbitrary support directions, but the scalable separator finds almost no useful cuts on the economically relevant objective.

There is one conditional research opportunity: characterize and compress the strong full-threshold disjunctive LP relaxation. On the 24 SCIP-certified hard instances it closes 91.4% of the compact-LP gap at the median. This is only worth pursuing if a new group-specific formulation, separation theorem, or dominance result can be proved relative to existing generic robust formulations. The strength of the relaxation by itself is not novel.

## Reproducible design

The complete command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python research/novelty_go_no_go.py all \
  --output-dir results/novelty_go_no_go_20260719 \
  --poly-instances 40 --directions 16 \
  --sizes 30,60,90 --seeds 0,1,2 \
  --families dense_frontier,correlated_risk,near_tie,many_breakpoints \
  --exact-max-n 60 --exact-time-limit 10 \
  --cut-sizes 30,60 --cut-samples 80
```

The study performed:

1. Exact enumeration of every integer assignment on 40 small instances.
2. 640 support-function comparisons against the exact integer hull.
3. Exhaustive enumeration of every inclusion-minimal group conflict on those small instances.
4. One hundred randomized interval-bound instances and 600 interval checks.
5. Thirty-six hard benchmark instances up to `n=90`.
6. Twenty-four controlled exact/anytime comparisons through `n=60`.
7. Twenty-four scalable group-conflict separation runs with SCIP-certified reference optima.
8. Objective agreement checks between certified interval, threshold, and SCIP solutions.

All repository tests pass after adding six prototype-specific validity tests.

## Result 1: small-instance polyhedral test

| Measure | Result |
|---|---:|
| Exact support problems | 640 |
| Problems with a nonzero compact-LP gap | 185 (28.9%) |
| Median disjunctive gap closure, conditional on a gap | 9.5% |
| Median exhaustive-conflict gap closure, conditional on a gap | 100.0% |
| Exhaustive conflicts reach the integer hull, conditional | 57.3% |
| Mean inclusion-minimal conflicts per instance | 35.6 |

This initially looks favorable for group conflicts, but the effect is highly objective-dependent. For the actual model objective, exhaustive conflicts close only 6.5% of the gap at the median, while the threshold disjunction closes 47.9%. The 100% median closure is driven by random positive and signed support directions. Moreover, conflicts fail to describe the hull in 42.7% of nontrivial cases, proving that stronger lifted or non-conflict facets are required.

## Result 2: interval Lagrangian bound

The interval bound was valid in all tests. Its worst apparent singleton underestimate relative to the fixed-theta LP was positive after outward numerical padding, so no invalid upper bound was observed.

| Measure | Result |
|---|---:|
| Random cases / checked intervals | 100 / 600 |
| Intervals tight within `1e-6%` | 88.5% |
| Median interval excess over exact fixed-theta LP maximum | approximately 0% |
| 90th-percentile excess on the hard benchmark | 0.148% |
| Median initialization speedup over all fixed-theta LPs | 2.75x |
| Median root gap to HullRound | 0.163% |

This is a useful computational primitive: it gives an immediate finite global upper bound without constructing every fixed-theta hull. It is not yet a sufficient contribution because full fixed-theta LP initialization is already small in absolute time on most tested instances, and the savings do not carry through to the exact solve.

## Result 3: end-to-end exact performance

Under the same 10-second limit on 24 instances with `n=30,60`:

| Measure | Interval prototype | Existing threshold solver | SCIP |
|---|---:|---:|---:|
| Certification rate | 87.5% | 87.5% | 100% |
| Median runtime | 0.332 s end-to-end | 0.286 s | 0.954 s overall |
| Median interval/threshold speed ratio | 0.80x | baseline | — |

The aggregate SCIP median hides the decisive tail result. On many-breakpoint instances, SCIP solves in roughly `0.005–0.19` seconds, while both custom methods time out on three of six cases or require seconds. The interval method prunes many threshold values—only 13.6% are solved at the median—but its repeated interval bounds and difficult singleton MCKPs erase that advantage.

HiGHS returned numerical rather than certified results for the compact mixed-integer formulation in this environment, so SCIP is the reliable external control.

## Result 4: scalable group-conflict separation

All 24 reference objectives were certified by SCIP.

| Measure | Result |
|---|---:|
| Median compact-LP gap | 0.735% |
| Median disjunctive-LP gap | 0.060% |
| Median conflict-cut LP gap | 0.735% |
| Median disjunctive gap closure | 91.4% |
| Median conflict-cut gap closure | 0.0% |
| Median cuts added | 0 |
| Instances where cuts improve compact LP | 25% |

The separator found one weak cut on each many-breakpoint instance and none elsewhere. Even where found, closure was small. This is a no-go for the proposed minimal-conflict family and randomized-rounding separator as a headline method.

## Scientific interpretation

The tests reject two attractive but insufficient narratives:

- “Interval search avoids breakpoint enumeration and is therefore faster.” It avoids many singleton solves but is not faster end to end and does not outperform SCIP.
- “The multiple-choice structure yields powerful robust GUB conflict cuts.” Exhaustive conflicts reveal some small-instance hull structure, but the relevant objective faces are weakly affected and practical separation fails.

The strongest observed object is the full-threshold disjunctive relaxation. Unfortunately, finite-threshold robust formulations and strong generic robust-binary formulations are established literature. A publishable contribution must therefore be a new theorem about how the partition/GUB structure represents or separates this relaxation—not another implementation of the relaxation.

## Only defensible next research gate

Do not alter or reframe the separate v3 manuscript. Continue the independent
v4 research line only through the following theory-first gate:

1. Compare the disjunctive relaxation formally and computationally with the strongest generic formulations of Atamtürk, Joung--Park, and Büsing--Gersing--Koster.
2. Enumerate and classify the non-conflict facets appearing when exhaustive group conflicts fail to reach the small-instance integer hull.
3. Seek a group-lifted inequality or compact extended formulation that strictly dominates the generic formulations on an infinite family.
4. Prove separation or give a rigorous complexity result.
5. Implement the result as a SCIP separator or formulation and require a material root-gap and runtime improvement over native SCIP and the generic robust methods.

If steps 2--4 do not produce a theorem, stop this top-journal algorithmic
project. These experiments neither evaluate nor reposition the separate v3
paper.

## Limitations

- The exact hull study uses `n=4–6`, three options per group, and synthetic data.
- The scalable cut separator is heuristic; failure does not prove that polynomial or exact separation is impossible.
- The prototype is Python, so absolute timing is not a definitive comparison with an optimized implementation. The absence of a relative advantage and SCIP's many-breakpoint dominance are nevertheless strong go/no-go evidence.
- The experiment does not yet include the authors' implementation of every generic robust formulation. This is precisely why no novelty claim should be made from these results alone.
