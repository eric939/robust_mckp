# Positive confirmation: compressed group-envelope interval oracle

Date: 19 July 2026

## Verdict

**The feasibility verdict is now positive for a focused methodological
contribution:** a compressed group-envelope algorithm for evaluating the
Lagrangian upper bound over all robust thresholds without constructing the
dense group-by-threshold cost tensor or solving the bounded-threshold clique LP.

This is not yet positive confirmation that the complete integer solver beats
SCIP. It is positive confirmation of a theoretically identifiable, numerically
valid, scalable, and competitive bounding contribution around which a revised
paper can be built—subject to completing the formal literature non-equivalence
argument and presenting the algebra below as proved propositions.

## Structural result

For a fixed threshold theta, let

    s_ij(theta) = m_ij - (d_ij-theta)^+,
    s_i^*(theta) = max_j s_ij(theta),
    c_ij(theta) = s_i^*(theta)-s_ij(theta),
    C(theta) = sum_i s_i^*(theta)-Gamma*theta.

The existing fixed-threshold MCKP Lagrangian bound at multiplier lambda is

    lambda*C(theta) + sum_i max_j {v_ij-lambda*c_ij(theta)}.

Substitution cancels every group baseline exactly:

    Phi(lambda,theta)
      = sum_i max_j {v_ij+lambda*s_ij(theta)}
        -lambda*Gamma*theta.

For one group and between consecutive local deviation breakpoints, partition
options into saturated options d_ij <= theta and active options d_ij > theta.
Then

    max_j {v_ij+lambda*s_ij(theta)}
      = max { A_i(lambda), B_i(lambda)+lambda*theta },

where

    A_i(lambda) = max_{j: d_ij<=theta} {v_ij+lambda*m_ij},
    B_i(lambda) = max_{j: d_ij>theta} {v_ij+lambda*(m_ij-d_ij)}.

Thus each group contributes only one constant and one common-slope line on each
local interval. Range-difference accumulation evaluates Phi at every global
threshold without iterating over every group at every threshold.

After sorting each menu by deviation, prefix maxima provide every saturated
constant and suffix maxima provide every active-line intercept. For B global
thresholds and K total options, the implemented evaluation therefore costs

    O(B + K log B)

per multiplier after O(B log B + sum_i m_i log m_i) setup, versus O(B*K) for the
dense implementation. The logarithmic term locates multiplier-dependent
constant/line crossovers in the global threshold array by batched binary search;
range accumulation and storage remain linear in thresholds plus options. The interval upper bound remains valid
because every evaluated Lagrangian value is a weak-duality upper bound; imperfect
one-dimensional multiplier minimization can weaken but cannot invalidate it.

## Preregistered experiment

The gates were fixed in `positive_confirmation_study.py` before the confirmatory
scaling run:

1. both methods reach relative gap 1e-6 on every large case;
2. at least 2x geometric-mean speedup for n >= 720;
3. win rate at least 75%;
4. median speedup above one on at least three of four families;
5. root bound no weaker on at least 75%; and
6. maximum absolute identity error versus the original dense oracle at most 2e-6.

### Independent validity campaign

Twelve held-out n=90 cases across all four families and three seeds were checked
against both the original dense oracle and exhaustive fixed-threshold LP scans.

| Measure | Result |
|---|---:|
| Maximum dense-oracle identity error | 7.413e-7 |
| Minimum slack over exhaustive LP maximum | +1.637e-7 |
| Identity gate | Pass |
| Validity gate | Pass |

Additional property tests cover signed objectives, negative margins, irregular
menu sizes, duplicate deviations, and an independently evaluated uncancelled
formula.

### Confirmatory scaling campaign

The campaign used 36 instances, n in {360,720,1440}, all four adversarial
families, three seeds, three timing repetitions, alternating method order, and
single-threaded numerical libraries. The primary large set contains the 24 cases
with n in {720,1440}.

| Preregistered measure | Required | Observed | Result |
|---|---:|---:|---|
| Tolerance rate | 100% | 100% | Pass |
| Geometric-mean adaptive speedup | >=2.0x | 7.14x | Pass |
| Median adaptive speedup | — | 7.45x | — |
| Win rate | >=75% | 100% (24/24) | Pass |
| Families with median speedup >1 | >=3/4 | 4/4 | Pass |
| Root-bound dominance rate | >=75% | 100% | Pass |

Family median end-to-end speedups over the bounded-z group-clique interval
baseline were:

| Family | Speedup |
|---|---:|
| dense_frontier | 8.16x |
| correlated_risk | 7.31x |
| near_tie | 7.52x |
| many_breakpoints | 8.37x |

The compressed root bound was never weaker and was often materially stronger;
the reported speedup includes construction, multiplier search, fixed-threshold
lower-bound evaluations, interval processing, and tolerance certification.

### Held-out stress scaling

Eight further cases at n in {2880,5760}, all four families, a new seed, and two
balanced timing repetitions gave:

- 15.67x geometric-mean speedup;
- 17.18x median speedup;
- wins on 8/8 cases and all four families;
- compressed time 0.43--2.60 seconds;
- clique time 3.53--64.49 seconds; and
- compressed tolerance certification on 8/8 cases.

The clique comparator certified 7/8 within 60 seconds and stopped at relative
gap 8.26e-5 on the n=5760 many-breakpoint case. Therefore the stress campaign's
joint-tolerance gate is formally false, but only because the comparator failed;
it does not weaken the primary preregistered positive confirmation.

## What changed from the previous no-go

The earlier interval implementation explicitly built fixed-threshold costs for
every group and every threshold. Its preprocessing dominated on unique-deviation
instances, allowing the clique LP to win by a large margin. The algebraic
cancellation and group-envelope range accumulation remove precisely that
bottleneck:

- the original dense oracle and compressed oracle are numerically identical;
- the new implementation scales linearly for bounded menus;
- the advantage increases with n rather than disappearing; and
- the method is simultaneously faster and at least as strong as the directly
  transferable bounded-z clique baseline on the confirmatory large cases.

## Novelty position and remaining obligation

The directly relevant literature already contains full robust disjunctions,
bounded-z formulations, clique strengthening, and threshold branch-and-bound.
It does not make threshold branching itself novel. The candidate contribution is
instead the MCKP/GUB-specific cancellation and batched envelope evaluation of a
stronger Lagrangian interval bound.

This direction addresses the explicit computational weakness identified by
Buesing, Gersing, and Koster: their bounded-z branch-and-bound can spend a large
fraction of time solving root LPs, and they identify faster relaxation solution
as future work. A targeted search found no directly matching algorithm, but
absence from search results is not a proof of novelty. Before submission, the
paper must formally compare the new bound and algorithm with Atamtuerk's RP4,
the Buesing--Gersing--Koster bounded-z/clique formulations, divide-and-conquer,
and robust polymatroid formulations.

## Scope of the positive verdict

Confirmed:

- exact algebraic identity at each multiplier;
- valid interval upper bounds;
- near-linear worst-case scaling for bounded menu sizes;
- broad, preregistered computational superiority over the relevant clique
  interval baseline; and
- a credible, focused methodological contribution.

Not yet confirmed:

- superiority of the complete integer solver over SCIP;
- formal novelty relative to every projection of RP4 or polymatroid formulation;
- performance on external real-world robust-MCKP instances; and
- acceptance at any particular journal.

The next manuscript should therefore be built around the compressed strong-bound
oracle and its theorem, not around minimal conflicts or generic interval
branching. Exact-solver integration and external-instance evaluation become
secondary computational sections rather than the novelty claim.

## Reproducibility

- Implementation: `research/compressed_interval_oracle.py`
- Confirmatory runner: `research/positive_confirmation_study.py`
- Tests: `tests/test_compressed_interval_oracle.py`
- The historical exploratory output directories are intentionally omitted from
  the clean v4 release; this verdict retains their durable findings. The
  protocol-fixed evidence used by the paper is in
  `results/v4_publication_20260720_final/`.
- Historical regression result at this gate: 112 tests passed.
