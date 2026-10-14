# ADR-003: scikit-learn and imbalanced-learn rather than a hand rolled resampler

## Status

Accepted.

## Context

SMOTE is roughly thirty lines: fit a nearest neighbours index on the minority class, pick a
neighbour, interpolate. Writing it by hand would remove a dependency and would look more
impressive in a portfolio repository.

## Decision

Use `imbalanced-learn` for every resampler and its fold-aware `Pipeline`, and scikit-learn
for folding, estimators and metrics. Write none of it.

## Alternatives considered

**A hand rolled SMOTE.** Rejected because it would measure the wrong thing. The claim is
about what happens to teams using the standard tools, in the standard way. A bug in a hand
rolled resampler would be indistinguishable in the output from the leak being measured, and
a reader would be right to discount every number on that basis.

**Writing the fold-aware pipeline by hand.** Rejected more firmly. `imbalanced-learn`'s
`Pipeline` is precisely the fix this repository recommends. Reimplementing the fix and then
measuring the reimplementation would be circular.

**statsmodels or another statistics framework.** Not needed. Nothing here requires more than
a mean and a difference.

## Consequences

The repository contains no original algorithm, and should not claim to. Its contribution is
the measurement apparatus: the disjoint untouched reference, the fingerprint that proves two
protocols ship one model, and the checker that keeps the prose honest. That is the work, and
pretending the resampler was also the work would undermine it.
