# ADR-002: a generated fixture rather than a public imbalanced dataset

## Status

Accepted.

## Context

The repository needs imbalanced binary data. The obvious choice is a well known public
dataset: a credit card fraud set, or one of the UCI imbalanced benchmarks. Real data is
more persuasive to a reader and costs nothing to download.

## Decision

Generate every scenario from `make_classification` with a fixed seed, and commit no data at
all.

## Alternatives considered

**A public imbalanced dataset.** Rejected for a reason specific to what is being measured.
Optimism here is defined as the cross-validated estimate minus the score the same model
achieves on an untouched reference. On a real dataset of fixed size, that reference is a
single finite holdout, so it is itself an estimate carrying its own sampling error. The
measured optimism would then be the sum of the leak and the holdout's noise, with no way to
separate them. A generator can draw a reference of any size from the same distribution,
which makes the reference the cleanest term in the subtraction rather than the dirtiest.

**A committed synthetic file.** Rejected as the worst of both: it fixes the size of the
reference without gaining realism, and it adds a binary to the repository that nothing can
regenerate.

**Downloading a dataset at run time.** Rejected outright. A reviewer has to be able to
reproduce every number offline, and a tool that fetches data at run time stops working the
day the URL moves.

## Consequences

The magnitudes published here are properties of the generator, not of any real problem, and
the README says so in its own section rather than in a footnote. The leak mechanism is
structural and transfers; the numbers do not. Adding a real dataset is listed as future
work precisely because it would answer a different question than the one measured here.
