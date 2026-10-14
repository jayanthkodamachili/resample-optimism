# ADR-005: what this repository deliberately cannot measure

## Status

Accepted. This is the boundary record, and it exists so the limits are a decision rather
than an oversight.

## Context

The experiments produce confident numbers. Several of them are properties of the setup
rather than of resampling in general, and a reader who takes them for more than they are
will reach wrong conclusions in their own work. Recording the boundary explicitly is worth
more than any additional experiment.

## Decision

Four things are out of scope, each for a stated reason, and the README carries them in a
section of their own rather than in a footnote.

## What is out of scope, and why

**Threshold-dependent evaluation.** Both metrics used here rank predictions. Resampling is
most often defended on recall at a fixed operating point, which is untested. This is the
sharpest limitation, because the headline result about resampling not paying for itself is
measured only on the ground its defenders do not choose to stand on. Listed as the Pivot
with an explicit trigger condition.

**Calibration.** Resampling changes the base rate the model sees, so it distorts predicted
probabilities by construction. Nothing here quantifies that. A reader should not conclude
that correct resampling is harmless for a system that uses the probabilities as prices.

**Real data.** See ADR-002. The leak mechanism is structural and transfers. The magnitudes
are properties of the generator. The README states this and no figure should be quoted as a
fact about any real dataset.

**Grouped or temporally ordered data.** Where the unit of observation differs from the unit
of resampling, the leak gets a second route in that this repository does not model.

## Observed against declared

Everything in `docs/metrics.json` is observed: produced by code in this repository, re-run
by CI. One thing is declared rather than observed, and should be read as a modelling
assumption: that the generator's distribution resembles a production problem in any
respect that matters. It was chosen to be awkward, not to be representative.

## Machine dependence

The figures here are scores and counts, both machine independent. No duration is published
as a headline anywhere, because a millisecond is a fact about the runner rather than about
the code. Where the experiments record wall clock, it stays in the JSON and the prose does
not quote it.

## Consequences

The repository makes a narrower claim than its numbers would support if read carelessly,
and says so in the most prominent place a limitation can go. A reviewer who reads this file
first should find nothing in the README that overreaches it.
