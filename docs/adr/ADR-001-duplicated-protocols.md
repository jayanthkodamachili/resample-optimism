# ADR-001: three protocols as duplicated functions, not one parameterised path

## Status

Accepted.

## Context

The three evaluation protocols differ by very little. A single function taking a
`leak: bool` and a `resampler` would express all three in about fifteen lines, and the
duplication in `protocols.py` would disappear.

## Decision

Keep them as three separate functions, with near identical bodies, in one file, in reading
order.

## Alternatives considered

**One parameterised function.** Shorter and conventional. Rejected because the difference
between the protocols would become a branch on a boolean, and a reader would have to
simulate the function twice in their head to see what the repository is about. The whole
argument here is that the leaked and the corrected path look nearly the same, so making
them look nearly the same on the page is the point rather than a cost.

**A strategy class per protocol.** More extensible and more indirection. There are three
protocols and there will not be a fourth, because the space of places a resampler can sit
relative to a fold has three interesting positions.

## Consequences

A change to the scoring call has to be made three times, and a test enforces that all three
report one score per fold so a missed edit shows up. In exchange, a reviewer who opens
`protocols.py` can see the entire claim of the repository without navigating anywhere. That
trade favours the reviewer, who is the actual audience.
