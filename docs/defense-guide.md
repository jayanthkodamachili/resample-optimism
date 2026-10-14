# Defense guide

A study guide, not a script. Every answer below should be re-derived by running the thing
it describes, because an interviewer's second question goes somewhere a memorised answer
cannot follow. Where an answer says "run this", run it before you need it.

## The one sentence version

Resampling an imbalanced training set is correct when you fit and wrong when you estimate,
and the two look identical in code. This repository measures the gap: the leaked protocol
reports **0.9979** mean ROC-AUC and the model it ships scores **0.8739** on data no
resampler has touched.

## The five questions this invites

### 1. Why is the leaked number higher? Describe the mechanism.

SMOTE interpolates a new minority point between a real minority point and one of its
nearest neighbours. Do that across the whole pool and then split into folds, and a
synthetic point derived from row *i* can land in the training fold while row *i* itself
lands in validation. The model has effectively seen a blurred copy of the validation row.
With random oversampling it is not blurred at all, it is the same row, which is why that
case reports **a perfect 1.0000**.

Run `python experiments/e2_by_resampler.py` and talk through the undersampling row, which
is the one that shows you understand the mechanism rather than the slogan.

### 2. If you fix it, does the model get better?

No, and this is the part worth being precise about. Both protocols fit the final model on
the same resampled pool with the same seed, so they produce the same model. The repository
proves it by hashing each model's predicted probabilities on the holdout: identical in
**10 of 10 seeds**. Fixing the protocol moves the estimate by about twelve points and moves
the model by nothing.

The practical consequence, worth saying out loud: a team that fixes this watches its
headline metric collapse and has improved nothing, which is why the bug survives review.

### 3. How do you know your reference is honest?

The holdout is carved off before any protocol runs, and no resampler is ever shown it.
There is a test asserting the pool and holdout share no rows. The reference is a generated
sample rather than a real dataset specifically so it can be drawn large from the same
distribution; on a real dataset the reference would carry its own sampling noise and the
measured optimism would be the leak plus that noise with no way to separate them. That
reasoning is ADR-002 and you should be able to give it without the file in front of you.

### 4. So should people stop using SMOTE?

Be careful here, because the repository does not support the strong version of that claim.
Under average precision, correct resampling improved **0 of 12 cells** and **48 of 60**
runs got worse. Under ROC-AUC on the identical cells, **9 of the same 12** improve. The two
disagree on sign. The average precision loss is **6.7 times** the gain under ROC-AUC, and
average precision is the more appropriate metric for an imbalanced problem, so the
defensible statement is: in these scenarios, under threshold-free metrics, it did not pay
for itself.

What the repository cannot say is anything about recall at a fixed threshold, which is
where resampling's defenders actually make their case. That is the Pivot, and naming it
unprompted is the strongest move available in this conversation.

### 5. What would you do differently?

The honest answer is the E4 history. The first version swept average precision alone and
produced a clean, strong, publishable conclusion. Adding ROC-AUC over the identical cells
reversed the sign. The lesson is not about metrics, it is that the experiment was designed
after the expected answer was already in mind, and a single-metric sweep could not have
told the difference between a real effect and a metric artefact.

The second thing: the first headline run used one seed and showed resampling hurting.
Ten seeds showed it mildly helping. Every published figure is now a mean over seeds with
the spread recorded next to it.

## Numbers to have at hand

| figure | value |
| --- | --- |
| leaked protocol, mean CV estimate | 0.9979 |
| same model, honest holdout | 0.8739 |
| leaked optimism | +0.1240 points |
| corrected optimism | -0.0164 points |
| seeds where both ship one model | 10 of 10 seeds |
| random oversampling, leaked CV | 1.0000 |
| random undersampling, optimism | -0.0222 for undersampling |
| worst estimator, decision tree | +0.3215 points |
| mildest estimator, logistic | +0.0095 points |
| ratio between them | 33.9 times |
| suite | 90 tests, 100 percent line coverage |

## Where you are thin, and what to say

- **You did not implement SMOTE.** Correct, and ADR-003 explains why reimplementing the fix
  and then measuring the reimplementation would be circular. Say that rather than
  apologising for it.
- **It is synthetic data.** Agree immediately, point at ADR-002 for the reason, and say
  which claim would change on real data (the magnitudes) and which would not (the
  mechanism, and the identical-model property).
- **The sweeps are five seeds.** True, and the per-cell spread is wide enough that
  individual cells are noisier than their means. The JSON carries the minimum and maximum
  for every cell; quote the spread rather than only the mean.
- **It is a small repository.** It is. It answers one question completely rather than four
  questions partially, and the receipts tooling is the part that generalises.

## Running it cold

```bash
pip install -e ".[dev]"
make verify          # lint, suite, re-measure everything, check the prose
```

If someone asks whether a number in the README is current, break one deliberately and show
them the checker failing. That demonstration is worth more than any explanation of it.
