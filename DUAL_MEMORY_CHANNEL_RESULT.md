# The dual branch's memory channel is dead at inference

Measured 2026-09-21 against `exp/dual` checkpoint `iter-21000-avg-2.pt`, code at
`eb3edf8`, the 76-utterance `bwe_eval` set, `--num-step 16`. Both the checkpoint
and the generated audio were deleted after these numbers were taken, so this
file is the record.

## The question

The dual branch and the band adapter share one leak: at training, the branch's
only utterance-specific channel to high-band detail is a backbone hidden state
computed on `xt = features * t + noise * (1 - t)`, which carries the true high
band scaled by `t`. Neither arm masks it. `mask_xt` exists on the adapter but
was `false` in every run, including `adapter_gan`.

They differ at inference, and only there.

The adapter's branch reads a hidden state computed on the state that carries the
adapter's own high-band writes, so the channel is off-distribution but alive --
an ordinary self-consistent bootstrap. The dual's is computed on `x`, the stock
trajectory, which by construction never receives the branch's writes:

```python
x = x + v * dt  # the stock trajectory, which never reads the branch
y = self._project(x, y + u * dt, mask)
```

`x`'s 8-12 kHz band sits at -78.9 dB against a -44.3 dB reference. The channel
carries nothing about this utterance's missing band, so the branch falls back on
its prior, which is what the veil was.

Note that the dual branch also takes `xt=y` directly, and `y` does carry its own
writes. The memory channel alone was dead, and that alone accounted for the
result below.

## The test

Recompute the hidden state with a second `_backbone_step` on the projected state
`y` instead of `x`, and feed that to the branch. The velocity `v` still comes
from `x`, so the low band, and with it the identity guarantee, is untouched.
Inference-only; no retraining. Gated by the environment variable
`ZIPVOICE_DUAL_MEMORY_FROM_PROJECTED`, where `1` selects this variant.

## Result at sr8000 (4 kHz prompt cutoff)

Band figures are `band_frames.py` speech rows, in dB.

| system | SIM-o | UTMOS | RTF | 6-8k | 8-12k | swing 8-12k |
|---|---|---|---|---|---|---|
| reference | - | - | - | -40.3 | -44.3 | 40.4 |
| stock | 0.610 | 3.89 | - | -77.2 | -78.9 | 10.7 |
| cascade (NVSR prompt) | 0.563 | 4.14 | - | -48.2 | -46.8 | 31.5 |
| dual, unchanged | 0.621 | 4.12 | - | -62.3 | -68.7 | 15.6 |
| **dual, memory from `y`** | **0.620** | **4.17** | 0.342 | **-48.9** | **-54.6** | 30.9 |
| dual, single forward on `y` | 0.347 | 4.14 | 0.198 | -51.4 | -57.1 | 34.0 |
| projgan (adapter arm) | 0.608 | 4.15 | 0.376 | -54.6 | -57.1 | 40.8 |

Taking the memory from `y` gains 14.1 dB at 8-12 kHz and 13.4 dB at 6-8 kHz.
Swing doubles, 15.6 to 30.9, so the added energy tracks speech rather than
sitting as a static veil. Identity is unchanged, 0.621 to 0.620, and UTMOS
rises. The acceptance gate passes on all three legs: +24.3 dB over stock,
UTMOS 4.17, SIM-o 0.620 above stock's 0.610.

It beats projgan on band level in both columns while costing less per sample,
RTF 0.342 against 0.376. projgan keeps the better swing, 40.8 against 30.9,
with the reference at 40.4.

sr16000 was not measured.

## The second backbone forward is not optional

Dropping it -- one forward on `y` whose velocity also drives `x` -- saves 0.14
RTF and costs SIM-o 0.620 to 0.347, below stock and barely above the 0.34 floor.
The band and UTMOS barely move, so the saving buys nothing.

The adapter arm predicted the direction but understated it: adapter30k scores
0.517 without projection against 0.618 with it, a loss of 0.101, where the dual
loses 0.273. The difference is feedback. In the adapter, the state the branch
writes and the state the backbone advances are the same tensor, so the backbone
sees a state it partly authored and corrects towards it. In the single-forward
dual, the backbone reads `y` and advances `x`, so `x` is stepped by a velocity
fitted to a different trajectory with nothing correcting the divergence, and the
error accumulates open-loop over all 16 steps.

This is an empirical result on one checkpoint at one rate, not a proof that no
single-forward scheme can work. A branch retrained on the single-forward
trajectory would be told how to correct exactly the mismatch that produces the
0.347, and has not been tried.

## Cost, stated plainly

Per solver step the unchanged dual runs one backbone forward and one branch
forward. This variant runs two backbone forwards and one branch forward, both
backbone calls at the same `t`, inside the same step, feeding the same step's
branch. There is no second model, no second checkpoint, no second TTS pass and
nothing that runs after the solve. The two calls can be batched along the batch
dimension exactly as `_backbone_step` already does for classifier-free guidance,
which would make it one launch at double batch rather than two launches.

At training only one forward was needed, because teacher forcing made `x` and
`y` the same state. This variant restores the training semantics at inference
rather than adding a stage.
