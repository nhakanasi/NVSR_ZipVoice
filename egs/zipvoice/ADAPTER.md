# The band adapter

This document records what the adapter line of the study is, how it is trained,
and how every number reported for it was produced. It covers that line only; the
jointly-trained ResUNet extender (`zipvoice_bwe.py`) and the AnyBand-style
retraining are separate designs and are described elsewhere.

The reported model is `exp/adapter/iter-30000-avg-2.pt`, run at inference with
`--decouple-residual 1`. Four further arms were trained to recover speaker
identity — two of them adding explicitly identifying conditioning — and all four
lost it; they are documented under "What was tried and failed" because the
negative results are what motivated the control experiment that did work.

## The problem

Zero-shot TTS is prompted with a recording of the target speaker. A released
ZipVoice is trained on 24 kHz speech and expects a 100-bin log-mel covering the
whole band. A prompt that arrives from a telephone line, a codec, or an old
recording carries energy only up to its own Nyquist, so the bins above the
cutoff sit at the empty-bin floor. The backbone has never seen that input, and
its output degrades: at a 4 kHz cutoff, UTMOS falls from 4.17 (full band) to
2.80.

The fix has to leave the released checkpoint alone. Retraining the backbone to
tolerate band-limited prompts is possible and was measured — it costs speaker
similarity at every bandwidth (−0.052 at 8 kHz, −0.089 at 22.05 kHz against the
same model with the backbone frozen), because a backbone that is updated on
band-limited data stops being the model whose identity behaviour the released
weights encode.

## Architecture

A ControlNet-style side branch. The frozen backbone runs on exactly the tensors
it was pretrained on. The branch reads the same conditioning, and its only
output is an additive residual on the backbone's predicted velocity:

```
vt = backbone(t, xt, text_condition, speech_condition)      # frozen
residual = adapter(hidden, cond_feature, padding_mask)
residual = residual * write_mask(band_mask)                 # masked write
vt = vt + residual
```

Two structural guarantees follow, neither of them left to a learned gate:

1. **The backbone is never perturbed at its input.** Whatever the prompt's
   bandwidth, the frozen model sees the tensors it was trained on.
2. **The residual is multiplied by the band mask**, so it is identically zero in
   observed bins and identically zero everywhere when the prompt is full band.
   Full band is an algebraic identity rather than a behaviour to learn: at
   24 kHz this model *is* the released checkpoint, at every point in training.

The mask is the soft ramp `band_limit` returns (a raised-cosine roll-off of
`transition_bins` width), blurred by `write_mask_blur` bins so the branch is not
asked to write a discontinuity at the cutoff.

### Conditioning streams

Three framewise streams exist in the code, each producing `(B, T, 256)`, summed
into the single `cond_feature` the branch cross-attends:

| stream | source | module | flag |
| --- | --- | --- | --- |
| band-limited mel + per-bin observed-ness mask | the condition tensor itself | `FrequencyEncoder` | always on |
| prompt spectral statistics: per-bin mean and standard deviation over the prompt frames, plus a has-prompt flag | the same band-limited condition | `SpeakerEncoder` | `--use-speaker-cond` |
| cutoff in Hz, sinusoidally embedded, broadcast over time | `cutoff_hz` | `RateEncoder` | `--use-rate-cond` |

**The reported arm uses the first stream only.** Both flags are off in
`exp/adapter/model.json`, which is the checkpoint every number in the Results
section comes from. The other two streams were each trained as a separate arm and
each lost identity; see below. They are kept in the code because the negative
results are part of the paper, not because the shipped model uses them.

Nothing any stream reads leaks the high band: all of them are taken from the
*band-limited* condition, the same tensor the backbone sees. In particular F0 was
never taken from the waveform — training band-limits only in the mel domain, so
the audio in the batch is clean and full-band, and anything read off it would
hand the branch the exact bins it exists to invent.

## Training

One run, 30k iterations, from the released `model.pt`:

- `--adapter-trainable adapter` leaves every released tensor out of the
  optimizer, not merely at zero gradient. ScaledAdam keeps a per-tensor scale,
  so exclusion — not a zero gradient — is what guarantees the released weights
  are untouched.
- Cutoffs are sampled per utterance from `[cutoff_min, cutoff_max]` with a
  curriculum (`--curriculum-rho 0.7`) that starts near easy (wide) bandwidths
  and reaches uniform by the end, and `--full-band-prob 0.2` so a fifth of the
  batch exercises the identity case.
- Band-limiting is randomised — roll-off width uniform in `[0.5, 1.5]×` the
  nominal, a per-utterance jitter on the floor, and Gaussian noise in the dead
  band — so the branch does not overfit an idealised brick wall.
- The prompt span is drawn by `condition_time_mask` at 0–30% of the utterance,
  and may be empty; the has-prompt flag exists so the branch can tell an empty
  prompt from a silent one.

### Losses

Three terms, over the generated span only. The reported arm runs the first two;
the third is computed and logged at weight 0 so its magnitude is known before any
run spends gradient on it, and only `adapter_spkv` actually weighted it.

1. **Flow matching**, `mean((vt − ut)²)` — the backbone's own objective.
2. **Masked reconstruction** on the endpoint estimate `x1_hat = xt + (1−t)·vt`,
   L1, weighted by `band_mask` and normalised by the total weight so a 4 kHz
   cutoff does not dominate the gradient merely by having more bins to invent.
3. **A dispersion floor** (`--variance-loss-weight`). Both terms above are
   minimised by a conditional mean: flow matching scores a velocity, the L1 an
   absolute level, and the average high band over the batch does well on each.
   This term pools each utterance to one spectrum over the writable bins, takes
   the across-batch standard deviation per bin, and charges the prediction for
   falling *below* the target's. It is a hinge, not a distribution match — a
   branch writing more varied bands than the data is not collapsing.

### What was tried and failed

Five arms, each adding either information or loss pressure to the baseline.
Every one of them lost SIM-o. Exact settings, read back from each run's own log:

**`adapter` — baseline.** 30k iterations from the released `model.pt`,
`base_lr 0.001`, `curriculum_rho 0.7` over the full 30k, cutoffs in
[1000, 12000] Hz, `full_band_prob 0.2`. Mel stream only; no speaker stream, no
rate stream, no F0, no dispersion floor, no adversarial term. SIM-o
0.246 / 0.517 / 0.711 at 4 / 8 / 16 kHz.

**`adapter_cond` — F0 stream + rate stream.** `use_f0_cond=true`,
`use_rate_cond=true`; otherwise identical to the baseline (fresh 30k run from
`model.pt`, same lr and curriculum). F0 was computed from the band-limited mel by
`f0_from_mel`, not from audio. **Worst arm in the study: 0.159 / 0.286 / 0.648**,
−0.087 at 4 kHz and −0.231 at 8 kHz against a 0.008 noise floor, with no UTMOS
gain and validation loss identical on every term. The reading: pitch is
informative but not *identifying*, so the branch can satisfy the objective by
writing the conditional-mean high band over every speaker at that pitch, which is
exactly the band that scores worst on speaker similarity.

**`adapter_spkv` — speaker stream + rate stream + dispersion floor.**
`use_speaker_cond=true`, `use_rate_cond=true`, `variance_loss_weight=5.0`; again
a fresh 30k run from `model.pt`. The speaker stream was the direct answer to the
F0 failure — per-bin mean and standard deviation of the prompt are identifying by
construction — and the dispersion floor attacks the same collapse from the loss
side. SIM-o **0.232 / 0.472 / 0.686**, below the baseline at every rate.

**`adapter_id` — speaker-perceptual loss.** A 10k finetune *off the baseline's*
`iter-30000-avg-2.pt`, `base_lr 0.0005`, `curriculum_rho 0.0` (uniform cutoffs,
since the branch is already trained), `identity_loss_weight=0.1` against a
pretrained WeSpeaker encoder on the vocoded endpoint estimate;
`fmatch_loss_weight` stayed at 0.0, so feature matching was not actually
exercised in this arm. SIM-o **0.238 / 0.514 / 0.716** — a clean null, all three
deltas straddling zero against the 0.008 floor.

**`adapter_gan` — cross-band discriminator.** A 10k finetune off
*`adapter_spkv`'s* checkpoint (not the baseline's), `base_lr 0.0005`,
`adapter_gan_start_step=0`, `adv_loss_weight=0.02`,
`feat_match_loss_weight=2.0`, `disc_lr=6e-5`, `gan_t_min=0.5`,
`variance_loss_weight=0.0`. The discriminator asks whether *this* invented band
follows *this* observed low band — the first term in the study that is not
satisfiable by a speaker-generic answer. SIM-o **0.204 / 0.372 / 0.693** against
its own starting point's 0.232 / 0.472 / 0.686: −0.028 at 4 kHz, −0.100 at 8 kHz
(about ten unpaired standard errors), +0.007 at 16 kHz (under one). Adversarial
pressure on the written band costs identity and buys nothing elsewhere.

Summary at a 4 kHz cutoff:

| arm | what changed | starts from | SIM-o @4 kHz |
| --- | --- | --- | --- |
| `adapter` | — (mel stream only) | `model.pt` | 0.246 |
| `adapter_id` | `identity_loss_weight 0.1` | `adapter` | 0.238 |
| `adapter_spkv` | speaker + rate streams, `variance_loss_weight 5.0` | `model.pt` | 0.232 |
| `adapter_gan` | adversarial 0.02 + feature matching 2.0, `disc_lr 6e-5` | `adapter_spkv` | 0.204 |
| `adapter_cond` | F0 + rate streams | `model.pt` | 0.159 |

None recovered identity, and the two that added *identifying* information
(`adapter_spkv`, `adapter_gan`) did no better than the one that added none. That
is what pointed at the control experiment below: the loss is not in the band the
branch writes, so no amount of information about that band can fix it.

## Why identity is lost: the trajectory-feedback control

The branch writes only the unobserved band, yet the identity loss does not live
there. Two measurements establish this.

**Lowpass control.** Lowpass every system back at its own cutoff, so only bins
the branch can never write remain, and score again. At 4 kHz the adapter is still
0.115 below stock, against a 0.122 full-band gap: **roughly 94% of the identity
loss sits in bins the branch never touches.**

**Decoupling.** The residual is added to `vt`, so it changes `x_t` at the next
solver step, and the frozen backbone's *own low-band generation* therefore drifts
onto a trajectory it was never trained on. `--decouple-residual 1` removes that
feedback: the backbone is evaluated at `x_t − delta`, the trajectory it would have
followed alone, while the solver still returns `x_stock + delta`. This restores
stock SIM-o exactly (0.246 → 0.368 at 4 kHz).

So the loss is not a property of the invented band. It is the frozen backbone
being pushed off-distribution by the branch's own writes accumulating through the
solve.

At a 4 kHz cutoff the same control shows the *gain* sharing that mechanism.
`decoup_a10` runs at full dose, so the entire residual still reaches the vocoder,
yet its DNSMOS P.808 is 3.239 against the coupled adapter's 3.674 and stock's
3.210. At that cutoff, then, the quality gain also lives in the trajectory
feedback rather than in the written band, and the two move together.

This does **not** generalise upward, and the rate sweep below is what shows it.
At 8 kHz and 16 kHz the decoupled branch keeps most of the quality gain
(UTMOS +0.14 and +0.15 over stock) while holding stock identity. So the
one-for-one trade is a property of the 4 kHz regime, where the branch must
invent five sixths of the spectrum and has little to write that is worth
anything on its own, rather than a property of the design.

### The two inference-time knobs: `a` and decoupling

Both are inference-only. Neither changes a weight, and the same 30k checkpoint
produces every number in this document.

**`a` — the residual scale** (`--residual-scale`). A scalar multiplying the
branch's masked residual before it is added to the backbone's velocity:

```
vt = vt_backbone + a * residual * write_mask(band_mask)
```

`a = 1.0` is the branch exactly as trained, since the scale is fixed at 1.0
throughout training. `a = 0.0` deletes the branch and reproduces the frozen
released model bit for bit. Intermediate values walk between the two without
retraining, so `a` is a *dose*: how much of the branch's correction is
administered. It is not a hyperparameter the model was fitted with; it is a dial
on a finished model, which is why a sweep over it costs only inference.

**Decoupling** (`--decouple-residual 1`). By default the residual is added into
`vt`, so it changes `x_t` at the next solver step, and every later step of the
frozen backbone is evaluated on a state that the branch has already moved.
Decoupling accumulates the branch's own contribution as `delta` and evaluates
the backbone at `x_t − delta`, the trajectory it would have followed alone, while
the solver still returns `x_stock + delta`. The branch's writes therefore reach
the output but never reach the backbone's input.

The two compose: `a` sets how much residual is added, decoupling sets whether
that addition is allowed to feed back. Every arm reported below is decoupled,
swept over `a ∈ {0.25, 0.5, 0.75, 1.0}`.

### Penalising the feedback during training: the drift term

The control above localises the identity loss: it is the frozen backbone being
pushed off-distribution by the branch's own writes feeding back through the
solve. Decoupling removes that feedback and restores stock SIM-o, but it also
removes the coupling the design exists to exploit — a decoupled branch is a
cascade in a single process. The training-side version of the same idea is to
leave the coupling in place and charge the branch for the part of it that does
damage.

What does damage is the off-diagonal Jacobian block
`J[low, high] = ∂B_low/∂x_high`: how far the backbone's *low*-band velocity moves
when the branch writes in the *high* band. Measured on this model its gain is
about 1.15–1.17. The term estimates `J·r` by finite difference — one extra
backbone forward at `x_t + r·dt`, differenced against the unperturbed one and
divided by `dt`, so the result is a directional derivative in velocity units —
and charges its energy outside the written band.

That extra forward stays in the graph, since the gradient the term exists to
create flows back through it, so a second full set of backbone activations is
live at once. Both arms below therefore ran at `--max-duration 36 --max-len 14`
against the other arms' 72 and 20. **Their numbers are not like-for-like with the
rest of this document**: smaller batches, mid-training checkpoints, no averaging.

**`adapter_drift` — unnormalised, `drift_loss_weight 10`.** The term as first
written charges `‖J·r‖²`. The branch can lower that by aiming better or by
writing less, and it took the second route. Against the baseline at matched
steps, validation drive ran 2.1–2.3x weaker (0.0378 against 0.0825 at step
4000), and band energy over the frozen model reached +0.76 dB at step 3000 and
+2.07 dB at 6000 — 1.31 dB per 3000 steps, where the untermed baseline reaches
+11.6 dB, so roughly 21k further steps to parity. SIM-o was high (0.382 at 3k,
0.367 at 6k) and UTMOS was not (3.22, 3.35).

That is the pre-registered falsifier for the idea: if SIM-o rises while energy
and UTMOS fall back onto the curve `--residual-scale` already traces without
retraining, the term is a brake, not a steering input. It fired, and the arm was
stopped at 6000 rather than run out.

**`adapter_driftn` — normalised, `drift_loss_weight 0.05`.** `‖J·r‖²/‖r‖²` is the
gain *along* `r`, unchanged when `r` is scaled, so shrinking the residual should
buy nothing and only its direction should be charged. Drive recovered as
intended: validation drive climbed from 0.66 of the baseline's at step 2000 to
0.86 at 6000. The output went well past the baseline — **band energy +19.8 dB**
over the frozen model (−10.59 against stock's −30.43), beyond even the cascade's
−13.74, and **UTMOS 4.09**, the first arm in the study to clear the 3.9 gate.
WER was unaffected at 3.09%.

**SIM-o collapsed to 0.179.**

Read against the untermed baseline's *30k* number, that looked like the residual
spilling outside the band it is meant to write. It is not, and the comparison was
the error. `adapter_driftn` was at step 6000; the baseline figure it was held
against was at step 30000. Running the untermed baseline's own `checkpoint-6000`
through the same evaluation gives the matched-step control:

| arm at step 6000 | band energy | SIM-o | UTMOS | WER (seed-tts) |
| --- | --- | --- | --- | --- |
| untermed baseline | +12.17 dB (−18.26) | 0.231 | 3.88 | 3.49% |
| `adapter_driftn` | +19.84 dB (−10.59) | 0.179 | 4.09 | 3.09% |
| `adapter_drift` (unnormalised) | +2.07 dB | 0.367 | 3.35 | 3.19% |

One confound in that table, which is why the arm is being taken further rather
than declared a win. The drift arms run `--curriculum-steps 10000` (the stage is
scheduled for 10k steps, not 30k), so at step 6000 they are 60% through the
easy-to-balanced cutoff ramp while the baseline, scheduled against 30000, is 20%
through. The drift arms have therefore seen more of the hard low-cutoff
condition that the 4 kHz evaluation is, and some of the energy gap is that
exposure rather than the term. The comparison that removes it is
curriculum-complete against curriculum-complete: `adapter_driftn` at 10000
against the untermed baseline at 30000. That is the arm now running.

Against the step-matched control the normalised term is **+7.7 dB of band energy
and +0.21 UTMOS for −0.052 of SIM-o**, not a collapse from 0.367 — 0.367 was the
*unnormalised drift arm's* SIM-o, never the baseline's. The baseline reaches only
0.246 at 30k, and it gets there while its band energy stays flat (+12.17 dB at
6k, +11.6 dB at 30k): what the remaining 24k steps buy the untermed branch is
identity, not energy.

That changes the reading of both arms. The unnormalised term is confirmed as a
brake — at matched step it gives up 10 dB of energy against the control to buy
SIM-o. The normalised term is the opposite: more drive, more energy and better
UTMOS than the control at the same step, at a real but modest identity cost, and
sampled at a step where the control itself has not yet recovered its own
identity.

The first explanation offered here for the collapse was wrong and is recorded
because the correction is the useful part. The claim was that `‖J·r‖²/‖r‖²` with
an attached denominator has a gradient component that lowers the loss by
*growing* `‖r‖`, so the denominator acts as a reward, and that detaching it
charges direction only. The gradients say the opposite. With the denominator
attached,

    r · ∇L = (2/‖r‖²)(‖J·r‖² − L‖r‖²) = 0

exactly: the radial component vanishes, so descent can rotate `r` but cannot
lengthen or shorten it. That is the genuinely scale-invariant, direction-only
form. Detaching the denominator to a constant `c` leaves `∇L = (2/c)·JᵀJ·r`,
which points along `+r` — descent shrinks the residual, so the detached form is
the brake again, now carrying a per-step adaptive weight. Measured: a detached
arm at weight 0.3 reached batch `‖r‖` 0.008096 by step 1000, the lowest of any
arm at that step, with the term at 69% of total loss.

The real hazard in the attached form is not a reward but a singularity: `1/‖r‖²`
diverges if the branch collapses. That is handled by flooring the denominator at
a drive of 1e-4 (residual RMS 0.01, an order below anything a working branch has
shown), below which the term degrades to the plain unnormalised penalty. The
code now does that and the detach is reverted.

Summary at a 4 kHz cutoff, band energy relative to the frozen model (stock
−30.43 dB, cascade −13.74 dB, untermed baseline at 30k −18.84 dB):

| arm | step | drive vs baseline | band energy | SIM-o | UTMOS | WER |
| --- | --- | --- | --- | --- | --- | --- |
| `adapter_drift` | 3000 | 0.42 | +0.76 dB | 0.382 | 3.22 | 3.02% |
| `adapter_drift` | 6000 | 0.41 | +2.07 dB | 0.367 | 3.35 | 3.02% |
| `adapter_driftn` | 6000 | 0.86 | +19.8 dB | 0.179 | 4.09 | 3.09% |
| untermed control | 6000 | 1.00 | +12.17 dB | 0.231 | 3.88 | 3.49% |

The two termed arms sit near opposite ends of the same trade-off between band
energy and identity that `--residual-scale` traces on a finished model: 0.382
SIM-o at +0.76 dB, and 0.179 at +19.8 dB. Read as a pair they price the
unconstrained system — roughly 20 dB of band energy costs about 0.2 of SIM-o —
and that is the trade any working constraint has to beat. The control sits
between them, which is the point: neither arm is a small perturbation of it.

They do not sit exactly on that curve, though, and the difference is worth
recording. Compared against the dose sweep at *matched* SIM-o, every arm is
slightly above it:

| arm | SIM-o | UTMOS | dose curve at that SIM-o | offset |
| --- | --- | --- | --- | --- |
| `adapter_drift` 3k | 0.382 | 3.22 | ≈3.06 (`dose025`) | +0.16 |
| `adapter_drift` 6k | 0.367 | 3.35 | ≈3.26 (interpolated) | +0.09 |
| `adapter_driftn` 6k | 0.179 | 4.09 | ≈3.87 (interpolated) | +0.22 |

So the training term buys something the inference dial cannot, consistently and
in the right direction. On its own that offset is too small to reach the gate:
the dose curve reaches the 0.34 SIM-o gate at roughly UTMOS 3.55, so even
carrying the largest offset the whole way there lands near 3.77, under the 3.9
gate, with band energy at that dose nowhere near +10 dB.

One lever was left: the arms above are read mid-schedule, and identity is what
the untermed branch spends its later steps acquiring (0.231 at 6k → 0.246 at
30k, at flat band energy). `adapter_driftn` entered that stretch 7.7 dB higher
and 0.052 SIM-o lower than the control did. It was resumed from
`checkpoint-6000` to its scheduled 10000 steps to see whether it follows the
same identity recovery. It does not.

| `adapter_driftn` | band energy | SIM-o | UTMOS | WER (seed-tts) |
| --- | --- | --- | --- | --- |
| step 6000 | +19.84 dB | 0.179 | 4.09 | 3.09% |
| step 10000, curriculum-complete | +23.40 dB (−7.03) | **0.161** | **3.87** | 2.42% |
| untermed baseline, 30k, curriculum-complete | +11.6 dB | 0.246 | 3.96 | 3.41% |

Identity moves the wrong way over the stretch that lifts the untermed branch's,
and UTMOS falls off the 3.9 gate it had cleared at 6000 while band energy runs
another 3.6 dB past a figure already double the baseline's. Curriculum-complete
against curriculum-complete the arm is **+11.8 dB of band energy for −0.085
SIM-o and −0.09 UTMOS**. The pre-registered falsifier asked for SIM-o to rise;
it fell. WER is the one number that improved (2.42% against the baseline's
3.41%), which is consistent with a branch writing more high-band detail — it is
intelligibility, not identity, and identity is what the term exists to protect.

Validation drive tells the same story from the training side: the ratio against
the untermed branch at matched steps ran 0.86 (6k), 0.85 (8k), 0.75 (10k). The
normalised term holds drive far better than the unnormalised form's 0.41, but it
is not scale-neutral in practice over a full schedule either, and the energy it
does produce is not the kind that preserves the speaker.

**The drift term is finished as a route to the gate, in both forms.** The
unnormalised form brakes; the normalised form drives hard in a direction that
costs identity monotonically. Penalising the low-band response of the residual
does not, by itself, make the residual speaker-consistent — it constrains where
the residual is *not* allowed to act, and leaves what it writes unconstrained.
The remaining levers are the ones that constrain the written band positively:
the adversarial and multi-resolution band losses.

Two notes for anyone rerunning these. Drive is validation `residual_rms`, never
the training figure, which is a cumulative epoch average and lags badly: at step
5000 of `adapter_drift` the training mean moved 0.03321 → 0.03389 while
validation moved 0.03049 → 0.03782 → 0.04462. And the normalised weight is not
transferable from the unnormalised one — 0.05 was chosen from the unnormalised
term's magnitude, but once past the small-`r` region the normalised term settled
near 3% of the total loss, under the 10–30% band the other terms are calibrated
to.

One caveat on the SIM-o column. Both arms train with the pretrained speaker
stream (`--use-pretrained-spk 1`), and inference must be handed the same encoder
or the branch is given a zero embedding flagged invalid — an input training only
ever saw for unusably short prompts. These two arms were scored with
`--speaker-encoder-checkpoint` passed to stage 2. Arms scored before that flag
was added there were not, so their SIM-o understates them and is not comparable
to this table.

## Evaluation

76 LibriTTS utterances, one prompt each. Prompts are degraded by resampling to
the target rate and back to 24 kHz, so the degradation is a real resampling chain
rather than the mel-domain band-limiting used in training. Every arm is scored
against `test_score.tsv`, which points at the **original 24 kHz** prompt: scoring
against the degraded prompt would measure the degradation instead of the
synthesis and would flatter every system equally.

| metric | model |
| --- | --- |
| SIM-o | WavLM-large ECAPA-TDNN, cosine similarity to the clean prompt |
| WER | Whisper-large-v3 (`wer`) and the Seed-TTS protocol (`wer_seedtts`) |
| UTMOS | UTMOS22 strong |
| DNSMOS | P.835 overall and P.808 |

Per-utterance scores are written beside the audio (`sim_scores.tsv`,
`dnsmos.tsv`) so arms can be compared with paired tests; the unpaired standard
error of a SIM-o mean at n=76 is 0.0099, which is the floor any claimed gap must
clear.

Inference uses 16 solver steps throughout. Baselines:

- **stock** — the released ZipVoice, prompted with the degraded audio directly.
- **cascade** — an off-the-shelf bandwidth extender applied to the prompt
  waveform, then the released ZipVoice. This is the system the adapter has to
  beat to be worth its parameters.

## Reproducing

```bash
cd egs/zipvoice
# stages 1-4 and 8 of run_bwe_libritts.sh first: features and the eval set
stage=1 stop_stage=3 bash run_adapter_libritts.sh          # train, synthesise, score
# dose / decoupling sweep at inference, no retraining:
python -m zipvoice.bin.infer_zipvoice --model-name zipvoice_adapter \
    --model-dir /c/zipvoice_data/exp/adapter/ \
    --checkpoint-name iter-30000-avg-2.pt --tokenizer emilia \
    --test-list /c/zipvoice_data/data/bwe_eval/test_sr8000.tsv \
    --res-dir /c/zipvoice_data/exp/results/decoup_a05_sr8000 \
    --num-step 16 --residual-scale 0.5 --decouple-residual 1
```

## Results

The reported configuration is the branch run with `--decouple-residual 1`, at a
residual scale `a` swept over 0.25, 0.5, 0.75, 1.0. Full per-rate tables are in
`exp/results/decoup_sr{8000,16000,22050}.csv`; `a` is blank for the baselines,
which have no residual.

**8 kHz prompt**

| system | a | SIM-o | WER | UTMOS | DNSMOS ovrl | P.808 |
| --- | --- | --- | --- | --- | --- | --- |
| cascade | | 0.563 | 3.08 | 4.14 | 3.336 | 3.970 |
| stock | | 0.610 | 3.02 | 3.89 | 3.314 | 3.539 |
| adapter | 0.25 | 0.609 | 3.71 | 3.97 | 3.300 | 3.570 |
| adapter | 0.50 | 0.610 | 3.53 | 4.00 | 3.297 | 3.593 |
| adapter | 0.75 | 0.612 | 3.08 | 4.03 | 3.310 | 3.597 |
| adapter | 1.00 | 0.611 | 3.71 | 4.03 | 3.313 | 3.610 |

**16 kHz prompt**

| system | a | SIM-o | WER | UTMOS | DNSMOS ovrl | P.808 |
| --- | --- | --- | --- | --- | --- | --- |
| cascade | | 0.727 | 3.02 | 4.22 | 3.346 | 3.983 |
| stock | | 0.730 | 2.74 | 4.08 | 3.346 | 3.969 |
| adapter | 0.25 | 0.728 | 2.96 | 4.12 | 3.315 | 3.963 |
| adapter | 0.50 | 0.731 | 2.85 | 4.19 | 3.316 | 3.963 |
| adapter | 0.75 | 0.731 | 2.91 | 4.20 | 3.326 | 3.963 |
| adapter | 1.00 | 0.733 | 2.79 | 4.23 | 3.330 | 3.966 |

**22.05 kHz prompt**

| system | a | SIM-o | WER | UTMOS | DNSMOS ovrl | P.808 |
| --- | --- | --- | --- | --- | --- | --- |
| cascade | | 0.739 | 2.96 | 4.22 | 3.336 | 3.982 |
| stock | | 0.741 | 2.96 | 4.17 | 3.334 | 3.969 |
| adapter | 0.25 | 0.733 | 2.57 | 4.19 | 3.315 | 3.968 |
| adapter | 0.50 | 0.733 | 2.57 | 4.20 | 3.315 | 3.965 |
| adapter | 0.75 | 0.733 | 2.45 | 4.20 | 3.313 | 3.965 |
| adapter | 1.00 | 0.732 | 2.57 | 4.20 | 3.305 | 3.970 |

### Reading

**Decoupling removes the identity cost without removing the quality gain, at
every rate except 4 kHz.** The coupled branch lost 0.093 SIM-o at 8 kHz and 0.019
at 16 kHz. Decoupled, SIM-o is within 0.003 of stock at both — inside the 0.0099
unpaired standard error at n=76 — while UTMOS rises 0.14 at 8 kHz and 0.15 at
16 kHz. At 16 kHz and `a = 1.0` the arm is at or above stock, cascade and the
coupled branch on both axes at once (0.733 SIM-o, 4.23 UTMOS), which no earlier
arm in the study achieved.

**The dose is monotone in quality and flat in identity.** Unlike the coupled
model, where UTMOS and SIM-o traded one for one across the dose sweep, the
decoupled model buys UTMOS with `a` at no measurable identity cost. `a = 1.0` is
the operating point; nothing in the range argues for backing the dose off.

**4 kHz is the exception, and it is where the mechanism shows.** There,
decoupling restored SIM-o fully (0.246 → 0.368) but gave back most of the
quality (UTMOS 3.96 → 3.24, P.808 3.674 → 3.239). At 4 kHz the branch is being
asked to invent five sixths of the spectrum, and at that reach its contribution
is mostly the *trajectory bending* — pushing the frozen backbone somewhere it
generates more confidently — rather than the band it writes. At 8 kHz and above,
the written band is itself worth something, so decoupling keeps the gain.

**Against cascade.** At 8 kHz the adapter is ahead on identity by 0.048 with
UTMOS 0.11 behind; at 16 kHz it is ahead on both; at 22.05 kHz the three systems
are within noise of each other on identity and the adapter has the lowest word
error (2.45–2.57 against 2.96). The adapter needs no separate extender model at
inference and is algebraically the released checkpoint on full-band prompts.

**A caveat on 22.05 kHz.** All four doses give SIM-o 0.732–0.733 against stock's
0.741, and the WER numbers repeat exactly across doses, which is what a very
small residual looks like: the band mask covers few bins there, so the arms are
nearly the same audio. The 0.008 shortfall is at the noise floor, not below it,
and should be reported as "matches stock" rather than as a loss.

## The band-energy metric was reading one octave, not the band

Every "+N dB" in this document up to here came from `local/band_energy.py`, which
reports a single hi/lo power ratio over everything above the cutoff. Speech power
falls steeply with frequency, so that ratio is dominated by the first octave
above the cutoff, and a system that fills 2–3 kHz and nothing else scores close
to one that fills 2–12 kHz. The joint gate's "energy ≥ +10 dB over stock" was
being applied through that lens.

Breaking the high band into octaves, each system measured relative to its own
100–2000 Hz band, at a 4 kHz cutoff (n = 76):

| system | 2-3k | 3-4k | 4-6k | 6-8k | 8-12k |
| --- | --- | --- | --- | --- | --- |
| references | -16.7 | -19.9 | -18.0 | -19.9 | -22.6 |
| cascade | -16.3 | -20.2 | -19.7 | -23.0 | -26.0 |
| `adapter_driftn10k` | -13.7 | -15.5 | -11.4 | -12.2 | -15.3 |
| `adapter` | -18.4 | -29.0 | -27.0 | -29.5 | -32.3 |
| `adapter_band` | -19.1 | -29.9 | -27.6 | -29.5 | -32.2 |
| decoupled, a=1.0 | -16.8 | -40.9 | -46.9 | -53.6 | -57.0 |
| stock | -28.4 | -53.1 | -55.0 | -60.9 | -64.4 |

Three readings change.

**The decoupled arm does not clear the energy gate.** Its +10.5 dB hi/lo ratio
is almost entirely a narrow lift just above the 2 kHz cutoff; from 3 kHz up it
sits 21–34 dB below the references and only 2–8 dB above stock. It is close to
empty where the band matters. Its UTMOS 3.24 is therefore a *muffled* failure,
not an artifact-y one, and its SIM-o 0.368 is cheap: a branch that writes almost
nothing cannot move the low band, so identity is preserved by inaction. The
paragraph above reading 4 kHz decoupling as "trajectory bending" that costs
quality should be read with this in mind — most of the quality loss is simply
missing high band.

**The coupled branch writes the right shape at the wrong level.** `adapter` is
within a few dB of the references' *contour* but sits 9–10 dB under it
everywhere above 3 kHz. That is not a conditioning failure or a capacity
failure. Nothing in the objective asks for a level.

**`adapter_driftn10k` is the only arm that writes more than the cascade**, by
4–7 dB over the references in every band above 3 kHz. The normalised drift term
did drive genuine broadband synthesis; it simply overshot into noise and
destroyed identity doing it. That is a more informative failure than the SIM-o
number alone suggested.

Judge future arms per octave. The single ratio cannot tell "filled the band"
from "lifted the octave above the cutoff".

## The multi-resolution band loss: a null, and what it rules out

`--band-loss-weight` pools the endpoint error over tiles of frames and mel bins
*before* the absolute value, at three scales (8x5, 32x10, 128x20 frames by bins,
roughly 85 ms / 340 ms / 1.4 s). Error that cancels inside a tile costs nothing,
so only each tile's net level offset is charged and the unpredictable fine
structure of a high band is not penalised for being wrong. It was built to
attack the 9–10 dB deficit directly, since that is a level and nothing else in
the objective is a level.

Calibrated the usual way: the term is always computed and logged, so a run at
weight 0 measures it. `band_loss` 0.0365 against `recon_loss` 0.0912 on a total
of 0.162, so weight 1.0 puts it near 18%, inside the 10–30% band the other added
terms are sized to. `adapter_band` is a 10k finetune off `adapter`'s
`iter-30000-avg-2.pt` at that weight, `base_lr 0.0005`, `curriculum_rho 0.0`.

It never descended:

| step | 1000 | 4000 | 7000 | 10000 |
| --- | --- | --- | --- | --- |
| `band_loss` | 0.04177 | 0.03989 | 0.04193 | 0.04267 |
| `residual_rms` | 0.1426 | 0.1381 | 0.1422 | 0.1494 |

Paid 18% of the objective for 10k steps and finished 2% above where it started.
Evaluation at 4 kHz: SIM-o **0.242**, seed-tts WER 3.71%, WER 2.68%, UTMOS
**3.95**, DNSMOS OVRL 3.280 / P808 3.691, against `adapter`'s 0.246 / 3.41% /
2.91% / 3.96 / 3.285 / 3.674. Every number inside the 0.008 noise floor, and the
octave profile 0.6–0.9 dB *down* in the three bands that moved at all.

**Why, and this is the part that generalises.** `band_bias` -- the same tile
offsets without the absolute value, added alongside `band_loss` -- says there is
no level bias at training time to remove. Six validation batches on the baseline
checkpoint:

| `band_loss` | 0.0379 | 0.0329 | 0.0525 | 0.0367 | 0.0847 | 0.0384 |
| --- | --- | --- | --- | --- | --- | --- |
| `band_bias` | 0.0041 | 0.0016 | 0.0341 | 0.0022 | 0.0709 | 0.0062 |

Positive in all six -- one step from ground truth the endpoint estimate is if
anything slightly *loud*, never quiet -- and in four of six the signed mean is
5–16% of the absolute mean, so most of what the term charges is unbiased spread.
Penalising an unbiased offset pulls the prediction toward each tile's mean, which
is shrinkage: the same brake mechanism as the unnormalised drift term, and the
0.6–0.9 dB drop in the profile is it.

So the 9–10 dB deficit **does not exist one step from clean features**. It is
produced by the 16-step solve: the branch writes into a velocity, the solver
integrates it, the frozen backbone reads the moved state and responds, and the
attenuation accumulates across steps. Every loss in this study scores
`x1_hat = xt + (1-t)*vt` and is therefore blind to it -- `recon_loss`,
`band_loss`, `variance_loss`, the identity and feature-matching terms on the
vocoded endpoint estimate, and the adversarial critic. That last one is worth
restating: `adapter_gan`'s null is not evidence that adversarial pressure fails,
it is evidence that a critic judging `x1_hat` is judging a spectrum which does
not have the defect.

The drift term is the only one that ever touched the mechanism, through a single
finite-difference probe of the backbone's response, and it is the only one that
moved the octave profile at all -- past the references, at the cost of identity.

**What remains.** Reaching this defect means training against the solve rather
than against a single step: a rollout, or a term on the accumulated response
over more than one step. That is expensive in memory, and it is also where the
coupling constraint bites hardest, since the cheap ways to stabilise a rollout
are the decoupled ones this design exists to avoid. No further single-step loss
term is worth a run.

## The state the branch conditions on, measured

The single-step blind spot above says the level deficit is produced by the
solve and does not exist one step from clean features. The obvious test is to
build the state the solver actually produces and measure the same quantities
there. A detached rollout does that cheaply: run the Euler solver from noise
under `no_grad` for `k` of 16 steps with the branch in the loop, hand the
resulting state to the loss, and take one attached step from it. Memory is one
activation peak rather than `k`, and the arm is not a decoupled trajectory --
the branch writes into every rollout step, it simply does not receive gradient
through them. On such a state the velocity target has to become the re-flow
target `ut = (features - xt) / (1 - t)`, because the state is no longer on the
straight line through the sampled noise.

Run at `--band-loss-weight 0` from the trained adapter checkpoint, 60 iterations
each, identical data order, cumulative averages at step 60:

| k  | fm_loss | recon_loss | band_loss | band_bias | residual_rms |
|----|---------|------------|-----------|-----------|--------------|
| 0  | 0.067   | 0.092      | 0.036     | -0.0006   | 0.121        |
| 4  | 0.071   | 0.178      | 0.096     | +0.0063   | 0.100        |
| 8  | 0.175   | 0.187      | 0.099     | +0.0038   | 0.082        |
| 12 | 0.755   | 0.190      | 0.104     | +0.0024   | 0.068        |

Three things come out of this, and only one of them was predicted.

The prediction was wrong. `band_bias` was supposed to go strongly negative on
rolled-out states -- a systematic level deficit that the flow path hides. It
does not. It is positive at every `k` and it *shrinks* as `k` grows, while
`band_loss` grows. The endpoint error on a rolled-out state is large and close
to zero-mean over tiles, which is the same shrinkage-not-bias verdict the
flow-path diagnostic returned, and it means a tile-level loss is no more the
right lever here than it was there. The multi-resolution band term should not
be revived on rolled-out states.

The blind spot itself is real and rollout removes it. `recon_loss` doubles from
0.092 to 0.19 and `band_loss` nearly triples, on the same weights and the same
data. That error is not visible anywhere on the flow path. Whatever the right
objective is, it has something to work with here and had nothing to work with
before. The `fm_loss` column is not evidence of the same thing: `1 - t` is 0.25
at `k = 12`, so the re-flow target is divided by a small number and the squared
error is amplified about sixteenfold. Most of 0.067 to 0.755 is that factor,
not state degradation. `recon_loss` and `band_loss` score `x1_hat` directly and
carry no such factor.

The direct measurement of the shortfall is `residual_rms`. The branch writes
0.121 on the flow path and 0.068 -- 44% less -- on a state it produced itself
twelve steps in, with no weight change between the two numbers. This is the
inference-time deficit observed at its source, and it is a property of the
branch's *response to its own input distribution*, not of any loss. It also
explains why the endpoint statistics stay unbiased: the model is not predicting
a too-quiet endpoint from the quiet state, it is predicting a plausible
endpoint via a large one-step extrapolation that the 16-step solve never takes.

What this leaves is an arm, not a term: train on mixed flow-path and rolled-out
states with the existing two-term objective and scheduled sampling
(`--rollout-prob` below 1.0, warm-started from the trained adapter, so the
branch is not learning from its own states from scratch as the earlier rollout arm did,
which collapsed the written band to +1.9 dB over stock). The lever is exposure, not a new loss.

## Scheduled residual rescaling at inference (training-free)

The rollout diagnostic showed `residual_rms` falling 44% between the flow path and
a rolled-out state, so the branch writes less at sampling time than it was trained
to write. The cheapest possible response is to multiply it back up during the solve
and see how far that gets, without retraining anything. That is what the
`--residual-scale-*` flags do: a scalar dose, optionally ramped across `t`, and
optionally log-frequency-tilted so the gain differs per mel bin.

The tilt is not decoration. Dose sensitivity is frequency dependent and strongly
super-linear in frequency: 2.5x buys +9.3 dB at 4-6 kHz (1.2x the nominal dB) while
2.0x buys +16.3 dB at 8-12 kHz (2.7x nominal). Correcting a roughly even deficit
therefore needs a gain that *falls* with frequency. Scaled bins also re-enter the
state that every later step reads, so a band raised at 6 kHz lifts its neighbours
at a gain near 1.0 (hi20 lifted 3-6 kHz by 4.5-5.9 dB without being applied there).

Per-octave energy at a 4000 Hz test rate, each system relative to its own
100-2000 Hz band, n=76 (`local/band_profile.py`):

```
system                     0-2k    2-3k    3-4k    4-6k    6-8k   8-12k
references                  0.0   -16.7   -19.9   -18.0   -19.9   -22.6
cascade                     0.0   -16.3   -20.2   -19.7   -23.0   -26.0
adapter (dose 1.0)          0.0   -18.4   -29.0   -27.0   -29.5   -32.3
hi15 (1.5 above 4.2k)       0.0   -17.6   -26.3   -21.2   -21.3   -22.0
b20w (2.0 over 4.2-8k)      0.0   -17.9   -26.6   -18.0   -17.7   -25.3
tilt3 (2.0@3k -> 1.37@10k)  0.0   -16.8   -21.8   -19.5   -20.7   -22.8
```

`tilt3` lands within 0.1-1.9 dB of the references in every octave and beats the
cascade above 4 kHz (-20.7 vs -23.0 at 6-8 kHz, -22.8 vs -26.0 at 8-12 kHz), from
the stage-4 checkpoint with no additional training. Reference-matched energy is
reachable training-free; that much is settled.

It is not an answer, because the gate is joint:

```
arm      dose                  SIM-o  UTMOS  seedtts WER   WER   DNSMOS OVRL/P808
gate                           >=0.34 >=3.9
adapter  1.0                   0.246  3.96   3.41%       2.91%  3.285 / 3.674
hi15     1.5 above 4.2k        0.198  3.99   2.94%       2.85%  3.290 / 3.709
b20w     2.0 over 4.2-8k       0.190  3.89   3.23%       3.02%  3.269 / 3.646
tilt3    2.0@3k -> 1.37@10k    0.174  3.83   3.59%       3.08%  3.271 / 3.692
```

Two things worth separating here. Naturalness is *not* what raising the band costs:
hi15 adds 8-11 dB above 4 kHz and its UTMOS goes up, 3.99 against the base arm's
3.96, with DNSMOS flat and WER slightly better. Only identity pays. SIM-o falls
monotonically with the dose, 0.246 -> 0.198 -> 0.190 -> 0.174, and every point sits
far under the 0.34 leg.

So rescaling reproduces, training-free, exactly the energy/identity curve that every
trained arm in this study has ridden. It buys a cheaper way to move along that curve,
not a way off it. If the deficit were purely exposure bias -- the branch writing the
right thing at the wrong amplitude -- then restoring the amplitude should have been
close to free in SIM-o, and it was not. Whatever the branch writes at sampling time
is not merely a quiet version of what it writes on the flow path.
