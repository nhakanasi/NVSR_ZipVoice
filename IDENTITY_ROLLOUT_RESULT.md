# Identity supervision on the sampled output: 2000-update result

Run `adapter_rollout_20260920`, 2026-09-20. Started from the pre-GAN
`adapter_shallow_recon_20260918/checkpoint-2000.pt`, 2000 updates,
`--adapter-trainable shallow`, GAN off, ordinary flow-matching and
reconstruction unchanged, checkpoints at 500 / 1000 / 2000.

The new term: on 10% of batches the model runs the deployed solve (16 steps,
guidance 0.5, t_shift 0.5, coupled residual) on one randomly chosen utterance
that fits an 8 s cap, and the speaker identity loss is charged on that sample
at weight 0.15. The first 15 steps run under `no_grad`; only the last carries a
graph, so the term teaches a local correction at the end of the trajectory and
cannot assign credit to earlier steps. The backbone is detached for the whole
solve, verified by the smoke check: 369 backbone tensors with `grad` None,
gradient confined to 30 branch tensors and 29 conditioning-encoder tensors.

Because ScaledAdam substitutes zeros for a missing gradient and applies its
momentum step to every parameter it owns, stepping on a rollout batch would
still drift the frozen backbone (measured: 3.19e-03 of movement on a parameter
whose gradient was None). `--identity-rollout-accumulate` (default on) therefore
holds the rollout gradient and folds it into the next ordinary update, which is
what keeps the exclusion exact.

The run also switched degradation from the mel floor to real waveform
resampling (`--resample-degrade 1`), matching the evaluation prompts.

## Numbers

SIM-o / UTMOS, 76 utterances, `--num-step 16 --guidance-scale 0.5 --t-shift 0.5
--project-low-band 0 --prompt-floor 0 --residual-scale 1.0`.

| system | sr8000 | sr16000 | sr24000 |
| --- | --- | --- | --- |
| stock (earlier run) | 0.610 / 3.89 | 0.730 / 4.08 | - |
| cascade (earlier run) | 0.563 / 4.14 | 0.727 / 4.22 | - |
| checkpoint-2000 before this run | 0.572 / 4.15 | 0.694 / 4.26 | 0.690 / 4.21 |
| 500 updates | 0.430 / 4.15 | 0.641 / 4.26 | 0.662 / 4.22 |
| 1000 updates | 0.438 / 4.18 | 0.625 / 4.30 | 0.646 / 4.29 |
| 2000 updates | 0.428 / 4.21 | 0.604 / 4.26 | 0.639 / 4.27 |

8-12 kHz energy, dB relative to each system's own 100-2000 Hz band:

| system | sr8000 | sr16000 | sr24000 |
| --- | --- | --- | --- |
| stock | -54.3 | -34.4 | -22.1 |
| before this run | -52.1 | -34.3 | -28.2 |
| 2000 updates | -13.8 | -12.2 | -23.5 |

Speech frames against silence frames, sr8000, 8-12 kHz: the clean reference is
-43.0 speech / -52.3 silence with a 34.4 dB swing; this run is -42.7 / -64.2
with a 35.1 dB swing. The band tracks speech rather than lying over the whole
file, and its speech-frame level lands on the reference.

Neither intelligibility nor the DNSMOS proxies deteriorated alongside the band.
At sr8000, Seed-TTS WER 3.28% -> 2.75%, DNSMOS SIG 3.568 -> 3.583, BAK 4.115 ->
4.089, P808 3.608 -> 3.919. These measure no corresponding deterioration; they
do not establish that the new high band is free of crackle. That question is
settled by listening to the matched pairs, which has not been done.

## What it means

The combined recipe failed on identity: SIM-o fell at every rate at every
checkpoint, and the rollout's own training value rose over the run, from about
0.50-0.54 early to 0.60-0.61 late. What the identity term contributed to that is
*unknown*. The examples it was scored on changed over the run and it competed
with an unchanged reconstruction objective, so a rising training value does not
isolate its effect; the term could equally have reduced a larger deterioration
caused by something else in the recipe. Nothing here licenses the claim that
identity supervision on the sampled output is itself harmful.

The bandwidth result is the opposite, and is almost certainly the degradation
change rather than the rollout: real resampling removes a train/test mismatch of
about 8.7 log-units in the dead band, and it is the first time this arm has
produced a genuine high band at a 4 kHz cutoff (+40 dB over stock at sr8000,
+22 dB at sr16000) with reference-like speech/silence structure.

## The control settles it

A second run of 2000 updates with `--identity-rollout-weight 0` and nothing else
changed. Matching it took a code change: the weight used to gate the rollout
itself, so weight zero removed the `torch.rand(())` draw, shifted the RNG stream
and changed every subsequent cutoff sample. The rollout now fires on
configuration -- `--identity-rollout-steps` and `--identity-rollout-prob` -- and
the weight only scales the update. At weight zero the identity value is still
computed and logged, under `no_grad`, because that value is the counterfactual.

The match is verified rather than assumed. `MetricsTracker` prints the running
`(fm, rollout)` frame split, which covers every batch and not only the logged
ones, and it is identical in the two runs at batches 45, 60, 500, 1000, 1500 and
2000 -- at 2000, `fm 1076250.37, rollout 146883.92` in both. Same batches, same
classification of each batch, same draws; the runs differ in the update and in
nothing else.

The result is that the identity term did nothing.

| checkpoint | rate | treated SIM-o / UTMOS | control SIM-o / UTMOS |
|---|---|---|---|
| 500 | sr8000 | 0.430 / 4.15 | 0.427 / 4.11 |
| 500 | sr16000 | 0.641 / 4.26 | 0.645 / 4.28 |
| 500 | sr24000 | 0.662 / 4.22 | 0.664 / 4.25 |
| 1000 | sr8000 | 0.438 / 4.18 | 0.441 / 4.18 |
| 1000 | sr16000 | 0.625 / 4.30 | 0.624 / 4.28 |
| 1000 | sr24000 | 0.646 / 4.29 | 0.648 / 4.29 |
| 2000 | sr8000 | 0.428 / 4.21 | 0.430 / 4.21 |
| 2000 | sr16000 | 0.604 / 4.26 | 0.600 / 4.27 |
| 2000 | sr24000 | 0.639 / 4.27 | 0.634 / 4.28 |

Largest SIM-o gap 0.005, largest UTMOS gap 0.04, signs mixed. The identity proxy
agrees: smoothed `rollout_identity_loss` runs 0.5624 / 0.5701 / 0.5918 / 0.6067
in the control at batches 1200 / 1500 / 1800 / 2000 against 0.5673 / 0.5693 /
0.5910 / 0.6056 in the treated run. The rise from about 0.50 to about 0.61
happens identically with the term switched off, so it was never evidence about
the term.

The band is the same too:

| 8-12 kHz, dB re. own 0-2 kHz | stock | before | treated | control |
|---|---|---|---|---|
| sr8000 | -54.3 | -52.1 | -13.8 | -13.7 |
| sr16000 | -34.4 | -34.3 | -12.2 | -12.1 |

and so is the speech/silence split (sr8000, 8-12 kHz: treated -42.7 / -64.2,
swing 35.1; control -42.8 / -64.0, swing 35.6).

So within this pair, `--resample-degrade` owns both halves: the whole +40 dB
band gain and the whole SIM-o collapse. The identity rollout at weight 0.15 with
this gradient routing is a no-op -- it moved neither the metric it optimises nor
any metric it could have traded against.

That statement is about this pair only. An earlier checkpoint, `adapter30k`,
trained with `resample_degrade: false` and still reaches 8-12 kHz energy far
above stock, so the flag is not what makes band extension possible. What it
produced here is band *past* the target, and the identity cost that came with it.

## Low-band projection recovers a little, and it is not the mechanism

`--project-low-band 1` pins the output's low band to the backbone at inference.
On `adapter30k` it had been worth +0.101 SIM-o, so it was the obvious free lever.
Applied to these checkpoints it is free but small (n=76, sr8000 / sr16000):

| system | SIM-o | UTMOS | 8-12 kHz speech frames, sr8000 |
|---|---|---|---|
| stock | 0.610 / 0.730 | 3.89 / 4.08 | -78.9 |
| cascade | 0.563 / 0.727 | 4.14 / 4.22 | -46.8 |
| rollout2000 | 0.428 / 0.604 | 4.21 / 4.26 | -42.7 |
| rollout2000 + proj | 0.470 / 0.620 | 4.20 / 4.29 | -41.1 |
| ctrl2000 + proj | 0.466 / 0.613 | 4.21 / 4.27 | -- |

Projection costs no bandwidth -- the high band rises 1.6 dB and the silence floor
drops 1.8 dB -- and it lifts SIM-o by 0.042 at sr8000 and 0.016 at sr16000. It
closes under a quarter of the gap to stock and leaves about 0.10 to cascade at
both rates. So most of the identity damage is not low-band drift, and it is not
reachable at inference time. The treated/control gap stays at 0.004 and 0.007
under projection, so the identity term remains a no-op here too.

## A caution about the two band tools

`band_profile.py` averages power across frames; `band_frames.py` averages dB
within speech frames. They rank systems differently when a system's energy is
concentrated in a minority of loud frames, and `adapter30k` is such a system:
near cascade on the power measure, 21 dB below cascade on speech frames. Cascade
itself scores near the reference on both, which is what shows the per-frame tool
is not at fault. Band claims should quote the per-frame speech figure, and this
report's earlier reading of `rollout2000` as "reference-like" holds only there --
on the power measure it sits about 10 dB above the reference, i.e. too bright.

## The gradient restriction this implementation carries

Ordinary batches update the shallow-trainable backbone parameters. Identity
batches cannot reach them: `_detach_backbone` freezes the backbone for the whole
solve, so the identity gradient lands on the branch and the conditioning
encoders through the final step alone. The branch's output is then
frequency-masked, and at a 4 kHz cutoff most of what it writes sits above the
speaker encoder's bandwidth, so its only route to that encoder is indirect,
through the vocoder.

So the run asked a restricted final-step correction to hold the voice steady
while the ordinary objective moved the parameters that carry it. Whatever the
control returns, the next implementation should let the final identity step
backpropagate through the shallow-trainable backbone parameters as well, keeping
the first `num_step - 1` steps detached. That gives the identity objective direct
access to the parameters the ordinary objective is changing without storing a
full 16-step graph. It remains truncated credit assignment and may still not be
enough, but it removes a concrete imbalance rather than adding another loss.

Against the acceptance criteria: UTMOS beats cascade at both degraded rates
(4.21 vs 4.14, 4.26 vs 4.22); SIM-o misses stock badly (0.428 vs 0.610 at
sr8000); the mel images carry real structure to 12 kHz rather than noise or
smear.

## The arm's output: projgan

The identity rollout is not the best point this arm reached. That is a
GAN-finetuned adapter run with inference-time low-band projection, referred to
here as `projgan`, and until now it appeared in no script, report or commit.

Identity, from `results/projgan_sr8000.log`:

- model dir `exp/adapter_gan` under the data root, checkpoint `iter-10000-avg-2.pt`
- `--model-name zipvoice_adapter --tokenizer emilia`
- `--num-step 16 --t-shift 0.5 --project-low-band 1 --prompt-floor 0 --residual-scale 1.0`
- `--guidance-scale` left at its default of 1.0, not the 0.5 used elsewhere in
  this report. That difference is untested and is a confound in any comparison
  against the rollout checkpoints.

Scores, 76 utterances, the same eval set and scripts as everything else here.
8-12 kHz figures are `band_frames.py` speech rows; swing is p90-p10 in that band.

| sr8000       | SIM-o | UTMOS | 8-12k speech | swing |
|--------------|-------|-------|--------------|-------|
| reference    | --    | --    | -43.0        | 34.4  |
| stock        | 0.610 | 3.89  | -78.9        | 10.7  |
| cascade      | 0.563 | 4.14  | -46.8        | 31.5  |
| projgan      | 0.608 | 4.15  | -57.1        | 40.8  |
| rollout2000  | 0.428 | 4.21  | -42.7        | --    |

| sr16000      | SIM-o | UTMOS | 8-12k speech | swing |
|--------------|-------|-------|--------------|-------|
| stock        | 0.730 | 4.08  | -60.5        | --    |
| cascade      | 0.727 | 4.22  | --           | --    |
| projgan      | 0.732 | 4.20  | -58.3        | 35.2  |

Against the acceptance criteria this is the only point in the arm that passes
all three. SIM-o matches stock (0.608 vs 0.610, 0.732 vs 0.730) and beats
cascade at sr8000 by 0.045. UTMOS matches cascade at both rates (4.15 vs 4.14,
4.20 vs 4.22) and beats stock by 0.26 at sr8000. The band is structural rather
than a veil: swing 40.8 is above both the reference's 34.4 and cascade's 31.5,
so the energy tracks speech instead of sitting as a constant hiss. Compare
`adv03_30000`, which reaches SIM-o 0.620 with swing 15.1 -- a static veil, and
the reason swing is quoted alongside level here.

Two shortfalls, stated plainly:

- Band level is 10.3 dB under cascade at sr8000 (-57.1 vs -46.8) and 14.1 dB
  under the reference. It is quieter than a real high band, just correctly
  shaped.
- At sr16000 it is close to inert: -58.3 against stock's -60.5 is +2.2 dB. The
  8 kHz-cutoff case is essentially unsolved; the arm's gain is concentrated at
  the 4 kHz cutoff.

### What projection costs

Projection is not free at inference. `_projected_sample` integrates **two**
trajectories from the same noise on the same timestep grid, and issues two
`solver.model` calls per step: a reference arm with `band_mask` all zero, whose
velocity is the frozen backbone's, and the adapter's own arm. Only after the
Euler update does the adapter state take the reference's low band back. So the
number of function evaluations doubles, 16 steps becoming 32 forwards.

The reference arm is not cheap either. `forward_fm_decoder` has no early-out for
an all-zero mask: the branch still runs, and its residual is zeroed afterwards by
the write mask. Both arms therefore cost backbone plus branch. An early-out there
is an obvious and untried saving.

Measured, same eval set, `--num-step 16`:

| | RTF |
|---|---|
| adapter, no projection | 0.185 |
| adapter + projection (guidance 1.0) | 0.372 |
| adapter + projection (guidance 0.5) | 0.434-0.444 |

A factor of two, as the structure predicts.

What is *not* doubled is the model. One weight set, one checkpoint, one copy in
memory -- the second trajectory reuses the same parameters. That is the
difference from the cascade, which loads a separate BWE model after the TTS pass:
AudioSR measures RTF 1.024 on `librispeech_pc_8000` and 1.062 on
`seedtts_en_8000`, roughly 1x real time on top of a 0.131 TTS pass. projgan at
0.376 is still about 3x cheaper end to end and holds one model resident, but the
honest statement is two solves of one model, not one solve.

The GAN itself adds nothing at inference: 0.3762 for projgan against 0.3715 for
the non-GAN adapter, both with projection. It is a training-time objective only.
