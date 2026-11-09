# Mel bandwidth extender for sampling-rate-agnostic prompts: results

Evaluation of `ZipVoiceBWE`, an NVSR-style mel bandwidth extender inserted in front of
ZipVoice's speech condition and trained jointly with the TTS model under the AnyBand
objective. Run on 2026-08-29 against LibriTTS dev-clean, and replicated on 2026-08-31
against VCTK 0.92 as a held-out out-of-domain test set. An application study comparing
plug-and-play deployment against joint fine-tuning was added on 2026-08-31, and the
tail analysis and the frozen-stage plug-and-play arm on 2026-09-01.

**Summary.** The extender substantially improves the perceived naturalness of speech
synthesised from band-limited prompts (UTMOS +0.33 at an 8 kHz prompt against stock
ZipVoice; +0.20 of that is attributable to the extender itself once the LibriTTS
fine-tuning is controlled for), but does not improve speaker similarity and costs −0.023
SIM-o at 8 kHz. The 24 kHz control does not significantly regress, so the guard against
damaging the clean-input case holds. Both effects reproduce on VCTK — 76 speakers the
model has never heard — at +0.23 UTMOS for −0.023 SIM-o, so this is a real property of
the extender and not an artefact of evaluating on the fine-tuning corpus. DNSMOS P.808
independently confirms the naturalness gain (+0.26 VCTK, +0.30 LibriTTS at 8 kHz), so it
is not an artefact of UTMOS either — though it also suggested the fine-tuning costs about
0.05 MOS on undegraded prompts.

**The application study revises two of those statements.** Bolting the trained extender
onto *unmodified* ZipVoice recovers about three quarters of the naturalness gain (P.808
+0.32 LibriTTS, +0.24 VCTK at 8 kHz) at no measurable speaker-similarity cost on unseen
speakers, where the jointly fine-tuned checkpoint gives up −0.036 SIM-o. Plug-and-play is
therefore the better deployment configuration **on the mean**. Looking at the lower tail
instead reverses the recommendation for one use case: the fine-tuned checkpoint has no
utterance below the rejection bar at an 8 kHz prompt on either corpus, where plug-and-play
still has one and stock has eight, and its advantage over plug-and-play on the baseline's
worst quartile is up to 1.7 times its advantage on the mean. Joint fine-tuning buys
worst-case robustness, and pays for it in speaker identity. That arm also supplies a
stricter null than the original control — identical weights rather than identical
input — which shows the noise floors were underestimated: UTMOS is nearer 0.05 than
0.01, and a known-zero effect reached p = 0.046 on SIM-o. The 8 kHz naturalness
results survive that easily; the 24 kHz P.808 fine-tuning penalty does not, and is
downgraded to suggestive. See **Application study 1**.

**That recommendation is scoped to 8 kHz and above.** At a 4 kHz prompt, where 83% of the
mel band has to be synthesised, the fine-tuned checkpoint is essentially rate-invariant
(UTMOS 4.27 and P.808 3.940, against 4.22 and 3.949 at 8 kHz) while stock ZipVoice falls to
UTMOS 2.80. Its advantage over plug-and-play grows roughly fourfold, to +0.34 / +0.22 P.808
at p = 1e−22 and 3e−11 — and it costs nothing, because SIM-o has collapsed to a floor for
every arm at that bandwidth and no contrast between them is significant. The crossover
between the two configurations therefore sits between 4 kHz and 8 kHz.

**Plug-and-play needs no training to deploy, but the extender it deploys comes out of the
joint adversarial stage.** Merging the extender from the end of the *frozen* stage instead
gives up 0.079 P.808 on LibriTTS and 0.051 on VCTK at an 8 kHz prompt (p = 0.000016 and
0.014), with nothing at any other rate and an exactly zero 24 kHz control — the two arms
produce bit-identical waveforms there, because the extender is provably inert at a Nyquist
cutoff. So the joint stage does two separable things, and both matter: it co-adapts the TTS
weights to the extender, and it improves the extender itself by an amount comparable to the
co-adaptation. The improvement to the extender survives transplant onto frozen released
weights and is available with no change to ZipVoice.

**Studies 2 and 3 are negative results about the recipe itself, and they simplify it.**
Removing the AnyBand cutoff curriculum, and separately removing every direct loss on the
extender output — reconstruction, all three discriminators, the adversarial and
feature-matching terms — changes nothing measurable in naturalness, intelligibility or the
failure tail on either corpus, and is marginally better on out-of-domain speaker
similarity. With `d_cross` and `d_harm` already shown dead across three runs, what
actually carries this method is two components: ZipVoice's own flow-matching gradient
through the frozen TTS, and the spectrogram discriminator. Dropping the rest also trains
1.7x faster. See **Application studies 2 and 3**.

## What was trained

Three stages, all on LibriTTS `train-clean-360` at 24 kHz, single GPU, `--max-duration 72`,
starting from the released Emilia ZipVoice checkpoint. Recipe: `run_bwe_libritts.sh`.

| Stage | Key flags | Iters | LR | Purpose |
|---|---|---|---|---|
| 5 | `--bwe-freeze-zipvoice 1 --bwe-fm-loss-weight 0 --bwe-gan-start-step -1` | 8000 | 1e-3 | Reconstruction-only pretrain of the ResUNet on our 100-bin 24 kHz mel. Replaces the released NVSR weights, which are 44.1 kHz / 128-bin and do not transfer. |
| 6 | `--bwe-freeze-zipvoice 1 --bwe-fm-loss-weight 1.0 --bwe-gan-start-step -1` | 30000 | 5e-4 | Extender trained against the frozen TTS. The flow-matching gradient teaches it to produce a mel the TTS finds useful, not merely one that looks right. Cutoff curriculum runs here. |
| 7 | `--bwe-freeze-zipvoice 0 --bwe-gan-start-step 0 --bwe-adv-loss-weight 0.1 --bwe-feat-match-loss-weight 2.0 --bwe-disc-lr 2e-4` | 15000 | 1e-4 | Joint adversarial fine-tuning, all three discriminators live, uniform cutoff sampling. |

Evaluated checkpoint: `iter-15000-avg-2.pt`, the 2-checkpoint average of the stage 7 tail.

## Evaluation protocol

`egs/zipvoice/local/prepare_bwe_eval.py` draws 76 prompt/target pairs from 38 dev-clean
speakers (breadth-first over speakers, at most 2 pairs each; prompts 3–10 s, targets at
least 40 characters). Each prompt is resampled down to 8000, 16000 and 22050 Hz and
written at that rate; the 24000 Hz condition is an untouched copy, deliberately not run
through a resampler so the control carries no transition band.

Degradation was verified empirically on one utterance, with every condition upsampled back
to 24 kHz before measuring band energy:

| condition | energy above 4 kHz | energy above 8 kHz |
|---|---|---|
| clean / sr24000 | 0.154130 | 0.0081 |
| sr22050 | — | 0.0068 |
| sr16000 | — | 0.00163 |
| sr8000 | 0.000379 | — |

The 8 kHz condition retains 1/400th of the energy above 4 kHz, and the 24 kHz control is
bit-identical to clean.

Every condition is scored against `test_score.tsv`, which points at the **original 24 kHz
prompt**, not the degraded one. Scoring against a degraded reference would measure the
degradation rather than the synthesis. Metrics are the repository's own evaluators: SIM-o
(`zipvoice.eval.speaker_similarity.sim`, WavLM-large ECAPA-TDNN), Seed-TTS WER
(`zipvoice.eval.wer.seedtts`, whisper-large-v3, `--lang en`), UTMOS
(`zipvoice.eval.mos.utmos`) and DNSMOS P.835 / P.808 (`zipvoice.eval.mos.dnsmos`).

Two systems are synthesised from identical prompts: `zipvoice_bwe` (the trained extender)
and stock `zipvoice` (the released Emilia checkpoint). The ablation and application-study
sections add further arms built from the same weights: `--bwe-bypass` variants, and a
`plugplay` merge of the trained extender onto the released ZipVoice weights.

## What the metrics score on natural speech

Every table below compares one arm against another, which establishes an ordering but not a
level. The 76 natural references built for the MCD evaluation supply the level: they are
human recordings of the same target sentences by the same speakers, and they were scored by
exactly the same evaluators against exactly the same prompts as every synthetic condition.

| metric | LibriTTS natural | VCTK natural |
|---|---|---|
| SIM-o | 0.740 | 0.726 |
| UTMOS | 4.11 | 4.06 |
| DNSMOS P.808 | 3.849 | 3.748 |
| DNSMOS SIG | 3.589 | 3.559 |
| DNSMOS BAK | 3.979 | 4.070 |
| DNSMOS OVRL | 3.260 | 3.280 |
| Seed-TTS WER | 5.13% | 1.11% |

Three things follow, and the first two change how the numbers above should be read.

**SIM-o near 0.74 is the ceiling, not a mediocre score.** SIM-o is a cosine between speaker
embeddings of two different recordings, so its upper limit is whatever two genuine recordings
of the same speaker achieve, and on this evaluation set that is 0.740. Stock ZipVoice at a
24 kHz prompt scores 0.739 on the same pairs. There is no headroom left on LibriTTS at all,
which is worth stating explicitly before anyone reads a SIM-o difference of 0.01 as a large
regression.

**The MOS predictors saturate above natural speech.** At a 24 kHz prompt stock ZipVoice
scores UTMOS 4.20 against natural speech's 4.11, and DNSMOS P.808 3.984 against 3.849; the
extender arm is 4.16 and 3.949. This does not mean synthesis is better than the recording. It
means both predictors are near the top of their useful range on clean read speech and mildly
prefer the smoother, less variable output of a vocoder. The practical consequence is that
UTMOS and P.808 headroom above roughly 4.1 and 3.85 carries no information, so only the
degraded conditions can separate systems at all — which is exactly the pattern the rate
ladder shows, and it is a property of the metrics rather than a limitation of the method.

**WER has no headroom either, for the same reason.** Whisper transcribes the natural
LibriTTS recordings at 5.13% and every synthetic arm at between 2.5% and 3.2%, so synthesis
is already well inside the error rate of the human speech it imitates. Real read speech has
disfluencies, breaths and reduced vowels that a vocoder's output does not. This retires WER
as a discriminating metric on this evaluation set more cleanly than the 0.6 pp noise floor
argument does: there is nothing left to measure above the floor set by the recordings
themselves. It remains worth reporting as a guard against a system that degrades
intelligibility, which is what the 4 kHz `resunet` collapse shows it catching.

**The VCTK gap is real, and it is not the metric being harder there.** Natural VCTK scores
SIM-o 0.726, only 0.014 below natural LibriTTS, so the ceiling barely moves between the two
corpora. Stock ZipVoice, meanwhile, drops from 0.739 to 0.622. The 0.10 shortfall on VCTK is
therefore the model failing to clone unfamiliar accents and a different recording chain, not
the corpus depressing the measurement. An earlier reading in this document attributed it to
the corpus shift acting on the metric; the natural-speech topline shows that is wrong. The
within-corpus contrasts are unaffected either way, since every arm pays the same 0.10.

## Control 1: how much of the 8 kHz SIM-o drop is the verifier hearing bandwidth

Stock ZipVoice loses 0.13 SIM-o on LibriTTS when its prompt is cut to 8 kHz. Part of that
could be the verifier reacting to a band-limited recording rather than the model losing the
speaker. To size that part, the 76 natural references per corpus were band-limited exactly as
the prompts were (`torchaudio.functional.resample` 24 kHz to the rate, then
`torchaudio.transforms.Resample` back, both at default settings) and scored with the same
SIM-o script against the same `test_score.tsv` prompts. The 4 kHz rate was not run.

| natural references at | LibriTTS | VCTK |
|---|---|---|
| 24000 (untouched) | 0.740 | 0.726 |
| 22050 | 0.740 | 0.726 |
| 16000 | 0.738 | 0.725 |
| 8000 | 0.654 | 0.651 |

The untouched row reproduces the natural-speech table above, so the pipeline matches. Cutting
genuine speech to 16 kHz costs nothing measurable. Cutting it to 8 kHz costs 0.086 on LibriTTS
and 0.075 on VCTK, with the same speaker, the same words and nothing synthetic involved.

**Registered rule and verdict.** The rule was fixed before the numbers were seen: the share of
stock's 8 kHz drop that bandwidth alone explains, (0.740 − natural at 8 kHz) ÷ (0.740 − 0.610)
on LibriTTS and (0.726 − natural at 8 kHz) ÷ (0.726 − 0.514) on VCTK. Two-thirds or more
meant go, one-third to two-thirds meant a weaker decomposition result, under one-third meant
the claim fails.

| | LibriTTS | VCTK |
|---|---|---|
| share explained by bandwidth | 0.665 | 0.356 |
| paired bootstrap 95% CI | [0.55, 0.82] | not available |

On the registered rule the verdict is **a weaker decomposition result on both corpora**.
LibriTTS sits on the two-thirds line (bootstrap probability of being at or above it 0.48).
VCTK has no interval because the stock VCTK wavs and per-utterance SIM-o scores were not
kept; only the mean 0.5144 survives.

**Post-hoc correction, made after seeing the numbers.** The VCTK denominator is not stock's
8 kHz drop. Stock scores only 0.621 on VCTK even at a 24 kHz prompt, so 0.105 of the rule's
0.212 is a shortfall that exists at full band and has nothing to do with bandwidth. Measured
from stock's own 24 kHz score, the drop is 0.621 − 0.514 = 0.107, and bandwidth explains
0.075 ÷ 0.107 = 0.71 of it. On LibriTTS, stock at 24 kHz (0.739) equals natural, so the
correction changes nothing there. This reading is reported as a correction, not as the
registered verdict.

**The same thing without a ratio.** Comparing each system with natural speech at the same
bandwidth avoids the denominator question altogether:

| SIM-o difference | LibriTTS | VCTK |
|---|---|---|
| bandwidth effect on genuine recordings (natural 24 kHz − natural 8 kHz) | 0.086 | 0.075 |
| stock's shortfall against natural, full band | 0.001 | 0.105 |
| stock's shortfall against natural, both at 8 kHz | 0.044 | 0.137 |
| identity lost to the degraded prompt (difference of the two rows above) | 0.043 | 0.032 |

On both corpora about 0.08 of stock's 8 kHz drop is the verifier reacting to bandwidth and
0.03 to 0.04 is identity the model actually loses, roughly two-thirds against one-third, and
the corpora agree. Stated as a level: at an 8 kHz prompt, a cloner that reproduces its
prompt's recording channel perfectly would score about 0.65, not 0.74.

### Paired rerun and a second verifier

The tables above were built from stock means computed when the stock wavs were not kept.
To pair everything per utterance, stock ZipVoice was resynthesised at 8 kHz and 24 kHz
prompts on both corpora (CPU, `--seed 666`, 16 steps; output in
`C:/zipvoice_data/exp/results_rs/`). The resynthesis reproduces the documented SIM-o means
to within 0.004 (LibriTTS 0.613 and 0.737, VCTK 0.518 and 0.622). Every row below is a
difference of means over the same 76 utterances, with a 95% paired bootstrap interval (10,000
resamples of utterances, one draw applied to all four systems).

The second verifier is CAM++ (3D-Speaker, trained on VoxCeleb,
`iic/speech_campplus_sv_en_voxceleb_16k`), with Kaldi fbank features at 16 kHz and the same
pairs and cosine scoring as SIM-o.

| | LibriTTS SIM-o | LibriTTS CAM++ | VCTK SIM-o | VCTK CAM++ |
|---|---|---|---|---|
| natural 24 kHz / 8 kHz | 0.740 / 0.654 | 0.750 / 0.606 | 0.726 / 0.651 | 0.748 / 0.583 |
| stock 24 kHz / 8 kHz prompt | 0.737 / 0.613 | 0.795 / 0.628 | 0.622 / 0.518 | 0.702 / 0.553 |
| bandwidth effect on genuine recordings | 0.086 [0.076, 0.097] | 0.144 [0.130, 0.158] | 0.075 [0.068, 0.083] | 0.165 [0.152, 0.179] |
| stock's shortfall, full band | 0.003 [−0.018, 0.023] | −0.045 [−0.068, −0.024] | 0.104 [0.084, 0.124] | 0.046 [0.030, 0.063] |
| stock's shortfall, both at 8 kHz | 0.041 [0.015, 0.067] | −0.022 [−0.045, −0.001] | 0.132 [0.111, 0.154] | 0.030 [0.012, 0.047] |
| **identity lost to the degraded prompt** | **0.038 [0.023, 0.052]** | **0.023 [0.009, 0.037]** | **0.029 [0.013, 0.045]** | **−0.017 [−0.035, 0.001]** |
| stock's own 8 kHz drop | 0.124 [0.109, 0.139] | 0.167 [0.151, 0.184] | 0.104 [0.087, 0.121] | 0.148 [0.132, 0.164] |
| registered share | 0.678 [0.55, 0.86] | 1.18 [1.01, 1.46] | 0.363 [0.32, 0.41] | 0.848 [0.77, 0.93] |
| corrected share (post hoc) | 0.694 [0.61, 0.80] | 0.861 [0.79, 0.94] | 0.725 [0.62, 0.86] | 1.11 [0.99, 1.25] |

**The registered verdict does not change.** With the pre-registered verifier (SIM-o), the
share is 0.678 on LibriTTS, which is on the two-thirds line (bootstrap probability of being at
or above it 0.55), and 0.363 on VCTK, with an interval entirely inside the one-third to
two-thirds band. That is a weaker decomposition result on both corpora. The post-hoc correction
is 0.69 and 0.73, and its intervals are now available.

**What the paper reports.** The registered SIM-o share stays the pre-registered outcome and is
reported as such. The paper's main quantity, for both verifiers, is the identity-lost row with
its interval, not a share.

- Shares are not reported for CAM++. They exceed 1 in two cells (1.18 registered on LibriTTS,
  1.11 corrected on VCTK), which no reading of "the verifier's share of the drop" allows.
- They exceed 1 because CAM++'s channel-copy bonus (below) sits inside the denominator: stock's
  own 8 kHz drop starts from a full-band score inflated above natural speech.
- The identity-lost row is a difference of differences on the same items, so the bonus cancels
  out of it. It is also the quantity the listening test measures directly (see "Listening
  test: design").

**What the second verifier changes.**

- **The bandwidth effect is not a quirk of one model.** Under CAM++ it is roughly twice as
  large: 0.144 and 0.165, against 0.086 and 0.075 under SIM-o. Under both verifiers, cutting
  genuine speech to 8 kHz costs more than stock's full-band shortfall on LibriTTS.
- **The size of the identity loss depends on the verifier.**
  - SIM-o finds a small, clearly non-zero loss on both corpora: 0.038 and 0.029, with
    intervals excluding zero.
  - CAM++ finds 0.023 on LibriTTS and nothing measurable on VCTK: −0.017, interval
    [−0.035, 0.001].
  - So the metric part is the majority of stock's 8 kHz drop under both verifiers, while the
    part that is real identity loss is between 0 and 0.04 depending on the verifier.
- **CAM++ rewards channel copying.** On LibriTTS, stock at a 24 kHz prompt scores above
  natural speech under CAM++ (0.795 against 0.750). The generated utterance shares the
  prompt's recording session and level, and a different natural utterance does not. CAM++
  gives this more weight than SIM-o does. It is also why the CAM++ shortfall rows go
  negative. It is a reminder that "natural" is not an upper bound for either verifier.

**The additivity premise holds approximately.** Stock's output at an 8 kHz prompt is
band-limited the way the prompt is. The median energy above 4.2 kHz, relative to total
energy, is:

- LibriTTS: −39.6 dB for stock against −40.9 dB for the natural 8 kHz references. At full
  band the figures are −17.5 dB for stock and −18.0 dB for natural.
- VCTK: −38.5 dB against −40.6 dB. At full band, −17.0 dB and −19.0 dB.

Stock adds about 1–2 dB of energy above the cutoff, which is noise-level, not a restored
band. Comparing its output with natural speech cut to the same band is therefore a like-for-like
comparison.

Limits that remain:

- Two verifiers, both scoring at 16 kHz. Neither can see above 8 kHz, so nothing here says
  whether a restored high band would be credited.
- Resynthesis ran on CPU. The noise draw therefore differs from the original GPU runs, but
  the means agree.
- The listening test that would say which verifier's identity-loss figure is closer to human
  judgement is not yet run.

### A second TTS model: stock F5-TTS

The same split was run for stock F5-TTS (v1 Base, `model_1250000.safetensors`, 336M
parameters, trained on Emilia). Its settings are upstream's defaults: 32 steps, sway sampling
−1, CFG 2.0, Vocos vocoder, `--seed 666`. It runs with the released vocabulary and pinyin
conversion. Upstream's demo chunking, silence trimming and cross-fading are not used, which
is the same rule as `f5_adapter/infer.py`.

The run did not go through `f5_adapter/infer.py`, because that script no longer imports on
the `dual-diffusion` branch (`BandAdapter` is gone from `zipvoice.models.modules.band_adapter`).
A standalone script built the model with upstream's `load_model`, so no adapter code was on
the path. It ran on the GPU. Output is in `C:/zipvoice_data/exp/results_rs/{libritts,vctk}/f5_sr{24000,8000}`.

The natural references and the bootstrap are the ones above: the same 76 items, 10,000
resamples, seed 0, and one draw shared by all systems. B is therefore identical, and
ZipVoice's rows reproduce exactly.

| | LibriTTS SIM-o | LibriTTS CAM++ | VCTK SIM-o | VCTK CAM++ |
|---|---|---|---|---|
| F5 24 kHz / 8 kHz prompt | 0.724 / 0.584 | 0.792 / 0.609 | 0.600 / 0.483 | 0.715 / 0.537 |
| F5's own 8 kHz drop D | 0.140 [0.123, 0.157] | 0.183 [0.167, 0.199] | 0.117 [0.098, 0.136] | 0.178 [0.158, 0.198] |
| B / D | 0.62 [0.55, 0.70] | 0.79 [0.72, 0.86] | 0.64 [0.56, 0.75] | 0.93 [0.82, 1.05] |
| **identity lost I** | **0.054 [0.039, 0.068]** | **0.039 [0.024, 0.054]** | **0.042 [0.025, 0.059]** | **0.013 [−0.008, 0.034]** |
| I(F5) − I(ZipVoice), paired | 0.016 [−0.000, 0.031] | 0.016 [0.003, 0.029] | 0.013 [−0.009, 0.034] | 0.030 [0.010, 0.050] |

- **The picture is the same.** B is the majority of F5's drop in every cell, and CAM++
  again finds no measurable loss on VCTK.
- **F5 loses a little more identity than ZipVoice.** Its I is 0.013–0.030 higher. The paired
  difference excludes zero with CAM++ only; with SIM-o the LibriTTS interval just touches zero.
- **The range across models and verifiers widens.** It is now 0 to 0.05, up from 0 to 0.04.
- **The additivity premise holds for F5 as well.** At an 8 kHz prompt, its median energy
  above 4.2 kHz is −38.8 dB on LibriTTS and −39.0 dB on VCTK, against −40.9 dB and −40.6 dB
  for the natural 8 kHz references. At full band it is −16.4 dB and −16.5 dB.
- **CAM++'s channel-copy bonus appears for F5 too.** On LibriTTS, F5 at a full-band prompt
  scores 0.792 against natural's 0.750.

### Speaker-level intervals

**Why.** LibriTTS has 38 speakers with exactly two items each. The two items of a speaker are
not independent, so resampling items treats 76 correlated values as 76 independent ones and
gives intervals that are too narrow. Resampling speakers, with a speaker's two items drawn
together, is the correct unit. VCTK has one item per speaker (76 speakers), so there the two
bootstraps are the same draw and no VCTK number changes.

**Method.** `speaker_boot.py` (session scratchpad 1be2914b) recomputes every LibriTTS interval
quoted in the paper from the same score files. 10,000 draws, seed 0, and the same draws for
every statistic on the same items. The item-level column reproduces the published intervals to
within 0.001, which checks the inputs. For the session subsets, speakers are drawn from those
present in the subset (23 same-session, 25 different-session). The session gap draws speakers
once from all 38 and takes both subset means from that draw.

| LibriTTS | verifier | estimate | item CI (published) | speaker CI |
|---|---|---|---|---|
| B = n24 − n8 | SIM-o | 0.086 | [0.076, 0.097] | [0.075, 0.100] |
| | CAM++ | 0.144 | [0.130, 0.158] | [0.127, 0.162] |
| I, ZipVoice | SIM-o | 0.038 | [0.023, 0.052] | [0.023, 0.053] |
| | CAM++ | 0.023 | [0.009, 0.037] | [0.007, 0.039] |
| I, F5-TTS | SIM-o | 0.054 | [0.039, 0.068] | [0.037, 0.071] |
| | CAM++ | 0.039 | [0.024, 0.054] | [0.023, 0.055] |
| I(F5) − I(ZipVoice) | SIM-o | 0.016 | [−0.000, 0.031] | [−0.002, 0.032] |
| | CAM++ | 0.016 | [0.003, 0.029] | [0.000, 0.031] |
| I, ZipVoice, −23 LUFS | CAM++ | 0.017 | [0.002, 0.031] | [−0.000, 0.033] |
| registered share | SIM-o | 0.68 | [0.55, 0.86] | [0.54, 0.89] |
| B / D, F5-TTS | SIM-o | 0.62 | [0.55, 0.70] | [0.54, 0.71] |
| | CAM++ | 0.79 | [0.72, 0.86] | [0.71, 0.87] |
| AudioSR − n8 | SIM-o | −0.041 | [−0.053, −0.030] | [−0.056, −0.028] |
| | CAM++ | −0.018 | [−0.030, −0.005] | [−0.033, −0.002] |
| AudioSR 16-band − n8 | SIM-o | −0.036 | [−0.048, −0.025] | [−0.050, −0.023] |
| | CAM++ | −0.016 | [−0.029, −0.003] | [−0.032, +0.000] |
| NVSR − n8 | SIM-o | −0.067 | [−0.082, −0.053] | [−0.085, −0.049] |
| | CAM++ | −0.041 | [−0.058, −0.024] | [−0.061, −0.021] |
| NVSR 16-band − n8 | SIM-o | −0.066 | [−0.081, −0.052] | [−0.085, −0.048] |
| | CAM++ | −0.039 | [−0.057, −0.022] | [−0.059, −0.020] |
| cascade − ZipVoice at 8 kHz | SIM-o | −0.055 | [−0.075, −0.035] | [−0.078, −0.032] |
| | CAM++ | −0.034 | [−0.052, −0.016] | [−0.056, −0.012] |
| different-session: ZipVoice 24 − n24 | SIM-o | +0.031 | [+0.001, +0.062] | [−0.004, +0.067] |
| | CAM++ | +0.082 | [+0.049, +0.118] | [+0.041, +0.129] |
| session gap (different − same) | SIM-o | +0.072 | [+0.034, +0.109] | [+0.032, +0.111] |
| | CAM++ | +0.079 | [+0.040, +0.119] | [+0.033, +0.129] |
| CAM++ − SIM-o offset of ZipVoice 24 − n24 | both | +0.048 | [+0.035, +0.062] | [+0.034, +0.063] |

Most intervals widen, by up to 0.011 at either end. Three conclusions change, all at the
margin:

- **Different-session lead under SIM-o.** ZipVoice at full band beating natural speech against
  different-session references (+0.031) is no longer significant under SIM-o. Under CAM++
  it still is. The session effect itself is unchanged.
- **Loudness-normalised CAM++ I on LibriTTS** (0.017) now has an interval that reaches zero
  (lower end −0.0005).
- **AudioSR with 16-band matching under CAM++ on LibriTTS** (−0.016) no longer excludes zero.
  The unmatched AudioSR penalty (−0.018) still does, so matching still changes little.

Unchanged: every B, every ZipVoice and F5 I, the cascade and NVSR penalties, and the
registered share's verdict. F5's I minus ZipVoice's still excludes zero with CAM++ on LibriTTS,
but only just (lower end +0.0004).

The paper now uses the speaker-level intervals for LibriTTS, including Figure 1, and says so
in the statistics paragraph. The listening test's analysis now resamples speakers too.

### Band-matched scoring does not replace the genuine-speech control

**Question.** The obvious alternative to Eq. 1 is to low-pass the reference to the prompt's
band as well, so that output and reference are compared in the same band. Does that remove
the bandwidth effect, and what happens to the residual and to the Table 3 contrasts?

**Method.** Session scratchpad 1be2914b, `bandmatch_prep.py`, `bandmatch_score.sh`,
`bandmatch_stats.py`. LP(x) is the 24 → 8 → 24 kHz torchaudio round trip that made the 8 kHz
prompts and the control-1 references. LP of the natural references reproduces the control-1
8 kHz files bit for bit. The band-matched score is SIM_m(x) = cos(E(LP x), E(LP prompt)), applied
to every scored signal, including ones that are already band-limited. The proposal under test
scores the 8 kHz condition band-matched and the 24 kHz condition as published:
B_m = n24 − n8m, D_m = s24 − s8m, I_m = D_m − B_m. "Low band only" scores both conditions
band-matched (s24m − s8m). Both verifiers, both corpora, existing audio only. Speaker-level
bootstrap, 10,000 draws, seed 0. The published B, D, I and cascade contrast are reproduced
exactly.

| | verifier | LibriTTS | VCTK |
|---|---|---|---|
| B (published) | SIM-o | 0.086 [0.075, 0.100] | 0.075 [0.067, 0.083] |
| | CAM++ | 0.144 [0.127, 0.162] | 0.165 [0.152, 0.179] |
| B_m (band-matched) | SIM-o | 0.020 [0.011, 0.030] | 0.017 [0.008, 0.026] |
| | CAM++ | 0.017 [0.004, 0.031] | 0.030 [0.019, 0.040] |
| I, ZipVoice | SIM-o | 0.038 [0.023, 0.053] | 0.029 [0.013, 0.045] |
| | CAM++ | 0.023 [0.007, 0.039] | −0.017 [−0.035, 0.001] |
| I_m, ZipVoice | SIM-o | 0.013 [−0.001, 0.029] | 0.015 [−0.001, 0.031] |
| | CAM++ | −0.008 [−0.028, 0.012] | −0.036 [−0.053, −0.021] |
| I, F5-TTS | SIM-o | 0.054 [0.037, 0.071] | 0.042 [0.025, 0.059] |
| | CAM++ | 0.039 [0.023, 0.055] | 0.013 [−0.008, 0.034] |
| I_m, F5-TTS | SIM-o | 0.032 [0.014, 0.050] | 0.035 [0.018, 0.052] |
| | CAM++ | −0.007 [−0.026, 0.012] | −0.017 [−0.036, 0.000] |
| low band only, ZipVoice | SIM-o | 0.024 [0.012, 0.037] | 0.015 [0.001, 0.031] |
| | CAM++ | 0.007 [−0.004, 0.019] | −0.017 [−0.029, −0.005] |
| low band only, F5-TTS | SIM-o | 0.040 [0.025, 0.056] | 0.035 [0.019, 0.052] |
| | CAM++ | 0.011 [0.001, 0.022] | −0.004 [−0.017, 0.009] |
| cascade − ZipVoice at 8 kHz, published | SIM-o | −0.055 [−0.078, −0.032] | +0.050 [+0.031, +0.069] |
| | CAM++ | −0.034 [−0.056, −0.012] | +0.065 [+0.045, +0.086] |
| cascade − ZipVoice at 8 kHz, band-matched | SIM-o | −0.039 [−0.058, −0.021] | +0.007 [−0.009, +0.024] |
| | CAM++ | −0.037 [−0.052, −0.023] | −0.020 [−0.034, −0.007] |
| AudioSR − n8, published | SIM-o | −0.041 [−0.056, −0.028] | −0.026 [−0.038, −0.014] |
| | CAM++ | −0.018 [−0.033, −0.002] | −0.004 [−0.022, +0.014] |
| AudioSR − n8, both low-passed | SIM-o | −0.010 [−0.016, −0.005] | −0.011 [−0.015, −0.008] |
| | CAM++ | −0.010 [−0.016, −0.004] | −0.017 [−0.021, −0.012] |
| NVSR − n8, published | SIM-o | −0.067 [−0.085, −0.049] | −0.006 [−0.017, +0.004] |
| | CAM++ | −0.041 [−0.061, −0.021] | +0.042 [+0.023, +0.063] |
| NVSR − n8, both low-passed | SIM-o | −0.034 [−0.043, −0.026] | −0.019 [−0.024, −0.013] |
| | CAM++ | −0.025 [−0.031, −0.018] | −0.016 [−0.022, −0.011] |

**Findings.**

- **Band matching removes most of B but not all of it.** It removes 77–88% of B in every cell.
  The rest, 0.017–0.030, excludes zero in all four cells. Two same-speaker signals that are both
  cut to 4 kHz still score lower than two full-band ones, so a band-matched score at 8 kHz is
  still not comparable with a full-band score at 24 kHz. The genuine-speech control of Eq. 1 is
  still needed; band matching cannot replace it.
- **The residual shrinks.** I_m is below I in all eight cells, by 0.007–0.046, and the gap
  excludes zero in seven of them. ZipVoice's SIM-o residual no longer excludes zero on either
  corpus. The cause is not energy the systems put above the prompt band. Their 8 kHz-prompt
  outputs have a median 4.2–8 kHz energy 39–40 dB below the 0–4 kHz band, against 41 dB for
  natural speech cut the same way. So band limiting costs the system's output more SIM than it
  costs natural speech: the bandwidth effect is not fully additive, and part of the published I
  is bandwidth. This fits the paper's reading of I as a residual that is not an identity
  measure on its own.
- **Low band only is the cleanest identity estimate, within its limits.** The genuine-speech
  counterpart is 0.000–0.003, so this contrast has no bandwidth term by construction. ZipVoice
  loses 0.015–0.024 under SIM-o and between −0.017 and +0.007 under CAM++. F5-TTS loses
  0.035–0.040 under SIM-o and between −0.004 and +0.011 under CAM++. It sees nothing above 4 kHz.
- **Cascade on LibriTTS: no change.** The cascade is worse than stock ZipVoice under both
  scorings.
- **Cascade on VCTK: the published advantage disappears.** It goes from +0.050 (SIM-o) and
  +0.065 (CAM++) to +0.007 (not significant) and −0.020. NVSR's CAM++ credit on VCTK goes from
  +0.042 to −0.016. The VCTK advantage therefore lives entirely in the added band. This fits the
  reading that it comes from NVSR having trained on the VCTK evaluation speakers.
- **Most of AudioSR's penalty is its added band.** On LibriTTS SIM-o the penalty falls from
  −0.041 to −0.010. For NVSR it falls only by about half (−0.067 to −0.034): NVSR also damages
  the band it was given.

**What band matching gives up.** It is blind to anything above the prompt band. A BWE stage or
cascade gets neither credit nor penalty for the band it adds, so a correct speaker-specific high
band and a wrong one score the same. It also needs the prompt's band to be known. It works as a
second reading next to Eq. 1, not as a replacement.

Not yet in the paper.

### Where the channel-copy bonus comes from: the recording session

**Question.** On LibriTTS under CAM++, stock at a 24 kHz prompt scores above natural speech
(0.795 against 0.750). The explanation offered above is that the output copies the prompt's
recording channel, which a different natural utterance may not share. This was tested directly.

**Method.**

- For each of the 76 LibriTTS items, the prompt's source utterance was recovered in dev-clean
  by matching speaker and normalised transcript. Every prompt matched exactly one utterance.
- Items were split by whether the prompt and the natural reference come from the same chapter.
  In LibriVox audiobooks a chapter is one recording session, so the same chapter means the same
  microphone, room and level.
- The groups are 36 same-chapter items and 40 different-chapter items.
- For each group: the bonus is stock 24 kHz minus natural 24 kHz, and identity lost is as in the
  paired table. Differences between groups use a 10,000-draw bootstrap.
- Scripts: `chapter_map.py` and `chapter_test.py` (session scratchpad). VCTK has one session
  per speaker, so this split does not exist there.

| | SIM-o same chapter | SIM-o different | CAM++ same chapter | CAM++ different |
|---|---|---|---|---|
| natural 24 kHz | 0.785 | 0.699 | 0.794 | 0.711 |
| stock 24 kHz prompt | 0.745 | 0.730 | 0.797 | 0.794 |
| bonus (stock − natural) | −0.041 [−0.061, −0.020] | +0.031 [+0.001, +0.062] | +0.004 [−0.013, +0.021] | +0.082 [+0.048, +0.119] |
| identity lost at 8 kHz | 0.057 [0.036, 0.078] | 0.021 [0.002, 0.040] | 0.029 [0.008, 0.050] | 0.018 [−0.000, 0.037] |

Difference in bonus, different chapter minus same chapter: SIM-o +0.072 [+0.034, +0.109];
CAM++ +0.079 [+0.040, +0.119].

**Result.** The channel-copy explanation holds.

- Stock's own score barely depends on the reference's session: 0.745 against 0.730 on SIM-o,
  0.797 against 0.794 on CAM++. It always carries the prompt's channel.
- Same-chapter pairs score higher. A natural reference from the prompt's own session scores
  0.08–0.09 above one from another session: 0.785 against 0.699 on SIM-o, and 0.794 against
  0.711 on CAM++.
- So stock beats natural speech only when the natural reference comes from a different
  session. Against a same-session reference, stock falls short on SIM-o and ties on CAM++.
- Under CAM++, the whole bonus of 0.045 comes from the different-chapter items.

**Why only CAM++ rates stock above natural overall.** Both verifiers show the same session
effect (+0.072 and +0.079), yet the pooled bonus is +0.045 on CAM++ and −0.003 on SIM-o. Two
further checks separate this (`chapter_offset.py`):

- **Mix.** 36 of the 76 references (0.47) share the prompt's chapter.
- **Stock minus natural, per subset, and the verifier offset** (CAM++ bonus − SIM-o bonus,
  paired within items):

| subset | SIM-o | CAM++ | CAM++ − SIM-o |
|---|---|---|---|
| same chapter (n=36) | −0.041 | +0.004 | +0.044 [+0.029, +0.061] |
| different chapter (n=40) | +0.031 | +0.082 | +0.051 [+0.030, +0.073] |

- The offset does not depend on session: different minus same is +0.007 [−0.020, +0.034].
- The two effects add. Both verifiers credit a shared session by the same amount. On top of
  that, CAM++ favours stock's output over natural speech by about 0.045–0.05 in either subset,
  and SIM-o does not.
- **Consequence for the pooled sign.** Under CAM++, stock is at or above natural in both
  subsets, so it wins at any mix. Under SIM-o, the two subsets have opposite signs. The pooled
  bonus crosses zero at a same-chapter share of 0.43, and the observed 0.47 sits just past it.
  Whether SIM-o rates stock above natural is therefore an accident of the mix; under CAM++ it
  is not.
- What the CAM++ offset responds to in stock's output is not identified here. Level is ruled
  out: rescored on copies of every clip normalised to −23 LUFS (the listening-test stimuli),
  the offset is +0.048 [+0.034, +0.062], the same as before, and +0.044 and +0.052 per subset
  (see "Rescored on the listeners' audio"). The noise floor remains a candidate, since CAM++ is
  also the verifier that reacts most to bandwidth.

**What this means for the decomposition.**

- Both verifiers credit shared recording conditions by about as much as the whole bandwidth
  effect on SIM-o. "Natural speech" is therefore not a single ceiling. It depends on whether the
  reference shares the prompt's session.
- Identity lost is larger against same-session references: 0.057 against 0.021 on SIM-o. A
  same-session reference is the stricter test, because the channel is held constant and only
  the speaker remains to differ.
- The 0.65 ceiling and the metric/identity split above average over a 36/40 mix of sessions.
  They would move with a different mix, and a paper should report them per group or state the
  mix.

### Control 2: an off-the-shelf extender does not win back the metric's part

**Question.** If an extender restored the missing band convincingly, band-limited speech
should score back toward the full-band level. Control 2 tests this with genuine recordings,
so the TTS model plays no part.

**Method.**

- The same 76 natural references per corpus were cut to 8 kHz exactly as in control 1 and
  saved at 8 kHz.
- They were restored with released AudioSR (`basic`, 50 DDIM steps, guidance 3.5, seed 42),
  using `audiosr_restore.py`. AudioSR's own post-processing splices the observed low band
  back, so only the band above 4 kHz is new.
- Each output was resampled to 24 kHz. AudioSR pads its output past the input by up to 5.1 s
  (median 2.4 s on LibriTTS, 1.3 s on VCTK). The pad is near-silent, about −67 dB relative
  to the speech, but duration is a known confound for these embeddings, so every file was cut
  back to its reference's length before scoring.
- Restoration time averaged 9.0 s per file on LibriTTS and 7.0 s on VCTK, on the 16 GB GPU.
- Scoring used the same pairs and the same two verifiers as control 1.

| | LibriTTS SIM-o | LibriTTS CAM++ | VCTK SIM-o | VCTK CAM++ |
|---|---|---|---|---|
| natural 24 kHz | 0.740 | 0.750 | 0.726 | 0.748 |
| natural 8 kHz | 0.654 | 0.606 | 0.651 | 0.583 |
| natural 8 kHz, AudioSR-restored | 0.612 | 0.588 | 0.625 | 0.578 |
| restored − 8 kHz, paired 95% CI | −0.041 [−0.053, −0.030] | −0.018 [−0.030, −0.005] | −0.026 [−0.038, −0.014] | −0.004 [−0.022, +0.013] |
| items where restoring lowers the score | 78% | 58% | 70% | 57% |

Before trimming, the restored scores were a further 0.01 lower: 0.602, 0.578, 0.615 and
0.563. The trailing silence alone cost measurable similarity, as Carbonneau et al.'s duration
finding predicts.

**Result.** AudioSR recovers none of the bandwidth effect. It leaves CAM++ where the
band-limited audio was, and it makes SIM-o worse on both corpora, with intervals that exclude
zero. The restored band is also too loud. Its median energy relative to the total, set against
natural full-band speech, is:

| band | LibriTTS restored | LibriTTS natural | VCTK restored | VCTK natural |
|---|---|---|---|---|
| 4.2–8 kHz | −17.3 dB | −18.5 dB | −14.7 dB | −20.4 dB |
| 8–12 kHz | −19.6 dB | −26.1 dB | −15.3 dB | −27.7 dB |

**What control 2 shows.**

- A generic high band added to genuine speech costs SIM-o. The low band is the real speaker
  (AudioSR splices it back unchanged), so the penalty comes entirely from the added band.
- The metric's part of the 8 kHz drop, 0.086 on SIM-o, is not something any plausible high
  band buys back. This extender scores below the band-limited input it started from.
- The two verifiers react in opposite ways. CAM++ reacts about twice as strongly as SIM-o to a
  missing band, but barely to a generic added one. SIM-o penalises both. Which system
  "clones better" at a degraded prompt can therefore depend on the choice of verifier.
- For stock ZipVoice, the ceiling reading stands: at an 8 kHz prompt, reproducing the
  prompt's channel scores about 0.65 on SIM-o.

**What control 2 does not show.** It says nothing about what a speaker-specific high band
would score, because none was tested. A plausible conjecture is that only such a band could
recover the metric's part. This belongs in the discussion, labelled as a conjecture, and not
as a result. Control 2b adds uncontrolled evidence that fits it but does not test it.

**VCTK caveat (found later).** All 76 VCTK evaluation speakers are in VoiceFixer's released
speech archive (Zenodo 5546723, all 108 VCTK 0.92 speakers), which is the source the AudioSR
paper gives for its speech. AudioSR has therefore probably seen every VCTK speaker here. The
VCTK column is contaminated in the same direction as NVSR's in control 2b. It is also the
column with the smaller penalty (−0.026 against −0.041 on SIM-o, and none on CAM++). LibriTTS
is the clean comparison. See the EER companion's training-data caveat.

**Caveat.** This is one extender at default settings, and its band is 1–6 dB too loud in the
range SIM-o sees. If matching the band's level removes the penalty, it is a spectral-balance
effect, which is Carbonneau et al.'s equalisation finding. If the penalty survives, the
verifier is reacting to what the band contains, which re-equalisation cannot fix. The
level-matched run below separates the two.

### Control 2, level-matched: the penalty is the band's content, not its level

**Method.** Each trimmed AudioSR output was rescaled in the STFT domain (2048-point Hann,
hop 512) so that its 4.2–12 kHz band carries the same energy as the natural 24 kHz recording
of the same utterance. Using the same utterance makes this an oracle level, stronger than any
re-equalisation available in practice. Two variants:

- **gain**: one gain over the whole 4.2–12 kHz range.
- **eq16**: 16 equal-width bands over 4.2–12 kHz, each matched separately. This is the
  bandwidth analogue of Carbonneau et al.'s 16-band re-equalisation, applied only to the added
  band so the genuine low band is untouched.

Script: `control2_levelmatch.py` (session scratchpad). Durations match the references exactly.

| restored − natural 8 kHz, paired 95% CI | LibriTTS SIM-o | LibriTTS CAM++ | VCTK SIM-o | VCTK CAM++ |
|---|---|---|---|---|
| AudioSR, trimmed | −0.041 [−0.053, −0.030] | −0.018 [−0.030, −0.005] | −0.026 [−0.038, −0.014] | −0.004 [−0.022, +0.013] |
| level-matched, one gain | −0.035 [−0.047, −0.024] | −0.017 [−0.030, −0.005] | −0.017 [−0.028, −0.005] | −0.012 [−0.030, +0.007] |
| level-matched, 16 bands | −0.036 [−0.048, −0.025] | −0.016 [−0.030, −0.003] | −0.017 [−0.029, −0.006] | −0.010 [−0.028, +0.008] |

The matching did what it was meant to. Median band energy relative to the total, in dB:

| band | variant | LibriTTS | VCTK |
|---|---|---|---|
| 4.2–8 kHz | natural 24 kHz | −18.5 | −20.4 |
| | AudioSR / one gain / 16 bands | −17.3 / −19.4 / −18.3 | −14.7 / −20.9 / −19.8 |
| 8–12 kHz | natural 24 kHz | −26.1 | −27.7 |
| | AudioSR / one gain / 16 bands | −19.6 / −22.9 / −26.0 | −15.3 / −20.8 / −25.4 |

The 16-band variant matches natural speech to within 0.3–2.3 dB in both ranges. The single gain
matches the total but leaves the 8–12 kHz range 3–7 dB hot, because AudioSR's band is flatter
than speech.

**Result.** The SIM-o penalty survives oracle level and 16-band spectral matching on both
corpora, with every interval excluding zero:

- Level accounts for about 12% of the SIM-o penalty on LibriTTS (0.041 to 0.036) and about 35%
  on VCTK (0.026 to 0.017), where AudioSR's band was loudest.
- The remainder, 0.036 and 0.017, is a reaction to what the band contains: its time–frequency
  structure, not its long-term level or spectral balance.
- CAM++ is essentially unchanged by matching: still a small penalty on LibriTTS, and still
  nothing measurable on VCTK.
- Matching the single gain and matching 16 bands give the same scores, so the residual 8–12 kHz
  excess under the single gain is not what SIM-o reacts to.

**Why this matters against Carbonneau et al.** Their re-equalisation removed about half of the
spectral-tilt effect. Here the bandwidth analogue of the same remedy, with an oracle target,
removes 12–35% of the penalty. What remains cannot be fixed by equalisation, because it is not
a balance problem. A verifier that sees only up to 8 kHz is penalising a generic 4.2–8 kHz band
for its content. That is a different confound from the ones they list.

**Limits.**

- One extender here. Control 2b below repeats this with NVSR, the extender behind this
  repository's cascade baseline.
- The level target is oracle. A practical system could only match to a speaker-average or
  corpus-average spectrum, which would match worse, so the survival result is conservative.
- Energy was matched over the whole utterance. Frame-level envelope differences (for example,
  high-band energy during silences or voiced frames) were not matched, and they are one
  candidate for the "content" the verifier reacts to.

### Control 2b: a second extender, and the cascade on a real system

**Why NVSR.** This repository's cascade baseline is NVSR prompt restoration followed by stock
ZipVoice. An earlier reading compared the cascade's shortfall with AudioSR's control-2 penalty,
but AudioSR is not the extender the cascade uses. The relevant test is NVSR on the same genuine
references.

**Method.**

- **Genuine references.** The same 8 kHz natural references as control 2 were restored with the
  released NVSR checkpoint (`nvsr_restore.py`, its own low-band splice kept). This took 12 s on
  LibriTTS and 7 s on VCTK on the shared GPU, a real-time factor of 0.02.
- Outputs are within 3 samples of the reference length and were fitted to it exactly. They were
  then level-matched in the same two ways as control 2 and scored with both verifiers.
- **Cascade.** The cascade wavs from the original runs were not kept. Stock ZipVoice was
  therefore rerun on the GPU (`Device: cuda:0` in its logs) on the existing NVSR-restored prompts (`test_nvsr8000.tsv`, `--seed
  666`, 16 steps; output in `C:/zipvoice_data/exp/results_rs/{corpus}/cascade_nvsr_sr8000`).
  LibriTTS SIM-o is 0.557, against the documented 0.563.
- **Scripts** (session scratchpad): `control2b_run.sh`, `control2_fit.py`, `cascade_rs.sh` and
  `cascade_compare.py`. All intervals are 95% paired bootstrap over the 76 items.

**NVSR on genuine speech** (restored − natural 8 kHz):

| | LibriTTS SIM-o | LibriTTS CAM++ | VCTK SIM-o | VCTK CAM++ |
|---|---|---|---|---|
| AudioSR, trimmed (control 2) | −0.041 [−0.053, −0.030] | −0.018 [−0.030, −0.005] | −0.026 [−0.038, −0.014] | −0.004 [−0.022, +0.013] |
| NVSR | −0.067 [−0.082, −0.053] | −0.041 [−0.059, −0.024] | −0.006 [−0.017, +0.004] | **+0.042 [+0.022, +0.063]** |
| NVSR, level-matched, 16 bands | −0.066 [−0.081, −0.052] | −0.039 [−0.057, −0.022] | −0.004 [−0.014, +0.007] | +0.045 [+0.025, +0.064] |

NVSR's band is 2–5 dB quieter than natural speech, where AudioSR's was louder. Level-matching
changes nothing, so control 2's conclusion extends to a second extender: whatever the verifiers
react to is not the band's level.

**The cascade on a real system** (cascade − stock at the raw 8 kHz prompt):

| | LibriTTS SIM-o | LibriTTS CAM++ | VCTK SIM-o | VCTK CAM++ |
|---|---|---|---|---|
| stock, 8 kHz prompt | 0.613 | 0.628 | 0.518 | 0.553 |
| cascade (NVSR prompt) | 0.557 | 0.594 | 0.569 | 0.619 |
| cascade − stock, 8 kHz | −0.055 [−0.075, −0.035] | −0.034 [−0.052, −0.016] | +0.050 [+0.031, +0.069] | +0.065 [+0.045, +0.086] |
| stock 24 kHz − cascade | +0.179 | +0.202 | +0.054 | +0.083 |

The cascade's output carries a full-level high band: 4.2–8 kHz at −19.4 dB on LibriTTS and
−16.9 dB on VCTK, against natural speech's −18.5 dB and −20.4 dB.

**The cascade's sign follows the extender's sign on genuine speech.**

- LibriTTS: NVSR costs genuine speech 0.067 SIM-o and 0.041 CAM++, and the cascade costs 0.055
  and 0.034. The cascade's shortfall is about the size of the penalty the verifier puts on
  NVSR's band.
- VCTK: NVSR is neutral on genuine speech under SIM-o and gains 0.042 under CAM++. The cascade
  gains 0.050 and 0.065.

**Why VCTK differs: NVSR was trained on these speakers.**

- NVSR was trained on VCTK 0.92, holding out only p360, p361, p362, p363, p364, p374, p376 and s5
  (ssr_eval README). 72 of the 76 VCTK evaluation items belong to speakers in NVSR's training
  set. Their recordings, probably including these exact utterances, were training targets.
- Splitting VCTK by whether NVSR saw the speaker:

| mean difference | SIM-o seen (n=72) | SIM-o held out (n=4) | CAM++ seen | CAM++ held out |
|---|---|---|---|---|
| NVSR − natural 8 kHz | −0.002 | −0.093 | +0.049 | −0.073 |
| cascade − stock 8 kHz | +0.055 | −0.041 | +0.073 | −0.062 |

- On the four held-out speakers, NVSR and the cascade behave as they do on LibriTTS. The
  difference between seen and held-out items is 0.09–0.14 in every cell. One-sided permutation
  p-values, reassigning the four held-out labels at random over the 76 items (20,000 draws), are:
  - NVSR − natural 8 kHz: SIM-o 0.0001, CAM++ 0.006.
  - cascade − stock 8 kHz: SIM-o 0.020, CAM++ 0.002.
- The held-out group is four speakers, so speaker traits other than NVSR's exposure cannot be
  excluded. The permutation treats items as exchangeable and does not address that. With that
  caveat, the likeliest reading is that an extender which has learned these speakers adds a
  band the verifiers accept, and one that has not adds a band they penalise.

**Consequences.**

- **Every VCTK cascade number in this repository is contaminated.** The extender was trained on
  72 of the 76 evaluation speakers. The cascade's apparent strength on VCTK, relative to
  LibriTTS, is at least partly this leakage. Those numbers should be marked as such, or the VCTK
  evaluation should be restricted to held-out speakers.
- **Earlier cascade reading, corrected.** The cascade's 8 kHz shortfall on LibriTTS (0.047
  documented, 0.055 in this rerun) should be compared with NVSR's genuine-speech penalty of
  0.067, not AudioSR's 0.041. It is roughly the size of the penalty on the added band. That
  suggests most of the cascade's SIM-o loss on LibriTTS is the verifier reacting to NVSR's band,
  not the model losing identity. The two effects are not shown to be additive, so this is an
  approximate reading.
- **Evidence bearing on the conjecture above.** Control 2 left open whether a speaker-specific
  high band could recover the metric's part. The VCTK split is an uncontrolled natural
  experiment. An extender exposed to the speaker does not incur the penalty, and under CAM++ it
  gains. This fits the conjecture but does not test it cleanly. That would need an extender
  trained with and without the evaluated speakers, everything else fixed. It stays a conjecture
  in the discussion.
- **The verifiers still split.** On genuine VCTK speech, the seen-speaker NVSR band is neutral
  for SIM-o and worth +0.049 to CAM++. Which verifier is used again changes whether an extender
  looks helpful.

### Against Carbonneau et al. (arXiv 2507.02176, SSW13 2025)

This is the closest prior work. It was read from the arXiv v1 HTML.

**What they did.** They ask whether ASV embeddings are sound for speaker-similarity scoring.
Each speaker's genuine recordings are split into an untouched group and a perturbed group,
which stands in for "synthesised". The score is the EER for telling the two groups apart:
50% means the perturbation is invisible, lower means the embedding reacts to it.

- Data: ARCTIC (18 speakers) and L2-ARCTIC (24). LibriSpeech is used only for a probe that
  predicts handcrafted features from embeddings.
- Models: seven. They include WavLM Large + ECAPA-TDNN from the UniSpeech
  speaker-verification code, the family behind SIM-o, although the paper never mentions
  Seed-TTS-eval.
- Perturbations:
  - utterance duration (short against long files)
  - white noise at 40, 20 and 0 dB SNR
  - pre-emphasis (1 − 0.97 z⁻¹) and its inverse
  - both of the above after re-equalising with a 16-band FIR matched to the genuine PSD
- Findings: embeddings encode duration, and they react to noise and spectral tilt; for
  WavLM-L+ECAPA, de-emphasis gives an EER of 0.32. Re-equalising restores about 50%. The
  embeddings capture timbre and pitch but not rhythm, and the paper proposes U3D for rhythm.
- Recommendations:
  - match durations and SNR between genuine and synthesised speech
  - re-equalise system output to the genuine spectral balance
  - report scores alongside genuine–genuine context rather than as raw cosines

**What they did not do.**

- No low-pass filtering, bandwidth reduction, resampling or codec condition appears. No
  sampling rate or cutoff is mentioned anywhere.
- No TTS or VC output is scored.
- No system's reported similarity drop is split into a metric part and a model part.

**What this control adds.**

1. **Bandwidth as a confound.** Cutting genuine speech to 8 kHz costs 0.075–0.086 SIM-o with
   the Seed-TTS-eval verifier, while 16 kHz costs nothing. Their confound list does not
   include it. Their re-equalisation remedy cannot reach it either, because equalisation
   cannot restore a band that is absent.
2. **Applied to a real system's reported number.** Their perturbed genuine audio is a proxy
   for synthesis. Here the matched-degradation natural score is set against stock ZipVoice's
   actual 8 kHz SIM-o on two corpora. That splits the drop into about 0.08 metric and 0.03–0.04
   identity lost, and gives a ceiling of about 0.65 for a perfect cloner at an 8 kHz prompt.
3. **The genuine–genuine context they ask for, at matched bandwidth.** Their advice to read
   raw scores against genuine pairs is followed here, but at the prompt's bandwidth rather
   than at full band. That choice is what changes the conclusion.
4. **A penalty on added content that equalisation cannot remove.** A generic high band
   added to genuine speech lowers SIM-o. Matching that band to the natural spectrum with an
   oracle 16-band target, their remedy transposed to bandwidth, removes only 12–35% of the
   penalty. The rest is a reaction to the band's content.
5. **Session matching as a second confound in the genuine–genuine context.** A natural
   reference from the prompt's own recording session scores 0.08–0.09 higher than one from
   another session, and a cloner that copies the prompt's channel can beat a different-session
   natural reference. Their context recommendation therefore needs the session held fixed.

**EER companion, in their format.**

- **Data:** LibriTTS dev-clean. 32 speakers have at least 20 utterances of 4–10 s; the other
  eight do not and are left out.
- **Split:** for each speaker, 20 utterances are drawn (seed 0) and split at random into a
  genuine half and a perturbed half of 10 each.
- **Trials:**
  - Target trials are all 45 pairs within the genuine half.
  - Non-target trials pair each genuine file with each perturbed file, 100 in all.
- **Scoring:** EER is computed per speaker, then averaged, with the standard deviation over
  speakers, as in their Table 2. An EER of 0.50 means the verifier cannot detect the condition.
- **Conditions:**
  - Emphasis and de-emphasis use their filter. RMS is matched back to the original afterwards;
    their paper does not say whether they did this.
  - White noise is added to the clean perturbed halves at 40, 20 and 0 dB SNR (seed 0), on the
    same 32 speakers, so the bandwidth rows can be ordered against noise within one corpus.
  - The bandwidth conditions use control 1's resampling.
  - NVSR and AudioSR each restore the 8 kHz halves. AudioSR output is trimmed to the
    reference length (median padding 2.2 s).
- **Paired contrasts:** each condition uses the same speakers and the same genuine half, so
  differences are taken per speaker and bootstrapped over speakers (10,000 draws, seed 0).
- **Scripts** (session scratchpad): `eer_prep.py`, `eer_noise.py`, `eer_run.sh`,
  `eer_run2.sh`, `eer_audiosr_now.sh`, `eer_score.py` and `eer_paired.py`.

This is Table 1 of the paper.

| condition | theirs, WavLM-L+ECAPA (ARCTIC + L2-ARCTIC) | SIM-o (WavLM-L+ECAPA) | CAM++ |
|---|---|---|---|
| same speaker, random split | 0.50 ± 0.02 | 0.49 ± 0.06 | 0.49 ± 0.05 |
| + emphasis | 0.43 ± 0.03 | 0.41 ± 0.07 | 0.49 ± 0.05 |
| + de-emphasis | 0.32 ± 0.06 | 0.36 ± 0.10 | 0.46 ± 0.08 |
| + white noise, 40 dB SNR | 0.48 | 0.48 ± 0.05 | 0.47 ± 0.05 |
| + white noise, 20 dB SNR | 0.38 | 0.40 ± 0.06 | 0.33 ± 0.08 |
| + white noise, 0 dB SNR | 0.10 | 0.12 ± 0.11 | 0.06 ± 0.11 |
| cut to 16 kHz | – | 0.49 ± 0.06 | 0.49 ± 0.05 |
| **cut to 8 kHz** | – | **0.25 ± 0.11** | **0.15 ± 0.12** |
| **cut to 8 kHz, NVSR-restored** | – | **0.14 ± 0.13** | **0.11 ± 0.12** |
| **cut to 8 kHz, AudioSR-restored** | – | **0.19 ± 0.11** | **0.13 ± 0.11** |

Paired per-speaker differences in EER (n = 32; negative means the first condition is more
detectable):

| contrast | SIM-o | CAM++ |
|---|---|---|
| 8 kHz cut − de-emphasis | −0.107 [−0.141, −0.077] | −0.313 [−0.361, −0.262] |
| 8 kHz cut − noise 20 dB | −0.150 [−0.179, −0.123] | −0.177 [−0.208, −0.145] |
| noise 0 dB − 8 kHz cut | −0.134 [−0.162, −0.105] | −0.088 [−0.114, −0.064] |
| NVSR-restored − 8 kHz cut | −0.119 [−0.150, −0.087] | −0.043 [−0.065, −0.022] |
| AudioSR-restored − 8 kHz cut | −0.068 [−0.099, −0.040] | −0.023 [−0.045, −0.003] |

For NVSR-restored − 8 kHz cut, the restored half is more detectable for 81% of speakers under
SIM-o (equal for 16%) and for 66% under CAM++ (equal for 22%). For AudioSR-restored − 8 kHz
cut, the shares are 78% (equal 6%) and 66% (equal 12%).

**Reading.**

- **The implementation is calibrated against theirs.** The random-split control, both
  emphasis rows and all three white-noise rows land within their standard deviations (or
  within 0.02 where they report none) for the same model family, on a different corpus.
- **Bandwidth is a stronger confound than anything in their tilt set.** Within this corpus
  and on the same speakers, the 8 kHz cut orders the same way under both verifiers: more
  detectable than de-emphasis and than 20 dB white noise, less detectable than 0 dB white
  noise. All three paired differences exclude zero under both verifiers. A 16 kHz cut is
  invisible, which agrees with control 1.
- **Restoring the band makes the audio more detectable, not less.** Three of the four steps
  carry this claim: NVSR under SIM-o, −0.119 [−0.150, −0.087]; NVSR under CAM++, −0.043
  [−0.065, −0.022]; and AudioSR under SIM-o, −0.068 [−0.099, −0.040]. The fourth, AudioSR
  under CAM++ (−0.023 [−0.045, −0.003]), only agrees in direction, and is not a finding on its
  own. Restored audio is easier to detect for 66–81% of speakers across all four. The steps
  agree in direction with controls 2 and 2b: to the verifier, the added band is not genuine.
- **Training-data caveat for the two extenders.**
  - NVSR was trained on VCTK only, so it has not seen these LibriTTS speakers.
  - The AudioSR paper says its speech is "the speech dataset from OpenSLR", downloaded "by
    following the link provided by VoiceFixer". LibriTTS is also hosted on OpenSLR, so the
    text alone cannot rule out overlap.
  - The link VoiceFixer's training repository gives is Zenodo record 5546723. Its speech
    archive (`Train_set_speech.zip`, 5.6 GB) was listed from its central directory without
    downloading it. It holds 43,873 VCTK 0.92 mic1 files from 108 speakers and nothing else. No
    LibriTTS file is in it.
  - VoiceFixer's paper also names HQ-TTS, 191 hours of 44.1–48 kHz speech from OpenSLR. Its
    component list is an image in the paper and was not checked. LibriTTS is a 24 kHz corpus,
    which makes it an unlikely member of a 44.1–48 kHz collection, but this is not verified.
  - Even with overlap, the conclusion cannot flip. The VCTK split in control 2b showed that
    seen speakers make an extender's band more acceptable to both verifiers, not less. Any
    overlap would therefore shrink AudioSR's penalty, and the measured penalty is a
    conservative estimate.
- **Tilt sensitivity varies across verifiers; bandwidth sensitivity does not.** CAM++ is close
  to immune to tilt (0.46–0.49), which is consistent with its per-utterance mean normalisation
  of fbank features. It is the more sensitive of the two to bandwidth (0.15). This extends
  Carbonneau et al.'s own observation that sensitivity differs widely across their seven
  verifiers (GE2E, for example, fails badly); CAM++ was not among them. The bandwidth confound,
  by contrast, holds for both verifiers tested here.

**Still to close before this comparison is safe.**

- The second verifier and the band-limit check on stock's output are done (see the paired
  rerun above). The bandwidth effect replicates under CAM++ at twice the size.
- The EER companion is done (above), including the white-noise rows and both extenders
  (NVSR and AudioSR).
- Control 2 is done: a post-hoc extender (AudioSR) does not recover the metric part and
  lowers SIM-o. The level-matched run shows the penalty survives oracle re-equalisation.
- Control 2b is done: NVSR's band is also penalised on unseen speakers, and level-matching
  does not change that. On speakers NVSR was trained on, the penalty is absent.

## Listening test: design (not yet run)

**Purpose.** The controls above establish what two verifiers do. Only listeners can say
whether those reactions match human judgement of who is speaking. Each verifier effect is
therefore turned into a paired comparison that listeners judge directly.

**Contrasts.** All are within-item pairs, and each has a verifier prediction to test.

| id | A vs B | what it tests | SIM-o Δ (A − B) | CAM++ Δ |
|---|---|---|---|---|
| C1 | natural 24 kHz vs natural 8 kHz | bandwidth effect on genuine speech | +0.086 | +0.150 |
| C2 | AudioSR-restored 8 kHz vs natural 8 kHz | penalty on a generic added band | −0.041 | −0.011 |
| C3 | NVSR-restored 8 kHz vs natural 8 kHz | the same, for the cascade's extender | −0.067 | −0.036 |
| C4 | stock 8 kHz prompt vs natural 8 kHz | stock's shortfall at matched bandwidth | −0.041 | +0.029 |
| C5 | cascade vs stock 8 kHz prompt | the cascade on a real system | −0.055 | −0.035 |
| C6 | stock 24 kHz prompt vs natural 24 kHz | channel-copy bonus, by session | see below | see below |
| C6 − C4 | (derived, same items) | identity lost to the 8 kHz prompt | see below | see below |

The Δ values are from LibriTTS, over all 76 items, scored on the loudness-normalised stimuli
that listeners hear (next paragraph). VCTK is excluded, because NVSR was trained on 72 of its
76 evaluation speakers (and AudioSR probably on all 76), so C2, C3 and C5 would be
contaminated there.

C4 is not the identity-loss measure. It compares stock's output with a different natural
utterance, so it carries the channel-copy bonus as well as any identity loss (C6 shows that
bonus at full band). Identity lost is C6 − C4, below.

**Rescored on the listeners' audio.** The stimuli are normalised to −23 LUFS and the verifiers
had scored the unnormalised files. Both verifiers were therefore rescored on normalised copies
of all 76 items in every condition, with the reference prompt normalised the same way (CPU;
`norm_copy.py`, `rescore_norm.sh` and `compare_norm.py` in the session scratchpad). The
predictions in this section are from that rescoring.

- SIM-o does not move. The per-item change has a median of 0.0002 and a maximum of 0.002, and
  every condition mean and contrast is unchanged to three decimals.
- CAM++ moves a little. The per-item median change is 0.001 and the maximum is 0.07. Contrasts
  against natural 8 kHz shift by up to 0.007: C1 +0.144 → +0.150, C2 −0.018 → −0.011 (its
  interval now includes 0), C3 −0.041 → −0.036 and C4 +0.022 → +0.029. C5 and C6 are
  unchanged.
- **The level hypothesis for CAM++'s offset is rejected.** CAM++ − SIM-o on C6 is +0.048
  [+0.034, +0.062] before and after, +0.044 on same-chapter items and +0.052 on
  different-chapter items. Loudness does not produce the offset.
- CAM++'s identity lost shrinks slightly: 0.023 [0.009, 0.037] → 0.017 [0.002, 0.032] over all
  76 items. SIM-o's stays 0.038 [0.024, 0.052].
- On the pilot's 24 items the contrasts are within 0.02 of the 76-item values, except SIM-o's
  C4, which is −0.062 there.

**C6's prediction, set before the test runs.** C6 is analysed with session (same or different
chapter) as a factor, and the stimuli are stratified 12/12 by it. The verifiers make two
predictions that can be tested separately:

| C6 quantity | SIM-o | CAM++ | verifiers |
|---|---|---|---|
| different chapter: stock − natural | +0.031 | +0.082 | agree in sign |
| same chapter: stock − natural | −0.041 | +0.004 | disagree: SIM-o says stock is worse, CAM++ says a tie |
| session effect (different − same) | +0.072 | +0.079 | agree in sign and size |
| pooled, at the 12/12 design mix | −0.005 | +0.043 | disagree |

- The session effect is the prediction both verifiers share. If listeners do not show it,
  the session bonus is a verifier artefact.
- The same-chapter row decides between the verifiers. There the channel is held constant, so
  only the voice differs.
- The table is from the normalised stimuli. Normalisation moves no entry by more than 0.001.

**Identity lost, the pre-registered human test.** Identity lost is C6 − C4 per item: stock's
shortfall against natural speech at an 8 kHz prompt, minus its shortfall at full band. Both
contrasts are oriented stock − natural, so a positive value means identity lost to the
degraded prompt. The channel-copy bonus appears in both and cancels. This is the same
difference of differences as the verifiers' identity-lost row.

- The Latin square places every item in both C4 and C6, rated by different raters (C4 and C6
  use different blocks within a list). The per-item difference is therefore between two item
  means from independent raters, and its interval comes from the item bootstrap.
- It is analysed by session, because the verifiers put most of it on same-chapter items:

| identity lost (C6 − C4) | SIM-o | CAM++ |
|---|---|---|
| same chapter (n=36) | 0.057 [0.036, 0.078] | 0.021 [−0.003, 0.043] |
| different chapter (n=40) | 0.021 [0.002, 0.040] | 0.013 [−0.006, 0.031] |
| all 76 | 0.038 [0.024, 0.052] | 0.017 [0.002, 0.032] |

- The verifiers are on a cosine scale and listeners on a CCR scale, so the prediction is about
  sign and presence, not size. SIM-o predicts identity lost that is heard, larger on
  same-chapter items. CAM++ predicts a smaller loss with no clear session difference.
- At the 24 pilot items with a 12/12 mix, each session group has 12 items. The session split
  is reported with its interval but is not expected to be decisive in the pilot.

**Task.** Use a comparison category rating (ITU-T P.800 CCR) against a reference.

- The listener hears the reference, which is the clean 24 kHz prompt the verifiers score
  against, then A and then B, in random order.
- They answer "Which one sounds more like the same person as the reference, and by how much?"
  on a −3 to +3 scale, with 0 meaning no difference.
- The instructions say to ignore recording quality, bandwidth and loudness, and to judge only
  the voice.
- The paired design matches the verifiers' paired bootstrap, and it is more sensitive than
  absolute similarity MOS for effects of this size.
- A separate block asks for quality (ACR, 1–5) on the same stimuli. This checks whether
  similarity judgements are leaking quality, which is the listener's version of the
  confound being studied. Each list rates all 7 systems on its own 8 items (56 clips), so over
  the three lists every item is rated in every system and each CCR pair has a per-item quality
  difference.

**Stimuli.**

- 24 LibriTTS items are drawn at random, stratified 12/12 by same or different chapter.
- All stimuli are loudness-normalised to −23 LUFS, and the verifier predictions are rescored
  on these normalised files, so listeners and verifiers judge the same audio. Durations are
  left as generated and differ by system (a stock output runs 7.4 s against 9.3 s for the
  matching natural clip); trimming would change what the verifiers scored.
- Two kinds of anchor trials check attention:
  - identical A and B, where the expected answer is 0
  - a different-speaker B from the same corpus, where the expected answer is a strong
    preference for A
- A rater who fails more than one of the four anchors is excluded.

**Allocation and raters.**

- The design has 6 contrasts × 24 items = 144 trials. They are split into 3 Latin-square lists
  of 48 trials plus 4 anchors. Each list covers every contrast on 8 items, and every item
  appears in every contrast across the lists.
- Aim for 20 raters per list, 60 in total, giving 20 ratings per item-contrast.
- Recruit native English listeners on Prolific. Use a headphone screening test (Woods et al.
  2017) before the task. About 40 minutes per rater.

**Analysis (to pre-register).**

- Ratings are oriented as preference for A (A − B). Raters who fail the headphone check (fewer
  than 5 of 6) or more than one of the four anchors are excluded before anything else.
- Each contrast's mean is estimated at item level: the per-item mean over raters, with a 95%
  interval from 10,000 bootstrap resamples of the 24 items. One resample is shared by all
  contrasts, so C4 and C6 stay paired within items. Rater and item variance components come
  from two-way ANOVA within each list × contrast table and are used to size the main study.
- The smallest effect of interest is 0.3 CCR points. For each contrast, and for identity lost:
  - an interval above +0.3 or below −0.3 means listeners hear an effect
  - an interval inside ±0.3 (equivalence) means listeners do not
  - anything else is inconclusive
- The main reading is agreement in sign and presence with each verifier. This decides the open
  questions directly:
  - whether C1's bandwidth effect is heard as a different voice (metric artefact or not)
  - whether C2 and C3's added-band penalty is heard
  - **identity lost (C6 − C4):** whether listeners hear it, and which verifier's figure is closer
    to what they hear. Reported over all items and by session (same, different, and same −
    different), against the table above.
  - C6 by session, against its prediction table.
- **Quality leakage, decided before any pilot data exist.** For each item and contrast, dq is
  the ACR difference A − B, each side the mean over kept raters for that item and system. The
  item-level model is
  CCR[item, c] = α_c + β · dq[item, c],
  with one intercept per contrast and one slope shared by all contrasts, so β is identified
  from within-contrast variation between items. α_c is contrast c's similarity effect at equal
  rated quality.
  - Report β with its interval, and each α_c with its interval from the same item bootstrap.
  - Read each α_c with the same ±0.3 rule, and report whether the reading changes from the
    unadjusted one. A similarity effect that survives the adjustment is not quality leakage.
  - A contrast whose observed dq range excludes 0 is flagged: its α_c extrapolates beyond the
    data and is reported but not read.
- A secondary analysis correlates the per-item mean CCR with the per-item verifier Δ, per
  contrast and per verifier.

**What it costs.** About 60 raters × 40 minutes. The stimuli exist already (control 1, control
2, 2b and the resynthesised stock and cascade outputs), and the normalised copies and anchor
pairs are built by the pilot's script.

**Pilot (built, not yet run).** An offline bundle for 6–8 raters is in
`C:/zipvoice_data/listening_pilot/`, and its `README.md` covers the build, sending and
analysis. It implements the design above: 3 lists of 48 test trials, 2 identical pairs and 2
same-gender foils, 56 ACR clips, the Woods headphone check and −23 LUFS normalisation.
`tools/pilot_analyse.py` runs the pre-registered analysis above (contrast means, C6 and
identity lost by session, quality leakage). It also estimates the variance components with
two-way ANOVA within each list × contrast table, and gives the CI half-width for each contrast
mean by raters per list and items per contrast. The main study's size is set from that output,
not from the 20-per-list guess above.

`tools/simulate_pilot.py` checks the analysis on data with known answers. With 20 raters per
list and a quality leak of 0.5 CCR points per ACR point, the fitted slope is 0.46 [0.30, 0.61],
and the adjusted contrasts land within 0.11 of the true effects. With no leak, the slope is
0.06 [−0.08, 0.18].

On simulated data, the item term sets a precision floor that more raters cannot lower. With
an item SD of 0.5, a contrast mean over 24 items has a 95% CI half-width of at least 0.18 at
any rater count. If the pilot shows an item SD of that size, the main study needs more items
rather than more raters.

Ethics approval is still to be confirmed. The page shows placeholder text for the approval
reference and the consent text until they are filled in.

## Results

n = 76 per condition. SIM-o p-values come from a paired t-test over per-utterance cosine
similarities.

| prompt rate | SIM-o bwe | SIM-o stock | Δ SIM | p | WER bwe | WER stock | UTMOS bwe | UTMOS stock | Δ UTMOS |
|---|---|---|---|---|---|---|---|---|---|
| 8000 | 0.586 | 0.610 | −0.023 | 0.014 | 2.52% | 2.92% | **4.22** | 3.89 | **+0.33** |
| 16000 | 0.722 | 0.730 | −0.008 | 0.158 | 3.24% | 2.76% | **4.19** | 4.08 | +0.11 |
| 22050 | 0.732 | 0.741 | −0.009 | 0.088 | 3.06% | 3.09% | 4.17 | 4.17 | 0.00 |
| 24000 (control) | 0.729 | 0.739 | −0.010 | 0.094 | 2.65% | 2.99% | 4.16 | 4.20 | −0.04 |

## Reading the numbers

**UTMOS is where the extender works.** The gain is monotone in the amount of degradation:
+0.33 at 8 kHz, +0.11 at 16 kHz, zero at 22.05 kHz, −0.04 at the control. That is exactly
the shape a working bandwidth extender produces. Stock ZipVoice falls to 3.89 on 8 kHz
prompts; the extender holds 4.22, which is above its own 24 kHz score.

**SIM-o shows no benefit and a small consistent penalty.** The delta is negative at every
rate, including the control. The 8 kHz result reaches p = 0.014, but it is one of four
uncorrected comparisons and the Bonferroni threshold would be 0.0125, so it is borderline
rather than established.

**The control condition is what makes SIM interpretable.** At a 24 kHz prompt,
`infer_zipvoice.py` computes `cutoff_hz = min(prompt_sampling_rate, sampling_rate) / 2`
= 12000 Hz, which is Nyquist, so `band_limit` is a no-op and the Eq. 10 hard composite
copies every observed bin through bit-exact. The extender is *identity* in that condition.
The −0.010 SIM and −0.04 UTMOS there therefore measure the 15000-step LibriTTS
fine-tuning of the ZipVoice weights, not the extender. Netting that offset out, the
extender-attributable SIM change is roughly −0.013 at 8 kHz and approximately zero
elsewhere.

**Interpretation.** The extender synthesises a high band that is plausible but generic
rather than speaker-specific: enough to remove the audible dullness a dead upper band
causes (UTMOS), not enough to carry the identity information the speaker-verification
model reads (SIM-o).

**WER carries no signal here.** All eight conditions land between 2.5% and 3.2%, with no
ordering by rate or by system. This was an eyeballed judgement when first written; it is
tested properly, per utterance and with a paired test, under [WER, tested rather than
eyeballed](#wer-tested-rather-than-eyeballed), which confirms it.

## Why the identity information is missing: two dead discriminators

Stage 7 validation at the final checkpoint:

```
global_batch_idx: 15000, validation: fm_loss=0.03527, bwe_recon=0.1181,
adv_spec=0.6429, fmatch_spec=0.01457, adv_cross=0.2285, fmatch_cross=0.002753,
adv_harm=0.3925, fmatch_harm=0.002529,
d_spec=0.6573, d_cross=0.9802, d_harm=0.9585, d_loss=2.596
```

The hinge discriminator loss at the trivial `D = 0` point is **1.0 per view** (3.0 for the
three-view sum); perfect separation is 0.0. Measured against that ceiling:

- `d_spec` 0.657 — working.
- `d_cross` 0.980 — essentially dead across the whole of stage 7 (range 0.937–0.983).
- `d_harm` 0.959 — essentially dead (range 0.914–0.998; it touched 0.998, momentarily
  worse than a zero-output discriminator).

Only one of the three adversarial signals ever functioned, so the only pressure toward
speaker-specific high-band detail came from `D_spec`. This is the evidence the plan asked
for before dropping `D_harm`, and it applies equally to `D_cross`.

`fm_loss` on held-out data was flat to the fourth decimal over the last ~4000 steps
(0.03551, 0.03523, 0.03536, 0.03545, 0.03542, 0.03538, 0.03527), so stage 7 had converged
on the one metric aligned with the actual objective.

## Ablation: separating the extender from the fine-tuning

The bwe-vs-stock comparison above changes two things at once, so it cannot attribute a
result to either. `--bwe-bypass` fixes that by running the *same* `iter-15000-avg-2.pt`
weights with part or all of the extender switched off:

| arm | prompt mel handed to `speech_condition` |
|---|---|
| `bwe` | the extender as trained |
| `bwe_bypass_resunet` | band-limited and re-floored, ResUNet never predicts into the dead band |
| `bwe_bypass_full` | untouched — bit-for-bit the same input stock ZipVoice receives |
| `stock` | untouched, released Emilia checkpoint |

`bwe` vs `bwe_bypass_full` therefore isolates the extender with the TTS weights held
fixed, and `bwe_bypass_full` vs `stock` isolates the LibriTTS fine-tuning with the input
held fixed.

### A noise floor comes free with the control

At a 24 kHz prompt the three extender arms are provably fed the same mel, and two of them
are bit-identical in output: `bwe` and `bwe_bypass_resunet` produce byte-identical audio
samples (only a 64-byte WAV header chunk differs), which is the strongest possible
confirmation that `band_limit` is a no-op at a Nyquist cutoff and that the Eq. 10 hard
composite copies observed bins through exactly.

`bwe_bypass_full` returns before `restore()` is called, which shifts the sampling RNG
stream, so it is the same input under a different noise draw. Its distance from `bwe` at
24 kHz is pure sampling noise, and it calibrates every other number in this document:

| metric | `bwe` @ 24 kHz | `bwe_bypass_full` @ 24 kHz | noise floor |
|---|---|---|---|
| SIM-o | 0.7294 | 0.7374 | 0.008 (p = 0.097) |
| UTMOS | 4.16 | 4.17 | 0.01 |
| Seed-TTS WER | 2.65% | 3.46% | 0.8 pp |

Per-utterance the noise is much larger than the means suggest — individual SIM
differences between two noise draws of the same input reach 0.128. **Every WER difference
reported in this document is inside the 0.8 pp noise floor**, which retires WER as a
signal here rather than merely finding it uninformative. The later per-utterance treatment
puts that floor at 0.6 pp measured on the paired differences rather than on the gap between
two condition means, and reaches the same conclusion from stronger evidence.

### UTMOS decomposition

| prompt rate | stock | `bwe_bypass_full` | `bwe` | fine-tuning | extender |
|---|---|---|---|---|---|
| 8000 | 3.89 | 4.02 | **4.22** | +0.13 | **+0.20** |
| 16000 | 4.08 | 4.17 | 4.19 | +0.09 | +0.02 |
| 22050 | 4.17 | 4.19 | 4.17 | +0.02 | −0.02 |
| 24000 (control) | 4.20 | 4.17 | 4.16 | −0.03 | −0.01 |

The +0.33 headline gain at 8 kHz splits into +0.13 from LibriTTS fine-tuning and **+0.20
from the ResUNet itself**, twenty times the 0.01 noise floor. At 16 kHz and above the
extender contribution collapses to within noise and the small residual gain is the
fine-tuning. So the extender is doing real work, and it does it only where the
degradation is severe — which is the correct behaviour for a bandwidth extender, not a
uniform offset.

### SIM-o decomposition

Paired t-test over per-utterance cosine similarities, n = 76.

| prompt rate | stock | `bwe_bypass_full` | `bwe` | fine-tuning Δ (p) | extender Δ (p) |
|---|---|---|---|---|---|
| 8000 | 0.6098 | 0.6090 | 0.5863 | −0.0008 (0.913) | **−0.0227 (0.019)** |
| 16000 | 0.7304 | 0.7238 | 0.7221 | −0.0066 (0.194) | −0.0017 (0.756) |
| 22050 | 0.7409 | 0.7348 | 0.7322 | −0.0061 (0.194) | −0.0026 (0.577) |
| 24000 (control) | 0.7394 | 0.7374 | 0.7294 | −0.0020 (0.672) | −0.0080 (0.097) |

The fine-tuning costs nothing measurable on speaker similarity at any rate. The −0.023 at
8 kHz belongs to the extender, is roughly three times the noise floor, and is the only
SIM effect in the study that survives.

**This is a real trade-off, and both halves of it are now attributable to the ResUNet:**
at an 8 kHz prompt it buys +0.20 UTMOS and costs −0.023 SIM-o. It synthesises a high band
that sounds right and belongs to no one in particular, which is exactly what a
reconstruction loss plus one working spectrogram discriminator (see the previous section)
would be expected to produce.

### The `resunet` arm: the TTS now depends on the extender

Re-imposing the training-time dead-band floor without letting the ResUNet predict into it
does not degrade synthesis, it destroys it:

| prompt rate | SIM-o | Seed-TTS WER | UTMOS |
|---|---|---|---|
| 8000 | 0.0505 | 192.89% | 1.28 |
| 16000 | 0.1337 | 104.03% | 1.42 |
| 22050 | 0.4362 | 36.32% | 2.89 |
| 24000 (control) | 0.7294 | 2.65% | 4.16 |

The transcripts are Whisper's silence hallucinations — ` thank you`, ` 🎵 🎵`,
` i'm sorry`, empty — so the output is not speech at all. WER above 100% comes from a few
utterances where Whisper loops on noise; 1713 insertions and 983 deletions over 1754
reference words at 8 kHz.

Three facts make this a finding rather than a bug: it is monotone in how much of the band
is floored, it vanishes exactly at 24 kHz where `band_limit` provably does nothing, and
the 24 kHz row is byte-identical to the `bwe` arm.

What it shows is that 15000 steps of joint training left ZipVoice *dependent* on the
extender's output distribution: a hard log-floor at the `speech_condition` input is now
far out of distribution, because the ResUNet always filled that region during training.
That is a strong statement about the ResUNet being load-bearing, but it is not a measure
of prediction quality — the input is a degradation the model has never seen, harsher than
anything the `full` arm or stock ZipVoice is asked to handle. `bwe` vs `bwe_bypass_full`
remains the honest comparison.

It is also a deployment note: this checkpoint must not be run with the extender disabled
on a band-limited prompt.

## Out-of-domain replication on VCTK

Everything above is measured on LibriTTS dev-clean, and the extender and the ZipVoice
weights were both fine-tuned on LibriTTS train-clean. A result on that corpus cannot
distinguish "the extender restores bandwidth" from "the model adapted to this corpus," so
the whole ablation was rerun unchanged on VCTK 0.92 — a corpus neither component has ever
seen, and one that differs in every way that matters: 76 utterances from **76 distinct
speakers**, a head-mounted DPA in a hemi-anechoic room rather than audiobook recordings,
many non-American accents, and short read newspaper sentences rather than prose.

Protocol is identical: the same `iter-15000-avg-2.pt` checkpoint, the same four arms, the
same four prompt rates, 76 pairs, prompts resampled 48 kHz to 24 kHz first so the clean
reference is at ZipVoice's native rate and the degraded copies derive from that same
signal. Prompt and target always come from the same speaker but are different utterances.

Absolute SIM-o is about 0.11 lower across the board than on LibriTTS (0.62 vs 0.74 at the
24 kHz control). Natural VCTK speech scores 0.726 against natural LibriTTS speech's 0.740 on
the same evaluator, so this is not the metric being harder on VCTK: it is ZipVoice cloning
unfamiliar accents and a different recording chain less well. It applies equally to every
arm, so the within-corpus contrasts are unaffected.

### The noise floor is smaller here

Same construction as before: at a 24 kHz prompt `bwe` and `bwe_bypass_full` are fed the
same mel and differ only in RNG stream.

| metric | `bwe` @ 24 kHz | `bwe_bypass_full` @ 24 kHz | noise floor |
|---|---|---|---|
| SIM-o | 0.6220 | 0.6204 | 0.0016 (p = 0.796) |
| UTMOS | 4.18 | 4.16 | 0.02 |
| Seed-TTS WER | 0.76% | 1.17% | 0.4 pp |

### UTMOS

| prompt rate | stock | `bwe_bypass_full` | `bwe` | fine-tuning | extender | extender on LibriTTS |
|---|---|---|---|---|---|---|
| 8000 | 3.92 | 4.02 | **4.25** | +0.10 | **+0.23** | +0.20 |
| 16000 | 4.08 | 4.13 | 4.18 | +0.05 | +0.05 | +0.02 |
| 22050 | 4.19 | 4.17 | 4.14 | −0.02 | −0.03 | −0.02 |
| 24000 (control) | 4.17 | 4.16 | 4.18 | −0.01 | +0.02 | −0.01 |

**The 8 kHz gain replicates, and is slightly larger out of domain: +0.23 against a 0.02
noise floor.** The shape replicates too — the extender contributes only where the
degradation is severe and does nothing at 22.05 and 24 kHz. The LibriTTS fine-tuning
contributes +0.10 here against +0.13 there, so it also transfers rather than being a
corpus-specific effect.

### SIM-o

Paired t-test over per-utterance cosine similarities, n = 76.

| prompt rate | stock | `bwe_bypass_full` | `bwe` | fine-tuning Δ (p) | extender Δ (p) | extender on LibriTTS |
|---|---|---|---|---|---|---|
| 8000 | 0.5144 | 0.5016 | 0.4783 | −0.0128 (0.254) | **−0.0233 (0.030)** | −0.0227 (0.019) |
| 16000 | 0.6217 | 0.6120 | 0.6147 | −0.0096 (0.189) | +0.0027 (0.716) | −0.0017 (0.756) |
| 22050 | 0.6296 | 0.6249 | 0.6251 | −0.0046 (0.548) | +0.0002 (0.978) | −0.0026 (0.577) |
| 24000 (control) | 0.6209 | 0.6204 | 0.6220 | −0.0006 (0.936) | +0.0016 (0.796) | −0.0080 (0.097) |

The cost replicates as precisely as the benefit does: −0.0233 here against −0.0227 on
LibriTTS, both significant, both confined to 8 kHz. Against this corpus's smaller noise
floor it is roughly fifteen times the floor. At 16 kHz and above the extender's SIM effect
is indistinguishable from zero and the sign is no longer even consistent.

### WER

| prompt rate | stock | `bwe_bypass_full` | `bwe` |
|---|---|---|---|
| 8000 | 0.83% | 1.20% | 0.89% |
| 16000 | 0.76% | 1.26% | 0.98% |
| 22050 | 0.92% | 0.88% | 0.76% |
| 24000 | 1.15% | 1.17% | 0.76% |

The whole table spans 0.5 pp, inside the 0.4 pp floor. WER carries no signal here either.
Absolute WER is far below LibriTTS's 2.6–3.5% because VCTK sentences are short and
cleanly read, which makes the metric easier but no more informative.

### The `resunet` collapse replicates

| prompt rate | SIM-o | Seed-TTS WER | UTMOS |
|---|---|---|---|
| 8000 | 0.0450 | 189.21% | 1.28 |
| 16000 | 0.0601 | 97.19% | 1.31 |
| 22050 | 0.4314 | 29.98% | 3.09 |
| 24000 (control) | 0.6220 | 0.76% | 4.18 |

Same magnitudes as on LibriTTS, and at 24 kHz the per-utterance SIM difference against
`bwe` is exactly 0.0000 on all 76 pairs (paired *t* undefined), independently reconfirming
that `band_limit` is a no-op at a Nyquist cutoff. ZipVoice's dependence on the extender's
output distribution is therefore a property of the checkpoint, not of the LibriTTS eval
set.

### What this settles

The extender does real, transferable work. The single effect that survived the LibriTTS
ablation — **+0.20 UTMOS at an 8 kHz prompt for −0.023 SIM-o** — reproduces on a corpus
neither the extender nor the fine-tuning has seen, at **+0.23 for −0.023**, with the same
rate profile and the same significance pattern. It is not corpus memorisation.

It also confirms the shape of the limitation. The extender synthesises a high band that
*sounds* right everywhere it is asked to and belongs to the *right speaker* nowhere: the
UTMOS gain transfers and the SIM cost transfers with it, unchanged, onto 76 speakers the
model has never heard. That is exactly what reconstruction plus one working spectrogram
discriminator predicts, and it is the same conclusion the dead-discriminator analysis
reached from the training logs.

## DNSMOS: a second opinion on sound quality

UTMOS is one model with one opinion, and it supplies the entire headline result, so
everything above rests on a single predictor. DNSMOS is an independent check with a
different architecture, different training data and a different labelling protocol. It
also decomposes quality, which UTMOS cannot: ITU-T **P.835** rates speech distortion
(SIG), background intrusiveness (BAK) and overall quality (OVRL) separately, while
**P.808** is an undecomposed crowdsourced overall MOS in the same spirit as UTMOS.

All four arms were rescored on both corpora — 32 conditions, 76 utterances each —
with `zipvoice/eval/mos/dnsmos.py`.

**Scope limit, stated up front.** DNSMOS runs at 16 kHz, so it only ever observes content
below 8 kHz. For an 8 kHz prompt the extender predicts 4–12 kHz and DNSMOS sees the
4–8 kHz half of it. For a 16 kHz or higher prompt *everything* the extender predicts sits
above 8 kHz and is invisible. Only the 8 kHz row is a real test of the extender; the other
rows measure the rest of the synthesis and serve as controls.

### P.808 replicates the UTMOS result, on both corpora

Mean P.808 MOS, and the paired extender contrast (`bwe` − `bwe_bypass_full`, n = 76).

| | | VCTK | | | | LibriTTS | | |
|---|---|---|---|---|---|---|---|---|
| prompt rate | stock | `bypass_full` | `bwe` | extender Δ (p) | stock | `bypass_full` | `bwe` | extender Δ (p) |
| 8000 | 3.488 | 3.565 | **3.826** | **+0.261 (<0.0001)** | 3.539 | 3.648 | **3.949** | **+0.301 (<0.0001)** |
| 16000 | 3.886 | 3.853 | 3.825 | −0.028 (0.175) | 3.969 | 3.933 | 3.937 | +0.004 (0.798) |
| 22050 | 3.897 | 3.866 | 3.827 | −0.040 (0.045) | 3.969 | 3.931 | 3.945 | +0.014 (0.426) |
| 24000 (control) | 3.919 | 3.868 | 3.850 | −0.018 (0.379) | 3.984 | 3.938 | 3.949 | +0.011 (0.514) |

The 24 kHz control gives the noise floor as before — same input, different noise draw —
at 0.018 (VCTK) and 0.011 (LibriTTS) MOS, neither significant. **The 8 kHz extender gain
is 15 to 27 times that floor, at t = 9.7 and t = 12.4.** This is the strongest effect
measured anywhere in the study, and it is a genuinely independent confirmation: a
different MOS predictor, on a corpus the model never saw, agrees with UTMOS on both the
sign and the rate profile.

The one nominally significant negative, VCTK at 22.05 kHz (p = 0.045), does not replicate
on LibriTTS (+0.014) and sits in a bank of 32 tests where roughly 1.6 false positives are
expected at α = 0.05. Treating it as an effect would be over-reading.

### P.835 sees almost nothing, and that is informative

Extender contrast on the three P.835 axes at the rate where DNSMOS can actually see the
extender's output:

| corpus | SIG Δ (p) | BAK Δ (p) | OVRL Δ (p) |
|---|---|---|---|
| VCTK @ 8 kHz | +0.021 (0.104) | +0.013 (0.126) | +0.029 (0.066) |
| LibriTTS @ 8 kHz | +0.005 (0.649) | −0.007 (0.625) | +0.007 (0.638) |

Nothing reaches significance, and the LibriTTS values are indistinguishable from zero. The
absolute levels explain why: BAK sits at 4.11–4.14 out of 5 in *every* arm at *every*
rate, including stock ZipVoice on a clean prompt. P.835 was built to grade noise
suppressors, where the question is how much residual background survives and how much the
speech was mangled getting there. Synthesised speech has no background to suppress, so BAK
is saturated, and missing bandwidth is not "distortion" in the sense SIG was trained on.

So the three-axis decomposition is the wrong instrument here rather than a contradicting
result — which is worth knowing, because reporting DNSMOS OVRL alone would have made the
extender look ineffective. **P.808 and UTMOS agree; P.835 abstains.**

### A cost at 24 kHz that UTMOS could not resolve

P.808 is precise enough to expose something the earlier metrics could not. The LibriTTS
fine-tuning (`bwe_bypass_full` − `stock`, input held identical) is a *penalty* on
undegraded prompts:

| prompt rate | VCTK Δ (p) | LibriTTS Δ (p) |
|---|---|---|
| 8000 | +0.077 (0.0038) | +0.109 (<0.0001) |
| 16000 | −0.033 (0.076) | −0.037 (0.0087) |
| 22050 | −0.031 (0.119) | −0.037 (0.0059) |
| 24000 (control) | −0.051 (0.016) | −0.046 (0.0033) |

> **Superseded.** A stricter null condition added by Application study 1 — `plugplay`
> against `stock` at 24 kHz, which holds the weights bit-identical rather than only the
> input — produces P.808 deltas of −0.020 and −0.037 with no real effect present. The
> penalty below is only about 1.3x that, so it is suggestive at best and the significance
> claimed in this subsection is overstated. The consistent sign across four conditions is
> the part still worth noting.

Significant on both corpora at 24 kHz, same sign, same magnitude. It is small — about
0.05 MOS, and about a quarter of what the extender buys at 8 kHz — but it is real, and it
was inside UTMOS's noise floor (the corresponding UTMOS deltas were −0.03 and −0.01). The
guard stated in the summary, that the 24 kHz control does not significantly regress, holds
for UTMOS and SIM-o but **not** for P.808: 15000 steps of joint fine-tuning cost a little
quality on clean prompts. This belongs in the same bucket as the `resunet` collapse — the
checkpoint has been specialised toward band-limited input.

### The `resunet` collapse, once more

| prompt rate | SIG | BAK | OVRL | P.808 |
|---|---|---|---|---|
| 8000 | 1.217 | 1.544 | 1.108 | 2.158 |
| 16000 | 1.285 | 1.649 | 1.203 | 2.261 |
| 22050 | 2.854 | 3.314 | 2.614 | 3.275 |
| 24000 (control) | 3.560 | 4.120 | 3.307 | 3.850 |

VCTK figures; LibriTTS matches. P.835 has no trouble detecting *this*, which confirms the
axes are functioning and simply have nothing to say about bandwidth. SIG at 1.22 out of 5
is the floor of the scale.

### What DNSMOS adds

The extender's naturalness gain at 8 kHz is now supported by two independent MOS
predictors on two corpora, at 9 to 27 times the measured noise floor in every case. It is
not an artefact of UTMOS. Combined with the SIM-o result, the characterisation stands
unchanged and is now better evidenced: **the extender makes band-limited prompts sound
substantially better and does not make them sound like the right person.**

## Application study 1: plug-and-play deployment

Everything above compares a jointly fine-tuned checkpoint against stock ZipVoice, which
confounds two questions a user actually faces. The practical one is: **can the trained
extender be bolted onto an unmodified ZipVoice, or does it only work on weights that were
fine-tuned alongside it?** That is the missing cell of a 2x2:

| | stock TTS weights | LibriTTS-fine-tuned TTS weights |
|---|---|---|
| **no extender** | `stock` | `bwe_bypass_full` |
| **extender** | **`plugplay`** (new) | `bwe` |

The `plugplay` checkpoint is built offline by taking the released ZipVoice state dict and
adding the 106 `bwe.*` tensors from `iter-15000-avg-2.pt`. The key sets are disjoint and
exhaustive — 889 stock parameters against 889 non-`bwe` parameters in the trained
checkpoint, verified equal as sets — so the merge is exact and needs no code change: the
existing `--model-name zipvoice_bwe` path loads it directly. Every one of the 889 TTS
tensors differs between stock and the fine-tuned checkpoint, so this arm really does
isolate the extender from the fine-tuning.

### A third null condition, and a correction to the noise floor

At a 24 kHz prompt `plugplay` and `stock` are the *same function*: the TTS tensors are
bit-identical and the extender is provably identity at a Nyquist cutoff. They differ only
in that the BWE code path consumes the RNG stream differently, which is confirmed by the
audio — all 76 files differ, none byte-identically. So this pair is a second, independent
draw from the null distribution, and it is a stricter one than `bwe` vs `bwe_bypass_full`
because the weights are identical rather than merely the input.

| metric | LibriTTS Δ (p) | VCTK Δ (p) | floor claimed earlier |
|---|---|---|---|
| SIM-o | −0.0085 (**0.046**) | +0.0056 (0.438) | 0.008 / 0.002 |
| UTMOS | −0.05 | +0.06 | 0.01 / 0.02 |
| DNSMOS P.808 | −0.020 (0.178) | −0.037 (0.075) | 0.011 / 0.018 |

**A known-zero effect produced p = 0.046 on SIM-o.** That is the clearest possible
demonstration that at n = 76 a single paired t-test at α = 0.05 is not sufficient evidence
here, and it revises two earlier statements in this document:

- The UTMOS noise floor is about **0.05**, not 0.01–0.02. The +0.20 and +0.23 extender
  gains at 8 kHz are therefore roughly 4x the floor rather than 20x. They remain the
  largest and most consistent effect in the study, replicated on two corpora with two
  independent MOS predictors, but the "twenty times the noise floor" figure was an
  artefact of estimating the floor from one control condition.
- The 24 kHz P.808 fine-tuning penalty reported in the DNSMOS section (−0.046 LibriTTS,
  −0.051 VCTK) is only about 1.3x this null's magnitude. **It should no longer be read as
  established.** The direction is consistent across four conditions, which is still
  suggestive, but the significance was overstated.

Nothing about the 8 kHz results changes: those effects are 5 to 9 times the widened floor
and reach t = 8.8 to 12.4.

### The extender transfers to unmodified ZipVoice

Naturalness at an 8 kHz prompt, the only rate where the extender does measurable work:

| corpus | metric | `stock` | `plugplay` | `bwe` | plug-and-play share of the gain |
|---|---|---|---|---|---|
| LibriTTS | UTMOS | 3.89 | **4.14** | 4.22 | +0.25 of +0.33 — 76% |
| LibriTTS | P.808 | 3.539 | **3.860** | 3.949 | +0.322 of +0.410 — 78% |
| VCTK | UTMOS | 3.92 | **4.15** | 4.25 | +0.23 of +0.33 — 70% |
| VCTK | P.808 | 3.488 | **3.726** | 3.826 | +0.238 of +0.338 — 70% |

Both plug-and-play P.808 gains are overwhelming against the null: t = 12.36 (LibriTTS) and
t = 8.77 (VCTK), p < 0.0001. **Roughly three quarters of the extender's benefit is
available with no change to ZipVoice at all.** The extender is not merely exploiting an
adaptation of the TTS weights to its own output distribution; it produces a prompt mel that
the *released* model finds more useful than the band-limited one.

This also removes the deployment hazard documented for the `resunet` arm. That failure —
synthesis collapsing when the extender is bypassed on a band-limited prompt — is a property
of the fine-tuned checkpoint, which learned to depend on the extender always filling the
dead band. Stock weights never learned that dependence, so a plug-and-play deployment
degrades gracefully rather than catastrophically if the extender is switched off.

### Fine-tuning buys naturalness and costs identity

The remaining quarter of the gain comes from the joint fine-tuning, and it is not free.
Contrast `bwe` − `plugplay`, which holds the extender fixed and varies only the TTS
weights:

| prompt rate | LibriTTS SIM-o Δ (p) | VCTK SIM-o Δ (p) | LibriTTS P.808 Δ (p) | VCTK P.808 Δ (p) |
|---|---|---|---|---|
| 8000 | −0.0029 (0.747) | **−0.0331 (0.0039)** | **+0.088 (0.0007)** | **+0.100 (0.0001)** |
| 16000 | **−0.0137 (0.0028)** | **−0.0173 (0.021)** | −0.029 (0.052) | −0.027 (0.102) |
| 22050 | −0.0042 (0.322) | **−0.0175 (0.011)** | −0.029 (0.130) | **−0.054 (0.0017)** |
| 24000 (control) | −0.0015 (0.764) | −0.0045 (0.419) | −0.015 (0.438) | −0.032 (0.075) |

The pattern is consistent and the two corpora agree on the sign in every cell: fine-tuning
adds about +0.09 to +0.10 P.808 at 8 kHz and subtracts speaker similarity at every
band-limited rate, most severely out of domain. On VCTK the SIM cost is significant at
8 kHz, 16 kHz and 22.05 kHz — three of the four cells — while the 24 kHz control is not,
which is the profile of a real effect rather than a floor artefact.

The full SIM-o picture makes the trade explicit. At an 8 kHz VCTK prompt:

| arm | SIM-o | Δ vs `stock` |
|---|---|---|
| `stock` | 0.5144 | — |
| `plugplay` | 0.5114 | −0.0030 (p = 0.73) |
| `bwe_bypass_full` | 0.5016 | −0.0128 (p = 0.25) |
| `bwe` | 0.4783 | −0.0361 |

**Plug-and-play recovers 70% of the naturalness gain at essentially zero speaker-similarity
cost out of domain** (−0.003, well inside the widened null), where the fine-tuned
checkpoint gives up −0.036. On LibriTTS, the corpus the fine-tuning saw, the SIM costs are
comparable (`plugplay` −0.0205, `bwe` −0.0235), which fits: the fine-tuning damage to
identity is largely a generalisation failure and shows up on unseen speakers.

### The failure tail, which the means were hiding

Every contrast above is a paired mean, and a mean improves for two very different
reasons: a system can lift its typical output, or it can remove a handful of bad ones.
For a product the second matters more — a listener notices the one clip that sounds
wrong, not the average. The per-utterance DNSMOS scores were already written to disk by
the scoring pass, so the lower tail can be examined at no extra cost. The rejection bar
below is the 10th percentile of the `stock` arm at the same rate and corpus, which adapts
to the corpus instead of fixing an arbitrary MOS value; by construction `stock` always has
8 of 76 utterances under it.

P.808 lower tail at an 8 kHz prompt:

| corpus | arm | mean | p10 | p25 | min | n below stock's p10 |
|---|---|---|---|---|---|---|
| LibriTTS | `stock` | 3.539 | 3.301 | 3.440 | 3.089 | 8 |
| LibriTTS | `bwe_bypass_full` | 3.648 | 3.349 | 3.502 | 3.080 | 4 |
| LibriTTS | `plugplay` | 3.860 | 3.568 | 3.717 | 3.225 | 1 |
| LibriTTS | `bwe` | **3.949** | **3.670** | **3.806** | **3.391** | **0** |
| VCTK | `stock` | 3.488 | 3.251 | 3.365 | 3.050 | 8 |
| VCTK | `bwe_bypass_full` | 3.565 | 3.234 | 3.376 | 3.042 | 10 |
| VCTK | `plugplay` | 3.726 | 3.417 | 3.594 | 3.106 | 1 |
| VCTK | `bwe` | **3.826** | **3.523** | **3.719** | **3.376** | **0** |

**The fine-tuned checkpoint has no failures at all at 8 kHz, on either corpus.** Its worst
single utterance scores 3.391 on LibriTTS and 3.376 on VCTK, above the bar that eight
`stock` utterances fall below, and its 10th percentile (3.670 / 3.523) is above `stock`'s
*mean* (3.539 / 3.488). Plug-and-play removes seven of the eight failures; fine-tuning
removes all eight and moves the minimum a further 0.166 (LibriTTS) and 0.270 (VCTK) above
plug-and-play's.

The same effect shows up in a paired test restricted to the utterances `stock` handles
worst. Selecting the lower quartile of the baseline uses only baseline scores, so it does
not bias the sign of the difference:

| corpus | contrast at 8 kHz | Δ over all 76 | Δ over stock's worst 25% | p |
|---|---|---|---|---|
| LibriTTS | `plugplay` − `stock` | +0.322 | +0.406 | <0.0001 |
| LibriTTS | `bwe` − `stock` | +0.410 | **+0.553** | <0.0001 |
| VCTK | `plugplay` − `stock` | +0.238 | +0.358 | <0.0001 |
| VCTK | `bwe` − `stock` | +0.338 | **+0.456** | <0.0001 |

Both arms help the hard cases more than the easy ones, which is what a bandwidth extender
should do. But the gap between them widens in the tail: fine-tuning's advantage over
plug-and-play is +0.088 / +0.100 on the mean and **+0.147 / +0.098 on the worst quartile**.
On LibriTTS the tail advantage is 1.7 times the mean advantage.

This is the strongest case for the joint fine-tuning in the whole study, and it was
invisible in the mean-based reporting. It also has to be stated with its counterpart: at
16 kHz and above the fine-tuned arm has *more* utterances under the bar than `stock` (12,
13 and 12 on LibriTTS against 8), so the fine-tuning trades tail robustness at the rate it
was trained to repair against a slightly heavier tail where there was nothing to repair.
The 24 kHz control moves too (11 for `plugplay`, 15 for `bwe_bypass_full`), so part of that
spread is the same sampling noise the null conditions established, and none of the
high-rate tail differences reach significance.

### Where the extender itself came from

Study 1 concluded that plug-and-play needs no training. That is true of the *deployment*,
but not of the artefact: the `plugplay` checkpoint takes its 106 `bwe.*` tensors from
`iter-15000-avg-2.pt`, the end of the **joint adversarial stage**. Somebody has to run the
joint stage before anyone can plug anything in.

The arm that separates these is a second merge that takes the extender from the end of the
frozen stage, `zipvoice_bwe_frozen/iter-30000-avg-2.pt`, where ZipVoice was held at
`requires_grad_(False)` throughout. Measuring the merge confirms the freeze held: across
the 889 TTS tensors, the frozen-stage checkpoint differs from the released weights by at
most **1.7e-05 in absolute value and 2.4e-06 relative**, which is float rounding introduced
by the two-checkpoint averaging rather than learning, while the joint-stage checkpoint
differs by up to **0.47 absolute and 9.8% relative**. The frozen-stage checkpoint was
therefore already a plug-and-play configuration; the merge only makes the TTS tensors
bit-exact.

The contrast then reads: `plugplay_frozen` versus `plugplay`, same released weights, same
protocol, extender from the frozen stage against extender from the joint adversarial stage.
If they match, the extender can be trained in isolation and the adversarial stage is
dispensable. If `plugplay_frozen` falls short, the joint adversarial stage is what produced
the extender that the recommended deployment ships — and study 1's conclusion becomes an
argument for the training recipe rather than against it.

**It falls short, and only where the extender does work.** P.808 paired against `plugplay`
over the same 76 utterances:

| prompt rate | LibriTTS Δ (p) | VCTK Δ (p) |
|---|---|---|
| 8000 | **−0.0786 (0.000016)** | **−0.0508 (0.014)** |
| 16000 | −0.0033 (0.67) | +0.0008 (0.94) |
| 22050 | +0.0006 (0.90) | +0.0054 (0.49) |
| 24000 (control) | 0.0000 (exact) | 0.0000 (exact) |

The 24 kHz row is not merely inside the null band — it is exactly zero. The two arms carry
byte-identical TTS tensors, and at a 12000 Hz cutoff the mask is all-observed, so the Eq. 10
composite copies the prompt mel through and the extender weights cannot enter the
computation. The synthesised waveforms agree to the last bit (maximum absolute sample
difference 0.0 across all 76 files, identical ASR transcripts, identical 2.39% corpus WER).
That makes this the tightest control in the study: the 8 kHz effect cannot be decoding
noise, weight drift or a sampling artefact, because the only thing that differs between the
two arms is 106 extender tensors, and those tensors are provably inert at 24 kHz.

The effect also scales with how much band the extender has to synthesise — large at 8 kHz
where 8000 Hz of the 12000 Hz mel is missing, indistinguishable from zero at 16 kHz and
22.05 kHz where 4000 Hz and 975 Hz are. UTMOS agrees in sign but is too coarse to resolve
it (4.12 against 4.14 on LibriTTS, 4.13 against 4.15 on VCTK).

Placed in the tail table, the four arms order monotonically at an 8 kHz prompt — on the
mean, on p10, on the minimum and on the count below the bar, on both corpora:

| corpus | arm | mean | p10 | min | n below stock's p10 |
|---|---|---|---|---|---|
| LibriTTS | `stock` | 3.539 | 3.301 | 3.089 | 8 |
| LibriTTS | `plugplay_frozen` | 3.782 | 3.510 | 3.160 | 2 |
| LibriTTS | `plugplay` | 3.860 | 3.568 | 3.225 | 1 |
| LibriTTS | `bwe` | **3.949** | **3.670** | **3.391** | **0** |
| VCTK | `stock` | 3.488 | 3.251 | 3.050 | 8 |
| VCTK | `plugplay_frozen` | 3.675 | 3.372 | 3.132 | 2 |
| VCTK | `plugplay` | 3.726 | 3.417 | 3.106 | 1 |
| VCTK | `bwe` | **3.826** | **3.523** | **3.376** | **0** |

On `stock`'s worst quartile the same ordering holds: `plugplay_frozen` repairs +0.375 on
LibriTTS and +0.307 on VCTK, against `plugplay`'s +0.406 / +0.358 and `bwe`'s +0.553 /
+0.456, all at p < 0.0001.

**So the joint adversarial stage is doing two separable things, and both of them matter.**
It co-adapts the TTS weights to the extender, which is what the `bwe` − `plugplay` contrast
measures, and it improves the extender itself, which is what this contrast measures. The
second effect survives transplant onto frozen released weights, so it is a property of the
106 extender tensors rather than of any co-adaptation. Its size, 0.079 P.808 on LibriTTS,
is comparable to the fine-tuning's own mean advantage over `plugplay` at the same rate
(+0.088). Two thirds of the frozen stage's remaining gap to `bwe` is therefore recoverable
without touching ZipVoice at all — by running the adversarial stage and keeping only the
extender.

Speaker similarity moves the other way, and worth recording even though it is not
significant. `plugplay_frozen` − `plugplay` on SIM-o at 8 kHz is **+0.0114 (p = 0.089)** on
LibriTTS and **+0.0094 (p = 0.15)** on VCTK — the frozen-stage extender is slightly *better*
at preserving identity. Neither cell reaches significance and both are inside the widened
null, so this is directional only; but the sign agrees across two corpora and it fits the
pattern running through the whole study, that adversarial training trades speaker identity
for naturalness. What is new here is that it does so inside the extender's own 106 tensors,
not only through co-adaptation of the TTS. The 24 kHz control is again exactly 0.0000.

This reframes study 1's conclusion rather than contradicting it. Plug-and-play still needs
no training *to deploy*, and it is still the recommended configuration for mildly degraded
prompts. But the extender it deploys is a product of the joint adversarial stage, and a
frozen-stage extender is measurably worse at the rate that motivated the work.

### The 4 kHz condition: where plug-and-play stops being enough

Every rate above tests the extender on a task it can partly avoid: at a 16 kHz prompt only
4000 Hz of the 12000 Hz mel is missing, at 22.05 kHz only 975 Hz. The 4 kHz condition is
the opposite extreme — **10000 Hz of 12000, or 83% of the band, has to be synthesised** —
and it is the sharpest available test of whether the study 1 recommendation generalises.

It does not. Naturalness at a 4 kHz prompt:

| corpus | metric | `stock` | `bypass_full` | `plugplay` | `bwe` |
|---|---|---|---|---|---|
| LibriTTS | UTMOS | 2.80 | 3.44 | 3.87 | **4.27** |
| LibriTTS | P.808 | 3.210 | 3.331 | 3.600 | **3.940** |
| VCTK | UTMOS | 2.96 | 3.62 | 3.91 | **4.27** |
| VCTK | P.808 | 3.094 | 3.386 | 3.577 | **3.795** |

Two things stand out. **The fine-tuned checkpoint is essentially rate-invariant**: at 4 kHz
it scores UTMOS 4.27 and P.808 3.940, against 4.22 and 3.949 at 8 kHz. Halving the prompt
bandwidth a second time costs it nothing measurable, while `stock` falls from UTMOS 3.89 to
2.80. And **the fine-tuning's advantage over plug-and-play grows roughly fourfold** as the
task hardens:

| contrast, P.808 | LibriTTS Δ (p) | VCTK Δ (p) |
|---|---|---|
| `bwe` − `plugplay` at 4000 | **+0.340 (1e−22)** | **+0.218 (3e−11)** |
| `bwe` − `plugplay` at 8000 | +0.088 (0.0007) | +0.100 (0.0001) |
| `plugplay` − `stock` at 4000 | +0.390 (3e−24) | +0.483 (6e−29) |
| `bwe` − `bypass_full` at 4000 | +0.609 (1e−30) | +0.409 (3e−16) |
| `bypass_full` − `stock` at 4000 | +0.121 (3e−05) | +0.292 (1e−11) |

Plug-and-play's share of the total gain falls from 78% at 8 kHz to **53%** at 4 kHz on
LibriTTS. It does not break — it still clears the failure tail completely, 0 of 76
utterances below `stock`'s bar on both corpora against `stock`'s 8 and `bwe_bypass_full`'s
8 and 4 — it simply stops being competitive.

**And the identity cost that was the argument against fine-tuning does not exist at this
rate.** SIM-o at 4 kHz has collapsed to a floor for every arm: `stock` scores 0.3678 on
LibriTTS and 0.3224 on VCTK, against 0.6098 and 0.5144 at 8 kHz. With the prompt reduced to
a 2 kHz band there is not enough speaker information left for any system to preserve, and
no contrast between the four arms is significant — the largest is `bwe_bypass_full` −
`stock` on VCTK at −0.0150 (p = 0.18), and `bwe` − `plugplay` is −0.0026 (p = 0.81) and
−0.0066 (p = 0.59). Fine-tuning therefore buys +0.34 / +0.22 P.808 and +0.40 / +0.36 UTMOS
here and pays nothing measurable for it.

WER carries no signal at 4 kHz either, as at every other rate. Corpus WER ranges 2.79–3.19
on LibriTTS and 0.66–1.71 on VCTK across the four arms, against the 0.6–0.8 pp noise floor
the null conditions established.

The conclusion is that study 1's recommendation was scoped to the rates it measured. **The
crossover sits between 4 kHz and 8 kHz**: below it, the fine-tuned checkpoint is better on
naturalness by a wide margin and costs nothing in identity because identity is already
gone; above it, plug-and-play gets most of the naturalness for free and keeps the speaker.

### What study 1 settles

For deployment on **mildly degraded prompts — 8 kHz and above** — **plug-and-play is the
better configuration**. At 4 kHz it is not; see *The 4 kHz condition* above.
It gets about three quarters of the mean quality benefit, costs no measurable speaker
similarity on unseen speakers, requires no training, leaves the released checkpoint
untouched, and degrades gracefully if the extender is disabled. The 15000-step joint
fine-tuning buys a further ~0.09 P.808 at 8 kHz on the mean and pays for it with
out-of-domain speaker identity.

The tail analysis qualifies that recommendation rather than overturning it. Fine-tuning's
advantage is concentrated exactly where a listener would notice: it eliminates the failure
tail at 8 kHz outright on both corpora, where plug-and-play leaves one utterance below the
bar, and its advantage on the baseline's worst quartile is up to 1.7 times its advantage on
the mean. So the two configurations answer different questions. If the deployment must
never produce an unacceptable clip from a telephone-band prompt, the fine-tuned checkpoint
is the one that delivers that, and the identity cost is the price. If the prompts are
mostly clean and speaker identity is the priority, plug-and-play is better.

The identity cost itself is worth attributing correctly. At an 8 kHz VCTK prompt the
fine-tuned arm gives up 0.0361 SIM-o against `stock`, but `bwe_bypass_full` — fine-tuned
weights with the extender switched off — already gives up 0.0128. Roughly a third of the
damage belongs to the fine-tuning on its own and two thirds to the extender operating on
co-adapted weights. Reporting the 0.0361 as a single extender cost overstates it.

That also sharpens the diagnosis from the dead-discriminator analysis. The joint gradient is
pushing the extender toward output that makes the *fine-tuned* TTS produce natural speech,
and the TTS is co-adapting to it; neither side is being pushed toward speaker-specific high
band, because no working discriminator supplies that pressure. Freezing the TTS removes the
co-adaptation and, with it, most of the identity cost — at the price of the tail robustness
documented above.

### WER, tested rather than eyeballed

Earlier sections reported WER as condition means and dismissed it against a noise floor
estimated from one control. `seedtts.py` already writes a per-utterance WER and error count
to `decode.txt`, so WER can be given the same paired treatment as SIM-o and the MOS
predictors without rerunning the ASR pass. Two aggregations are reported because they
answer different questions: **Seed-TTS WER**, the unweighted mean of per-utterance rates
that the official Seed-TTS evaluation uses, and **corpus WER**, total errors over total
reference words, which is what one long transcript would give. Per-utterance WER is heavily
zero-inflated, so a Wilcoxon signed-rank test is reported next to the paired *t*.

Corpus WER, all four arms:

| | | LibriTTS | | | | VCTK | | |
|---|---|---|---|---|---|---|---|---|
| prompt rate | `stock` | `bypass_full` | `plugplay` | `bwe` | `stock` | `bypass_full` | `plugplay` | `bwe` |
| 8000 | 3.02 | 3.02 | 2.62 | 2.62 | 0.79 | 1.18 | 1.58 | 0.92 |
| 16000 | 2.74 | 2.96 | 2.45 | 3.02 | 0.79 | 1.31 | 1.05 | 0.92 |
| 22050 | 2.96 | 3.02 | 2.45 | 2.96 | 0.92 | 0.92 | 0.92 | 0.79 |
| 24000 (control) | 2.91 | 3.08 | 2.39 | 2.51 | 1.18 | 1.18 | 0.79 | 0.79 |

**The null condition sets the bar.** `plugplay` against `stock` at 24 kHz is the same
function under a different noise draw, and it moves per-utterance WER by −0.56 pp on
LibriTTS and −0.20 pp on VCTK (p = 0.30 and 0.79). So roughly **0.6 pp of per-utterance WER
is pure sampling noise at n = 76**, which independently confirms the 0.8 pp floor estimated
earlier from the other control.

Paired contrasts, per-utterance WER in percentage points:

| prompt rate | contrast | LibriTTS Δ (p<sub>t</sub> / p<sub>W</sub>) | VCTK Δ (p<sub>t</sub> / p<sub>W</sub>) |
|---|---|---|---|
| 8000 | plug-and-play vs stock | −0.16 (0.78 / 0.64) | +0.57 (0.46 / 0.40) |
| 16000 | plug-and-play vs stock | −0.21 (0.77 / 0.96) | +0.32 (0.58 / 0.57) |
| 22050 | plug-and-play vs stock | −0.69 (0.28 / 0.37) | −0.01 (0.99 / 0.92) |
| 24000 | plug-and-play vs stock | −0.56 (0.30 / 0.40) | −0.20 (0.79 / 0.58) |
| 8000 | extender on fine-tuned weights | −1.36 (0.059 / **0.034**) | −0.31 (0.56 / 0.40) |
| 16000 | fine-tuning on top of plug-and-play | +0.69 (**0.017** / **0.017**) | −0.10 (0.87 / 0.93) |

**WER still carries no signal, and now that is a measured result rather than an
impression.** Two cells reach nominal significance out of 24 tests, where about 1.2 false
positives are expected at α = 0.05; neither replicates on the other corpus, and both are
smaller than or comparable to the null condition's own excursion. Intelligibility is
therefore unaffected by the extender, by the fine-tuning, and by prompt bandwidth down to
8 kHz — which is the expected result, since ZipVoice takes its text from the transcript and
the prompt only conditions the voice.

One directional observation worth recording without claiming it: `plugplay` has the lowest
corpus WER of any arm in all four LibriTTS rows (2.39–2.62 against stock's 2.74–3.02). The
sign is consistent across four conditions, but no individual test is significant and the
24 kHz null moves by −0.56 pp on its own, so this is reported as directional only.

## Application studies 2 and 3: the training-side evidence

Study 2 asks whether the designed BWE loss earns its place, and study 3 whether the
easy-to-balanced cutoff curriculum does. Both are complete on the training side and
neither has been evaluated yet, so nothing here is a conclusion about output quality.
What follows is the measured training behaviour, recorded now because it is already
informative and because two of these numbers constrain what the evaluation can find.

**Both arms differ from the shipped `bwe` run in exactly one thing**, so `bwe` is the
only baseline either needs.

Study 3 is the full recipe with `--bwe-curriculum-rho 0.0`, which makes `sample_cutoff`
draw uniformly over 1–12 kHz from step 0 instead of starting biased toward high, easy
cutoffs. All three stages ran: 8000 pretrain, 30000 frozen, 15000 joint.

Study 2 was rescoped, and the narrower claim has to be stated wherever its numbers
appear. The original design retrained the extender from scratch with every mel loss
weight at zero — a 30000-step frozen stage plus a 15000-step joint stage — and in fp16
it died at step 5400 with `grad_scale is too small, exiting: 3.8e-06`, having decayed
the loss scale continuously from 65536. The arm now runs only the joint stage, starting
from the shipped run's own `zipvoice_bwe_frozen/iter-30000-avg-2.pt`, which is
byte-identical initialisation to `bwe`, and changes only the loss weights for those
15000 steps: reconstruction, adversarial and feature matching all zero, leaving
ZipVoice's flow-matching gradient as the extender's sole supervision. It therefore
cannot say whether that gradient alone could have trained an extender from nothing. It
says whether the designed loss earns its place *in the joint stage*, against an arm
identical in every other respect — which is the better-controlled of the two contrasts,
since the original changed both stages at once, and it costs 15000 steps instead of
45000.

It also runs in bfloat16 rather than the float32 the crash first suggested. bfloat16
carries float32's exponent range, so it needs no loss scaling and cannot reach the
underflow that killed the float16 attempt, while keeping tensor-core throughput; the
GPU here is compute capability 12.0, well past the Ampere minimum. This required a
small addition to `train_zipvoice_bwe.py`: `--amp-dtype {fp16,bf16}`, with the autocast
dtype and the GradScaler resolved once into `params.amp_torch_dtype` and
`params.use_grad_scaler` so no call site re-derives the decision. The scaler is enabled
only for float16. Precision is therefore one more way this arm differs from the shipped
recipe, alongside the loss weights.

### Study 3: the curriculum is not preventing a collapse

Validation losses at the end of the frozen stage, where the curriculum is active and
where AnyBand applies it:

| | fm_loss | bwe_recon | total |
|---|---|---|---|
| shipped, curriculum on | 0.03746 | 0.1028 | 0.1403 |
| study 3, curriculum off | 0.03747 | **0.09618** | 0.1336 |

Turning the curriculum off produced **better** mel reconstruction, by 6.4%, at a
flow-matching loss identical to four decimal places. Nothing collapsed, diverged or
needed rescuing: the run held `grad_scale` at 32–128 for all 30000 steps, the same band
the shipped run sits in.

That is worth stating plainly, because the curriculum's stated purpose is to keep early
training tractable by starting on easy, high cutoffs. On this recipe and this corpus it
is not doing that — uniform sampling from step 0 is stable on its own, and the
curriculum's early bias toward easy cutoffs appears to cost reconstruction accuracy
rather than buy stability.

The joint stage reverses the ordering, compared at the last step both runs validated on:

| at step 14000 | fm_loss | bwe_recon | d_spec | d_cross | d_harm |
|---|---|---|---|---|---|
| shipped | 0.03538 | 0.1154 | — | — | — |
| study 3 | 0.03541 | 0.1285 | 0.667 | 0.988 | 0.959 |

This is not straightforwardly "worse". The joint stage is adversarial, and `bwe_recon`
there is traded against realism rather than minimised — the shipped run's own
reconstruction rises from 0.1028 to 0.1181 across the joint stage while its output
improves on every perceptual metric. Which direction the trade went can only be settled
by the evaluation.

The discriminator values are a third independent confirmation of the dead-discriminator
finding: `d_cross` at 0.988 and `d_harm` at 0.959 against a trivial hinge baseline of
1.0 per view, with only `d_spec` at 0.667 doing work. Two of the three discriminators
have now failed to learn anything in three separate training runs.

### Study 2: the extender does not fall apart without supervision

Validation trajectory with every direct loss on the extender's output removed. The
reconstruction term is still computed and logged at weight zero, contributing exactly no
gradient, which makes it a free diagnostic: it reports what mel accuracy the
flow-matching gradient drifts to having never once been asked to preserve it.

| step | fm_loss | bwe_recon |
|---|---|---|
| 0 | 0.03758 | 0.09643 |
| 2000 | 0.03576 | 0.1030 |
| 4000 | 0.03541 | 0.1059 |
| 6000 | 0.03538 | 0.1061 |
| 8000 | 0.03538 | 0.1069 |

Reconstruction degrades by 11% and then plateaus by step 4000; the flow-matching loss
*improves*, from 0.03758 to 0.03538. The extender is not collapsing — it is trading a
little mel accuracy for a representation the TTS finds more useful, which is exactly
what the joint gradient is supposed to do and what the `resunet` bypass arm showed from
its extreme end.

The number that complicates the story: **0.1069 is better mel accuracy than the shipped
run's own 0.1181** at the end of its joint stage. Removing every direct loss on the
extender's output left it closer to the true mel than keeping reconstruction at weight
1.0 alongside the adversarial terms did. The adversarial pressure, not the absence of
supervision, is what moves reconstruction furthest from the target. Whether that is a
cost or the entire point is what the perceptual metrics have to answer.

### One quantity none of this moves

Across every run in this table — curriculum on or off, mel losses present or absent,
float16 or bfloat16 — the final validation `fm_loss` lands between 0.03527 and 0.03541,
a spread of 0.00014. ZipVoice's own objective is insensitive to all of these choices.
Whatever separates these arms will not be visible in the training loss, which is why the
evaluation on synthesised audio is the only thing that can decide any of it.

### What the evaluation found: both components are dispensable

The synthesis and scoring pass ran on 2026-09-01, the same protocol as every other arm:
76 utterances per condition, four prompt rates, two corpora, `iter-15000-avg-2.pt` in both
cases, 16 sampling steps, `--bwe-bypass none`.

UTMOS, with the shipped run as the reference:

| corpus | prompt rate | `bwe` | `ablfm` (study 2) | `ablnocur` (study 3) |
|---|---|---|---|---|
| LibriTTS | 8000 | 4.22 | 4.24 | 4.25 |
| LibriTTS | 16000 | 4.19 | 4.19 | 4.18 |
| LibriTTS | 22050 | 4.17 | 4.19 | 4.16 |
| LibriTTS | 24000 | 4.16 | 4.18 | 4.16 |
| VCTK | 8000 | 4.25 | 4.21 | 4.23 |
| VCTK | 16000 | 4.18 | 4.20 | 4.19 |
| VCTK | 22050 | 4.14 | 4.16 | 4.21 |
| VCTK | 24000 | 4.18 | 4.16 | 4.19 |

DNSMOS P.808 means, which have a measured null of 0.011 (LibriTTS) and 0.018–0.037 (VCTK):

| corpus | prompt rate | `bwe` | `ablfm` | `ablnocur` |
|---|---|---|---|---|
| LibriTTS | 8000 | 3.949 | 3.917 | 3.935 |
| LibriTTS | 16000 | 3.937 | 3.936 | 3.931 |
| LibriTTS | 22050 | 3.945 | 3.933 | 3.940 |
| LibriTTS | 24000 | 3.949 | 3.955 | 3.950 |
| VCTK | 8000 | 3.826 | 3.819 | 3.850 |
| VCTK | 16000 | 3.825 | 3.815 | 3.829 |
| VCTK | 22050 | 3.827 | 3.817 | 3.815 |
| VCTK | 24000 | 3.850 | 3.831 | 3.837 |

**Nothing here separates the three arms.** The largest naturalness difference anywhere in
the two tables is 0.032 P.808 and 0.04 UTMOS, both inside the null conditions established
in Application study 1. The tail analysis agrees: no worst-quartile contrast against `bwe`
reaches an effect larger than 0.079 P.808, and none replicates across corpora. WER is
likewise unmoved — at an 8 kHz LibriTTS prompt the paired per-utterance deltas against
`bwe` are +0.13 pp for `ablfm` (p = 0.54) and +0.18 pp for `ablnocur` (p = 0.25), well
under the 0.6 pp sampling noise the null conditions measured.

SIM-o is the one metric where anything moves, and it moves in the ablations' favour:

| corpus | prompt rate | `bwe` | `ablfm` Δ (p) | `ablnocur` Δ (p) |
|---|---|---|---|---|
| LibriTTS | 8000 | 0.5863 | +0.0057 (0.24) | +0.0011 (0.79) |
| LibriTTS | 16000 | 0.7221 | +0.0023 (0.51) | +0.0050 (0.15) |
| LibriTTS | 22050 | 0.7322 | +0.0012 (0.72) | −0.0014 (0.66) |
| LibriTTS | 24000 | 0.7294 | +0.0047 (0.15) | +0.0011 (0.73) |
| VCTK | 8000 | 0.4783 | **+0.0156 (0.029)** | **+0.0122 (0.043)** |
| VCTK | 16000 | 0.6147 | +0.0080 (0.091) | +0.0076 (0.10) |
| VCTK | 22050 | 0.6251 | **+0.0097 (0.018)** | **+0.0082 (0.047)** |
| VCTK | 24000 | 0.6220 | −0.0014 (0.75) | **+0.0115 (0.014)** |

Every one of the twelve VCTK cells is positive except one, and four reach p < 0.05. That
is suggestive — both ablations recover part of the identity cost documented in study 1 —
but it must be reported as directional, for two reasons.

**The 24 kHz condition is not a null across these arms.** In study 1 it was, because the
extender is provably identity at a Nyquist cutoff and `plugplay` carried bit-identical TTS
weights. Here the extender is still identity, but each arm ran its own 15000 joint steps
and therefore carries *different* TTS weights. `ablnocur` moves +0.0115 at 24 kHz with
p = 0.014, as large as its 8 kHz effect. Whatever is happening is not specific to
bandwidth extension; it is a property of the fine-tuning these runs produced.

**It does not replicate on LibriTTS.** All eight LibriTTS cells are inside the noise. With
24 tests at α = 0.05, about 1.2 false positives are expected, and the document's own
standing rule — set by a known-zero effect reaching p = 0.046 — is that no single test at
this sample size carries a conclusion. Cross-corpus replication is what carried the 8 kHz
naturalness results, and it is absent here.

### What studies 2 and 3 settle

**Neither the cutoff curriculum nor any direct loss on the extender output earns its place
in this recipe.** Study 3 removed the AnyBand curriculum entirely and study 2 removed the
reconstruction loss, all three discriminators, the adversarial terms and the
feature-matching terms, leaving ZipVoice's flow-matching gradient as the extender's only
supervision. Neither change is detectable in naturalness, intelligibility or the failure
tail on either corpus, and both are marginally *better* on speaker similarity out of
domain.

That is a negative result about our own design choices, and it is the most actionable
finding in the document. Combined with the dead-discriminator analysis — `d_cross` at
0.98 and `d_harm` at 0.96 against a trivial baseline of 1.0, across three independent runs
— the NVSR + AnyBand stack reduces on this task to two components that do work: the joint
flow-matching gradient, which is what makes the extender produce a mel the TTS can use,
and `d_spec`, which is the only discriminator that learned anything. The curriculum, the
reconstruction loss, `D_cross` and `D_harm` can all be removed. Study 2's configuration
trains at 2930 steps/h against the full recipe's 1730, so dropping them is also a 1.7x
speedup.

The training-side signal predicted this. Final validation `fm_loss` across every run in
the study spans 0.03527 to 0.03541 — a spread of 0.00014 — and the evaluation confirms
that the arms are as indistinguishable in synthesised audio as they were in the loss.

One caveat bounds the claim. Study 2 was rescoped to the joint stage only: it starts from
the shipped frozen-stage checkpoint, which was itself trained with `bwe_recon` active for
30000 steps. So the result is that direct supervision is dispensable **in the joint stage**,
given an extender that has already been pretrained with it. It does not show that the
extender could be trained from scratch on the flow-matching gradient alone.

### Still outstanding

Nothing from the planned matrix is unscored. Every arm has been evaluated at 4, 8, 16,
22.05 and 24 kHz on both corpora, on UTMOS, DNSMOS P.808 and P.835, Seed-TTS WER and
SIM-o, with per-utterance scores retained for the tail analysis.

What remains open is not measurement but experiment. Three questions the evaluation raised
and cannot answer from the data on disk: whether the extender can be trained without
`bwe_recon` from the start, which needs the frozen stage retrained (next step 2); whether
extender dropout during joint training would keep the tail robustness while removing the
identity cost, which would let one checkpoint serve both regimes (next step 1); and what
the extender actually predicts above 8 kHz, which no metric used here can see because
DNSMOS and UTMOS both run at 16 kHz (next step 6).

## The reconstruction-only baseline

Every arm compared so far shares the same supervision. The obvious external comparison is
an audio-domain super-resolution front end: upsample the prompt waveform with an
off-the-shelf model, hand the result to stock ZipVoice, and see whether the mel-domain
extender was worth building. That arm could not be run. `ssr_eval` fails to build because
it declares a Python-2-era dependency, and `audiosr` fails to build NumPy from source under
the only interpreter available here (3.13); neither has a wheel for this platform. Rather
than report nothing, the comparison was made along the axis the paper actually claims.

The `plugplay_pretrain` arm — called *recon-only* below — is the stage-0 extender merged
onto the released ZipVoice weights. It is the same ResUNet, at the same insertion point, in
front of the same TTS parameters as the `plugplay` arm. The only difference is how it was
trained: L1 reconstruction of the masked band, which is NVSR's objective, with no
flow-matching gradient and no discriminators. It is therefore an NVSR-style extender in
everything except the domain it operates in, and it isolates the one variable the paper
argues about. `local/make_plugplay_checkpoint.py` built it from
`zipvoice_bwe_pretrain/iter-8000-avg-2.pt`, taking the 106 extender tensors and leaving the
889 TTS tensors at their released values.

All five prompt rates were scored on both MOS predictors; the full ladder is in the
next subsection. The two rates where the extender does substantial work carry the
result.

The reading is consistent across both corpora and both MOS predictors. Reconstruction-only
supervision is not useless: it beats stock at every rate on P.808, significantly, so a
mel-domain extender trained the NVSR way does help. But it recovers a small fraction of
what the same architecture recovers when the flow-matching loss supervises it. At 4 kHz on
LibriTTS it gains 0.086 P.808 over stock where the flow-matching-supervised extender gains
0.390, and on VCTK 0.123 against 0.483; UTMOS separates them by a wider margin still (3.13
against 3.87 on LibriTTS, 3.22 against 3.91 on VCTK). At 8 kHz the gap narrows to 0.115 on
LibriTTS and vanishes on VCTK's P.808, though UTMOS still separates the two by 0.26 there.
So the advantage of task-aware supervision grows with the amount of bandwidth that has to
be invented, which is what the mechanism predicts: with 8 kHz of the 12 kHz band missing,
reconstruction has almost no constraint on what to put there, and only the downstream TTS
loss says which of the many plausible high bands the model can actually use.

Two secondary observations. First, DNSMOS OVRL disagrees with P.808 on this arm at 4 kHz —
it puts recon-only *below* stock on LibriTTS (-0.100, p = 2.8e-06) while P.808 puts it
above. OVRL has been the less discriminating of the two throughout this document, and at
4 kHz all arms sit within 0.17 of each other on it; the P.808 and UTMOS agreement is the
stronger evidence. Second, recon-only costs no speaker similarity anywhere — 0.379 against
plug-and-play's 0.377 at LibriTTS 4 kHz, 0.514 against 0.511 at VCTK 8 kHz — which is
consistent with the finding elsewhere in this document that identity is not what any of
these extenders carry.

The caveat this arm does not remove: it is still a mel-domain extender inside the TTS
model, so it does not answer what a waveform-domain super-resolution front end would do. It
answers the narrower and more directly relevant question of whether the supervision is
where the gain comes from, holding architecture, insertion point and TTS weights fixed. The
audio-domain arm remains open work.

### The full rate ladder

The whole point of the method is that the gain should appear where bandwidth is missing and
nowhere else, so both MOS predictors were run over every arm at every prompt rate. UTMOS,
means over 76 utterances:

| corpus | prompt rate | recovered | stock | recon-only | plugplay | bwe |
|---|---|---|---|---|---|---|
| LibriTTS | 4000 | 10000 Hz | 2.80 | 3.13 | 3.87 | **4.27** |
| LibriTTS | 8000 | 8000 Hz | 3.89 | 3.95 | 4.14 | **4.22** |
| LibriTTS | 16000 | 4000 Hz | 4.08 | 4.13 | 4.14 | 4.19 |
| LibriTTS | 22050 | 975 Hz | 4.17 | 4.17 | 4.17 | 4.17 |
| LibriTTS | 24000 | 0 Hz | 4.20 | 4.15 | 4.15 | 4.16 |
| VCTK | 4000 | 10000 Hz | 2.96 | 3.22 | 3.91 | **4.27** |
| VCTK | 8000 | 8000 Hz | 3.92 | 3.89 | 4.15 | **4.25** |
| VCTK | 16000 | 4000 Hz | 4.08 | 4.16 | 4.17 | 4.18 |
| VCTK | 22050 | 975 Hz | 4.19 | 4.20 | 4.18 | 4.14 |
| VCTK | 24000 | 0 Hz | 4.17 | 4.23 | 4.23 | 4.18 |

DNSMOS P.808, same utterances:

| corpus | prompt rate | recovered | stock | recon-only | plugplay | bwe |
|---|---|---|---|---|---|---|
| LibriTTS | 4000 | 10000 Hz | 3.210 | 3.297 | 3.600 | **3.940** |
| LibriTTS | 8000 | 8000 Hz | 3.539 | 3.745 | 3.860 | **3.949** |
| LibriTTS | 16000 | 4000 Hz | 3.969 | 3.987 | 3.965 | 3.937 |
| LibriTTS | 22050 | 975 Hz | 3.969 | 3.963 | 3.974 | 3.945 |
| LibriTTS | 24000 | 0 Hz | 3.984 | 3.964 | 3.964 | 3.949 |
| VCTK | 4000 | 10000 Hz | 3.094 | 3.217 | 3.577 | **3.795** |
| VCTK | 8000 | 8000 Hz | 3.488 | 3.721 | 3.726 | **3.826** |
| VCTK | 16000 | 4000 Hz | 3.886 | 3.895 | 3.852 | 3.825 |
| VCTK | 22050 | 975 Hz | 3.897 | 3.881 | 3.881 | 3.827 |
| VCTK | 24000 | 0 Hz | 3.919 | 3.882 | 3.882 | 3.850 |

DNSMOS OVRL over the same grid stays inside 3.19 to 3.35 for every arm at every rate, a
total spread of 0.16 against P.808's 0.89, and it does not order the arms consistently. It
is reported in the P.835 section above and is not repeated here; the two predictors that
resolve anything are UTMOS and P.808, and they agree.

Three things are visible in these two tables, and they are the argument for the method.

**The gain is monotone in recovered bandwidth and vanishes when there is none.** Full
system minus stock on P.808 runs +0.730, +0.410, -0.032, -0.024, -0.035 down the LibriTTS
ladder and +0.701, +0.338, -0.061, -0.070, -0.069 down the VCTK ladder. UTMOS traces the
same curve: +1.47, +0.33, +0.11, 0.00, -0.04 and +1.31, +0.33, +0.10, -0.05, +0.01. The
extender is worth nothing above a 16 kHz prompt and worth a great deal below 8 kHz, which
is exactly the profile a bandwidth extender should have and is the strongest single piece
of evidence that the measured effect is the mechanism rather than an artefact of
fine-tuning.

**Above 16 kHz the arms converge to within the noise floor.** Every cell at 22050 and 24000
sits inside 0.05 UTMOS and 0.04 P.808 of stock, which are the noise floors established
earlier in this document. The small negative signs there are the previously documented cost
of 15000 steps of LibriTTS fine-tuning on undegraded prompts, not a failure of the
extender: `plugplay`, which carries released TTS weights, matches stock at 22050 on both
corpora.

**The 24 kHz control is exact, and now on a second pair of arms.** At a 24000 prompt the
`plugplay` and `recon-only` arms produce per-utterance scores that are identical on all 76
pairs in both corpora -- the paired *t* statistic is undefined because every difference is
exactly zero. The two arms differ in all 106 extender tensors and share their 889 TTS
tensors, so this reconfirms directly that the hard composite makes the extender inert at a
Nyquist cutoff: two different extenders, byte-identical output. The same control was
established earlier between `bwe` and `plugplay_frozen`; it now holds across an arm trained
with a completely different objective.

**Reconstruction-only supervision tracks the same curve at a quarter of the amplitude.**
The recon-only column rises and falls with the others but never gets far from stock:
+0.086 and +0.207 P.808 at 4000 and 8000 on LibriTTS against plug-and-play's +0.390 and
+0.321, and +0.123 and +0.233 on VCTK against +0.483 and +0.238. The gap between the two
supervisions is itself monotone in recovered bandwidth on LibriTTS -- 0.303 P.808 at
10000 Hz recovered, 0.115 at 8000 Hz, nothing above -- which is the finding stated in the
previous subsection, now visible across the whole ladder rather than at two points.

Paired per-utterance P.808 tests for recon-only against each reference arm, all rates
(n = 76 per cell):

| corpus | rate | vs stock | vs plugplay | vs bwe |
|---|---|---|---|---|
| LibriTTS | 4000 | +0.086 (p = 1.6e-04) | -0.303 (p = 1.7e-19) | -0.644 (p = 1.8e-39) |
| LibriTTS | 8000 | +0.207 (p = 3.9e-11) | -0.115 (p = 9.9e-06) | -0.203 (p = 5.3e-16) |
| LibriTTS | 16000 | +0.018 (p = 0.19) | +0.022 (p = 0.036) | +0.051 (p = 0.0040) |
| LibriTTS | 22050 | -0.005 (p = 0.69) | -0.011 (p = 0.12) | +0.018 (p = 0.34) |
| LibriTTS | 24000 | -0.020 (p = 0.18) | 0.000 (exact) | +0.015 (p = 0.44) |
| VCTK | 4000 | +0.123 (p = 1.0e-05) | -0.360 (p = 1.5e-21) | -0.578 (p = 5.0e-32) |
| VCTK | 8000 | +0.233 (p = 3.5e-14) | -0.005 (p = 0.84) | -0.104 (p = 5.5e-05) |
| VCTK | 16000 | +0.010 (p = 0.65) | +0.043 (p = 0.022) | +0.070 (p = 7.6e-04) |
| VCTK | 22050 | -0.017 (p = 0.47) | 0.000 (p = 0.99) | +0.054 (p = 0.0050) |
| VCTK | 24000 | -0.037 (p = 0.075) | 0.000 (exact) | +0.032 (p = 0.075) |

The recon-only-over-`bwe` cells at 16000 and above are the fine-tuning cost on undegraded
prompts appearing again from the other direction, not evidence that the weaker arm is
better; recon-only carries released TTS weights and `bwe` does not. Nothing between 4000
and 8000 Hz recovered is sampled, so where the threshold sits between a 16 kHz and an
8 kHz prompt is unmeasured.

Word error rates for the recon-only arm across the ladder are 3.65%, 3.19%, 3.14%, 2.68%
and 2.39% on LibriTTS and 1.97%, 0.66%, 2.5%, 1.18% and 0.79% on VCTK, and speaker
similarity is 0.379, 0.607, 0.696, 0.733 and 0.731 on LibriTTS and 0.352, 0.514, 0.573,
0.634 and 0.626 on VCTK. Both track the prompt rate and neither separates the arms beyond
the noise floors established earlier.

### A signal-level metric: mel-cepstral distortion

Every quality number above is a learned MOS predictor, and both of the ones used resample to
16 kHz internally. Neither can see the band the extender is supposed to restore, which is an
uncomfortable position from which to claim a bandwidth extender works. Mel-cepstral distortion
does not have that problem: it compares WORLD spectral envelopes directly, at 24 kHz, so it
observes the whole band and cannot be satisfied by output that merely sounds plausible.

The evaluation sets did not carry the natural recording of each target sentence, because none of
the three original metrics needed it. It was recoverable after the fact: both preparation scripts
name their rows `{speaker}_{target utterance id}`, so `local/prepare_mcd_references.py` re-reads
the corpora and writes 76 references per corpus at 24 kHz. `local/mcd_eval.py` then computes
order-34 mel-cepstra from the WORLD envelope with coefficient zero dropped, aligns them by
dynamic time warping, and scales by `10 / ln(10) * sqrt(2)`, which is the usual convention.

One caveat governs how these numbers may be read. The reference is a *different rendition* of the
same sentence by the same speaker, with its own prosody and phone durations, so a large part of
the distance is delivery the model was never asked to reproduce. Warping removes the duration
component but not the delivery one. That is why every arm sits near 7.8 dB at a 24 kHz prompt
rather than near zero, and it is why these figures must not be compared against MCD numbers from
a resynthesis task. They are valid for ranking arms given identical text, identical speakers and
identical references, which is exactly the comparison made here.

Means over 76 utterances, in dB, lower is better:

| corpus | prompt rate | recovered | stock | recon-only | plugplay | bwe |
|---|---|---|---|---|---|---|
| LibriTTS | 4000 | 10000 Hz | 19.302 | 15.762 | 10.195 | **9.938** |
| LibriTTS | 8000 | 8000 Hz | 15.450 | 11.888 | 9.529 | **9.039** |
| LibriTTS | 16000 | 4000 Hz | 10.952 | 10.341 | 8.574 | **8.020** |
| LibriTTS | 22050 | 975 Hz | 8.108 | 8.125 | 8.136 | **7.854** |
| LibriTTS | 24000 | 0 Hz | 7.896 | 7.881 | 7.881 | 7.768 |
| VCTK | 4000 | 10000 Hz | 18.100 | 15.605 | 10.025 | **9.460** |
| VCTK | 8000 | 8000 Hz | 14.140 | 11.258 | 8.981 | **8.702** |
| VCTK | 16000 | 4000 Hz | 11.176 | 10.168 | 8.945 | **8.184** |
| VCTK | 22050 | 975 Hz | 8.023 | 7.960 | 8.056 | **7.777** |
| VCTK | 24000 | 0 Hz | 7.790 | 7.702 | 7.702 | 7.672 |

Paired per-utterance tests against stock, n = 76 per cell, negative is an improvement:

| corpus | rate | recon-only | plugplay | bwe |
|---|---|---|---|---|
| LibriTTS | 4000 | -3.540 (p = 1.9e-26) | -9.107 (p = 5.4e-45) | -9.364 (p = 5.6e-44) |
| LibriTTS | 8000 | -3.562 (p = 6.9e-16) | -5.921 (p = 8.0e-32) | -6.411 (p = 1.5e-31) |
| LibriTTS | 16000 | -0.611 (p = 0.043) | -2.378 (p = 2.1e-27) | -2.931 (p = 7.9e-29) |
| LibriTTS | 22050 | +0.017 (p = 0.69) | +0.027 (p = 0.48) | -0.254 (p = 1.6e-05) |
| LibriTTS | 24000 | -0.016 (p = 0.69) | -0.016 (p = 0.69) | -0.129 (p = 0.0050) |
| VCTK | 4000 | -2.496 (p = 8.7e-20) | -8.075 (p = 1.8e-48) | -8.640 (p = 4.1e-48) |
| VCTK | 8000 | -2.882 (p = 3.8e-23) | -5.158 (p = 1.3e-40) | -5.438 (p = 1.4e-42) |
| VCTK | 16000 | -1.008 (p = 9.4e-08) | -2.231 (p = 3.8e-33) | -2.992 (p = 1.4e-36) |
| VCTK | 22050 | -0.063 (p = 0.19) | +0.033 (p = 0.51) | -0.247 (p = 9.2e-05) |
| VCTK | 24000 | -0.088 (p = 0.050) | -0.088 (p = 0.050) | -0.118 (p = 0.017) |

Four things follow.

**The signal-level metric agrees with the predictors, and separates the arms far more sharply.**
The full system beats stock by 9.4 dB at a 4 kHz prompt on LibriTTS and 8.6 dB on VCTK, and the
ordering stock < recon-only < plug-and-play < full system holds in every one of the ten rows
where anything is being recovered. Where UTMOS and P.808 compress into a fraction of a point,
MCD spreads the same comparison over more than 11 dB.

**The rate profile is the same one, only cleaner.** The gain shrinks monotonically as recovered
bandwidth shrinks -- 9.4, 6.4, 2.9, 0.25, 0.13 dB down the LibriTTS ladder for the full system --
and it is the same curve the MOS predictors traced. MCD sees it at 16 kHz prompts too, where the
predictors could not resolve anything: 2.9 dB on LibriTTS and 3.0 dB on VCTK, both at p below
1e-28. This is a band the predictors are structurally blind to, and the improvement there is real
and large.

**The full system now wins everywhere, including where the predictors said it lost.** At 22050
and 24000 the fine-tuned arm was slightly *behind* stock on P.808 and UTMOS, which was attributed
to the cost of 15000 steps of LibriTTS fine-tuning. On MCD it is ahead at every rate on both
corpora -- 0.254 and 0.129 dB on LibriTTS, 0.247 and 0.118 dB on VCTK, all significant. The two
metrics are not in conflict: the fine-tuned model is closer to the reference speaker's spectrum
and marginally less preferred by a naturalness predictor. Which of those matters more is a
question about the application, not about the method.

**The 24 kHz inertness control reappears at signal level.** At a 24000 prompt the `plugplay` and
recon-only arms score identically to three decimals on both corpora -- 7.881 and 7.702 -- with
paired differences of exactly zero, as they did on DNSMOS. Two extenders differing in all 106
tensors produce byte-identical audio when the cutoff is at Nyquist. The hard composite is doing
what it was designed to do.

## Cost of the extender

`local/cost_bench.py` measures the extender against a full synthesis on the same machine
(RTX 5070 Ti, fp32, 16 ODE steps, a 6.19-second prompt of 580 mel frames, 10 timed
iterations after warm-up):

| quantity | value |
|---|---|
| ZipVoice parameters | 122.66 M |
| extender parameters | 7.46 M (106 tensors) |
| parameter overhead | +6.1% |
| extender forward, GPU | 3.58 ms |
| extender forward, CPU | 66.71 ms |
| `sample()`, 16 steps, 6.19 s of output | 873.62 ms |
| extender share of one synthesis | 0.41% |
| synthesis real-time factor | 0.1412 |

The parameter overhead and the runtime share differ by an order of magnitude, and the
reason is structural rather than an artifact of this measurement. The extender runs once,
over the prompt; the flow-matching solver runs sixteen times, over the output. Prompt
length and output length are independent, so the 0.41% figure moves with their ratio, but
the direction does not: the extender would have to process a prompt roughly forty times
longer than the utterance being synthesised before it cost as much as the solver.

The second half of the argument is what is *absent* from the added path. The extender
consumes and produces the mel that ZipVoice already computes and already conditions on, so
there is no analysis-synthesis round trip, no second vocoder, and no waveform-rate work of
any kind. An audio-domain super-resolution front end cannot make that claim: it must
produce a waveform, which the feature extractor then immediately discards down to a
100-bin mel. The GPU and CPU timings above bracket the deployment cases, and the CPU number
matters more than the GPU one for the edge target: 66.71 ms once per prompt is below the
threshold where a user would notice it at all.

The absolute timings were re-measured on an idle GPU, and they hold: 3.77 ms for the extender,
879.02 ms for `sample()`, a share of 0.43% and an RTF of 0.1421, against the 3.58 ms / 873.62 ms
/ 0.41% / 0.1412 quoted above. A repeat run taken while a scoring job shared the device gave
4.28 ms and 1122.26 ms (share 0.38%, RTF 0.1814), which shows how far the absolute numbers move
with load. The ratio is the stable quantity and is what the paper should report.

### End-to-end real-time factor

The benchmark above times the extender in isolation. The other half of the question is whether
the cost is visible in a real synthesis run, so `infer_zipvoice.py`'s own aggregate RTF was read
off three full passes over the LibriTTS evaluation set at an 8 kHz prompt, 76 utterances each,
16 ODE steps, on an otherwise idle GPU. Three repeats per arm:

| arm | run 1 | run 2 | run 3 |
|---|---|---|---|
| stock ZipVoice | 0.1500 | 0.1499 | 0.1503 |
| plug-and-play extender | 0.1477 | -- | -- |
| full system | 0.1474 | 0.1473 | 0.1476 |

Run-to-run spread within an arm is 0.0004, so the measurement is tight, and the extender arms
come out *faster* than stock by about 0.0026, or 1.7%. That is not a speed-up from adding
computation. RTF here is total inference time divided by total generated audio duration, and the
arms do not generate the same durations: the fine-tuned model predicts slightly longer utterances
for the same text, which lowers the denominator. The comparison is therefore confounded and
should not be quoted as evidence that the extender is free.

What it does establish is an upper bound. Whatever the extender costs, it is smaller than a
confound of 1.7% in the total, which is consistent with the 0.43% share measured directly and
inconsistent with any claim that the added path is a material inference expense. Vocoder time is
0.0007 to 0.0008 in all three arms, so essentially the whole budget is the flow-matching solver,
exactly as the isolated benchmark says.

## The in-trajectory high-band branch: negative results

`ZipVoiceDual` keeps the stock backbone frozen and adds a branch that writes only the bins
above the prompt's cutoff, inside the same solve, so the low band is the backbone's own and
cannot drift. Every trained variant (endpoint and velocity parameterisation, teacher-forced
and rollout rows, a differentiable solve loss) produced the same failure: a high band at
roughly the right level with no formant structure and no voiced/unvoiced gating. This
section records why supervised training cannot fix that, so the next attempt starts from
the right premise. All numbers are sr8000, `dual_vfm` checkpoint-6000 where a checkpoint
matters, 16-step guided Euler (t_shift 0.5, guidance 1.0) as in the evaluation.

### The information is there; the branch does not use it

Ridge probes from each input stream to the reference high band (31 bins), fitted on 50
utterances and scored on 26 held out. `level` is the per-frame mean over the high bins,
`shape` the per-frame deviation from it; R² = 0 means "predict the average frame".

| stream | level R² | shape R² |
|---|---|---|
| text condition alone | −0.009 | −0.002 |
| reference low band, ±2 frames | 0.600 | 0.164 |
| prompt condition (`encode_condition`) | 0.134 | 0.211 |
| low band + prompt condition | 0.634 | 0.270 |
| everything the branch reads at t = 0.85 | 0.732 | 0.339 |
| branch endpoint at t = 0.85, fitted readout | 0.286 | 0.016 |
| branch endpoint at t = 0.85, raw | −0.032 | −0.936 |

The branch's inputs predict a third of the spectral shape linearly; its output predicts
none of it, and scored directly it is worse than the average frame. The low-band row is
linear over about 50 ms, while the "everything" row includes the backbone's 512-dim hidden
state over the whole utterance, so the gap between them mixes information with capacity
and is not yet a measure of what text and prompt add.

### Why the training rows cannot supply the pairing

Teacher-forced rows leak the answer: the noised target high band sits in the branch's own
state and in the backbone memory computed on it (memory on the full teacher state reaches
shape R² 0.85 at t = 0.85). Rollout rows avoid the leak but pair the ground-truth high band
with a low band the rollout generated, which is not the one that high band belongs to, so
MSE teaches the average. The rollout also runs the backbone unguided on a uniform 4-step
grid; from the same noise, guided and unguided low bands diverge from 0.03 to 0.57 endpoint
standard deviations across the 16 evaluation nodes, so the rollout states are also off the
inference trajectory.

Two ways to obtain properly paired rows were tested and both fail.

**Inverting the sampler onto the ground-truth low band.** Each guided Euler step was
inverted exactly on the evaluation grid by damped fixed-point iteration, targeting
mel(lowpass(reference)). Pre-registered pass: replay NMSE on the low band median ≤ 0.05,
p90 ≤ 0.15, and median ≤ 0.2 × the unpaired anchor. On 6 items, stopped early because the
outcome was not in doubt:

| quantity | value |
|---|---|
| replay NMSE, fp16 (training precision) | median 0.134, p90 0.797 |
| replay NMSE, fp32 | median 0.276, p90 1.188 |
| unpaired anchor (forward from random noise) | median 1.816 |
| steps hitting the 40-iteration cap | 72% |
| fp16 vs fp32 replay disagreement | up to 1.1 |
| replay spread across 3 inversions of one item | ~0.41 |
| inverted noise z*: excess kurtosis; neighbour correlation time / frequency | ~2.1; 0.45 / 0.33 |

The guided sampler is locally expansive, so the fixed point does not converge mid-to-late
in the solve, the inverted noise is far from Gaussian (the branch would never see such a
state at inference), and the replay depends on arithmetic precision. The inverted pairs
are neither accurate nor in-distribution.

**Forward pairing by prompt swap.** From one noise, solve once with the band-limited
prompt (the trajectory the branch sees) and once with the clean 24 kHz prompt (whose high
band would be the target), and ask whether the low bands agree. Same pass marks. On 40
items: low-band NMSE median 0.611, p90 1.105, minimum 0.186, no item below 0.05. fp16
against fp32 on the same solve is 0.000 on every item, so this is not precision. Swapping
the prompt moves the low band nearly as far as drawing new noise does, so a clean-prompt
high band is not paired with the band-limited trajectory's low band either.

Together: no ground-truth pairing exists for "the high band that belongs to this generated
low band". Supervision has to come from something that can evaluate an arbitrary low band,
such as a clean-prompt teacher's score used as a distribution-level loss, not from regression
onto stored targets.

### What SIM-o can see

Both speaker-similarity scripts (`zipvoice/eval/speaker_similarity/sim.py` and `cpsim.py`)
resample to 16 kHz before the verifier. Anything above 8 kHz is invisible to SIM-o. At a
16 kHz prompt the entire generated band is above what the verifier hears, so no high-band
method can move SIM-o there; at 8 kHz only the 4–8 kHz part of the generated band enters it.

## Caveats

**The stock arm is not the BWE=identity ablation.** Stock ZipVoice is a different
checkpoint that never saw LibriTTS fine-tuning, so every bwe-vs-stock delta confounds
"the extender helps" with "the fine-tuning helps." This is why the results table above is
not the final word; the `--bwe-bypass` ablation section resolves it, and the numbers there
supersede the informal "net the 24 kHz offset out" attribution made in *Reading the
numbers*. That estimate turned out to be right in magnitude at 8 kHz (−0.013 guessed,
−0.023 measured) but wrong in mechanism: the 24 kHz offset is sampling noise, not a
fine-tuning penalty.

**Sample size, and what a single p-value is worth here.** 76 utterances from 38 speakers
on LibriTTS, 76 from 76 speakers on VCTK. The `plugplay` 24 kHz condition (Application
study 1) is a known-zero effect that nonetheless reached p = 0.046 on SIM-o and moved
UTMOS by 0.05, so **no result in this document should be believed on the strength of one
test at α = 0.05.** What carries the 8 kHz conclusions is replication: two corpora, two
MOS predictors, agreement to within 0.001 SIM-o and 0.03 UTMOS on the extender contrast,
and t-statistics between 8.8 and 12.4 on P.808. SIM deltas below about 0.01 and MOS
deltas below about 0.05 are not resolvable at this sample size and are reported as
directional only.

**Degradation model.** The evaluation uses real resampling, but training band-limits in
the mel domain with a randomised 2–4 bin transition, jittered floor and low-level noise.
That mismatch is mitigated, not eliminated.

## Suggested next steps

1. ~~Build the `--bwe-bypass` inference flag and rerun.~~ Done; see the ablation section.
   The remaining question it raises is whether ZipVoice's dependence on the extender's
   output distribution is desirable. Training the joint model with the extender randomly
   dropped out would keep the fine-tuned weights usable on clean prompts without it, and
   would likely also remove the 0.05 P.808 penalty the fine-tuning currently costs on
   undegraded prompts.
2. **Drop `D_cross`, `D_harm`, the cutoff curriculum and the joint-stage reconstruction
   loss.** `D_cross` and `D_harm` learned nothing in 15000 steps across three independent
   runs, and their feature-matching terms (`fmatch_cross` 0.0028, `fmatch_harm` 0.0025,
   against `fmatch_spec` 0.0146) contribute little to the loss. Studies 2 and 3 then show
   the curriculum and all direct supervision of the extender output are undetectable in
   the evaluation as well. The reduced recipe is the joint flow-matching gradient plus
   `d_spec`, and it trains at 2930 steps/h against 1730. What studies 2 and 3 do *not*
   show is that the extender could be trained from scratch this way: both start from a
   frozen stage that used `bwe_recon` for 30000 steps. Retraining that stage without it
   is the experiment that would close the question. Note the scope: what is dispensable is
   the *supervision inside* the joint stage, not the joint stage itself. The frozen-stage
   plug-and-play arm shows the joint stage improves the extender by 0.079 P.808 at 8 kHz,
   so the reduced recipe still has to run it.
3. If speaker similarity at low prompt rates is the goal, the extender needs a
   conditioning path or a loss that carries speaker identity; reconstruction plus a single
   working spectrogram discriminator supplies only "plausible high band."
4. **Ship whichever checkpoint matches the prompts the deployment actually receives.**
   Above 8 kHz, plug-and-play wins on the mean and costs no measurable speaker similarity
   on unseen speakers, while the fine-tuned checkpoint wins in the tail — eliminating every
   8 kHz failure on both corpora — and pays −0.017 to −0.033 SIM-o for it. At 4 kHz the
   trade disappears and the fine-tuned checkpoint simply wins, by +0.34 / +0.22 P.808 at no
   significant identity cost. If prompts below 8 kHz are in scope at all, ship the
   fine-tuned checkpoint. Item 1 above is the fix that
   would let one checkpoint do both: dropping the extender out during joint training
   should keep the tail robustness while removing the co-adaptation that costs identity.
   Whichever ships, the extender to merge is the one from the end of the joint stage, not
   the frozen stage.
5. **Report tails, not only means, in any follow-up.** The single largest effect
   attributable to the joint fine-tuning at 8 kHz — the failure count going from one to
   zero, and the worst-quartile advantage exceeding the mean advantage by up to 1.7x —
   was invisible in mean-based reporting and was recovered from score files that had
   been sitting on disk since the original scoring pass. `local/bwe_stats.py` costs
   nothing to run and should be part of the standard scoring pass.
6. A full-band quality metric would help. DNSMOS and UTMOS both run at 16 kHz, so no
   metric in this study can see what the extender predicts above 8 kHz, which is all of
   its output at a 16 kHz or higher prompt. Log-spectral distance against the clean mel,
   restricted to the predicted band, would measure that directly.

## Reproducing

```bash
cd egs/zipvoice
# Stages 8-10: build the eval set, synthesise both systems at all rates, score.
stage=8 stop_stage=10 bash run_bwe_libritts.sh
```

Outputs land in
`${exp_root}/results/{bwe,bwe_bypass_resunet,bwe_bypass_full,stock}_sr{8000,16000,22050,24000}/`.

The out-of-domain replication is a separate recipe with the same four arms:

```bash
cd egs/zipvoice
bash run_bwe_vctk_eval.sh
```

Its stage 1 streams the ~80 prompt recordings it needs out of the 11.7 GB VCTK archive
over HTTP range requests (`pip install remotezip`), so no full download is required;
pass `vctk_zip=/path/to/VCTK-Corpus-0.92.zip` to read a local copy instead. Outputs land
in `${exp_root}/vctk_results/`.

The plug-and-play arm needs a merged checkpoint, which is built offline and then run
through the ordinary inference path:

```bash
cd egs/zipvoice
python3 local/make_plugplay_checkpoint.py \
      --stock-dir ${download_dir}/zipvoice \
      --bwe-dir ${exp_root}/zipvoice_bwe \
      --checkpoint-name iter-15000-avg-2.pt \
      --output-dir ${exp_root}/zipvoice_bwe_plugplay

for rate in 8000 16000 22050 24000; do
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_bwe \
            --model-dir ${exp_root}/zipvoice_bwe_plugplay/ \
            --checkpoint-name iter-15000-avg-2.pt \
            --tokenizer emilia \
            --test-list ${eval_dir}/test_sr${rate}.tsv \
            --res-dir ${results_dir}/plugplay_sr${rate} \
            --bwe-bypass none --num-step 16
done
```

The frozen-stage plug-and-play arm is the same merge with a different source checkpoint,
and needs no other change because the frozen stage never updated the TTS weights:

```bash
cd egs/zipvoice
python3 local/make_plugplay_checkpoint.py \
      --stock-dir ${download_dir}/zipvoice \
      --bwe-dir ${exp_root}/zipvoice_bwe_frozen \
      --checkpoint-name iter-30000-avg-2.pt \
      --output-dir ${exp_root}/zipvoice_bwe_plugplay_frozen
```

Both merged checkpoints are then evaluated exactly like the others, with `--bwe-bypass
none` so the extender runs, since the released TTS weights never learned to depend on it.

The tail analysis reads the per-utterance score files `zipvoice.eval.mos.dnsmos` already
wrote, so it needs no resynthesis and no GPU:

```bash
cd egs/zipvoice
python3 local/bwe_stats.py tail --dnsmos-dir ${data_root}/dnsmos
python3 local/bwe_stats.py tail --dnsmos-dir ${data_root}/dnsmos       --arms stock plugplay_frozen plugplay bwe --rates 8000
```

The paired WER contrasts read the `decode.txt` that stage 10 already wrote, so they need
no second ASR pass:

```bash
cd egs/zipvoice
python3 local/bwe_stats.py wer       --results-dir ${results_dir}       --conditions stock_sr8000 bwe_bypass_full_sr8000 plugplay_sr8000 bwe_sr8000       --baseline stock_sr8000
```

### Environment fixes required for stage 10

Scoring needed four fixes on Python 3.13 with current dependency versions:

- `s3prl` is imported by the WavLM `hubconf.py` inside the downloaded eval models and is
  not listed in `requirements_eval.txt`.
- `zhconv`, `zhon` and `funasr` are imported at module level in
  `zipvoice/eval/wer/seedtts.py`, so they are required even for `--lang en`.
- `jiwer` 4.x removed `compute_measures`; pin `jiwer==3.1.0`.
- `transformers` now honours the checkpoint's own dtype in `from_pretrained`, and
  whisper-large-v3 ships as fp16, so float32 input features no longer match the weights
  (`RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same`).
  Fixed in `zipvoice/eval/wer/seedtts.py` by casting features to `model.dtype`.

DNSMOS additionally needs two ONNX models that are not in the k2-fsa repository. Put them
in the `mos` subdirectory of `--model-dir`:

```bash
cd ${eval_model_dir}/mos
base=https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS
curl -sL -O ${base}/sig_bak_ovr.onnx   # P.835 SIG/BAK/OVRL
curl -sL -O ${base}/model_v8.onnx      # P.808 overall MOS
```

The eval models come from `hf download k2-fsa/TTS_eval_models` (~14 GB). Note that
`huggingface-cli` was removed in `huggingface_hub` 1.x and now only prints a deprecation
notice while exiting non-zero; `hf` is its replacement.
