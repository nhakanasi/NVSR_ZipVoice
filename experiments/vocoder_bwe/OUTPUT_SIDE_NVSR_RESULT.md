# Upbanding at the output instead of at the prompt

Measured 2026-09-30 on the 76-utterance `bwe_eval` set (LibriSpeech-PC), stock
ZipVoice `download/zipvoice/model.pt`, `--num-step 16`, NVSR from the released
`ssr_eval` checkpoint. Nothing was trained.

The question was whether replacing Vocos with a vocoder that already extends
bandwidth beats the systems measured so far, without spending a training run.
NVSR is such a model: narrowband mel -> ResUNet predicts the wideband mel ->
voicefixer's 44.1 kHz vocoder renders it. The existing cascade baseline runs
that path on the *prompt*; this runs it on stock ZipVoice's *output*.

`bwe_eval` is LibriSpeech-PC and NVSR was trained on VCTK, so the speaker
overlap that contaminates `vctk_bwe_eval` does not apply here.

## Result at sr8000 (4 kHz prompt cutoff)

Band figures are `band_frames.py` speech rows, in dB. The stock row reproduces
the previously recorded -78.9 / -77.2 and SIM-o 0.610 / UTMOS 3.89 exactly, so
the rows carried over from `DUAL_MEMORY_CHANNEL_RESULT.md` are comparable.

| system | SIM-o | UTMOS | RTF | 4-6k | 6-8k | 8-12k | swing 8-12k |
|---|---|---|---|---|---|---|---|
| reference | - | - | - | -35.2 | -40.3 | -44.3 | 40.4 |
| stock | 0.610 | 3.89 | 0.147 | -50.3 | -77.2 | -78.9 | 10.7 |
| cascade (NVSR on the prompt) | 0.563 | 4.14 | - | - | -48.2 | -46.8 | 31.5 |
| dual, memory from `y` | 0.620 | 4.17 | 0.342 | - | -48.9 | -54.6 | 30.9 |
| projgan (adapter arm) | 0.608 | 4.15 | 0.376 | - | -54.6 | -57.1 | 40.8 |
| **NVSR on the output** | **0.599** | **3.96** | **0.157** | **-35.3** | **-48.3** | **-45.4** | **27.1** |

## What it wins

**Band level, by a clear margin, and it is the only system that gets the shape
right.** 8-12 kHz lands at -45.4 against the reference's -44.3, inside 1.1 dB,
where the best trained arm reached -54.6. At 4-6 kHz it is -35.3 against -35.2,
effectively exact, a band every other arm left 15 dB down.

**Cost.** RTF 0.157 all in: 0.147 for stock synthesis plus 0.010 for the
restoration. The dual variant costs 0.342 and projgan 0.376, because both add
backbone forwards inside the solve. Worth recording that Vocos itself is 0.0008
of the stock RTF, 0.5%, so anything done in the vocoder's place is free by
comparison.

**Identity is nearly stock**, 0.599 against 0.610, losing 0.011. Prompt-side
cascade loses 0.047, because restoring the prompt makes ZipVoice imitate NVSR's
artifacts; restoring the output cannot do that, since the prompt is untouched.

## What it loses, and why it is not the answer

**UTMOS 3.96, against 4.14 for prompt-side cascade, 4.15 for projgan and 4.17
for the dual variant.** It is the worst of every band-filling system and only
0.07 above stock. The acceptance gate as stated passes on all three legs
(+33.5 dB over stock, UTMOS >= 3.9, SIM-o >= 0.34), but the stronger criterion
-- UTMOS matching or beating the cascade -- fails, and it fails by more than the
spread between all the other arms.

**Swing 27.1 at 8-12 kHz**, below the dual variant's 30.9 and well below
projgan's 40.8, with the reference at 40.4. So the energy is at the right level
but tracks speech less well than either trained arm: correct average, more veil.

The likely cause of both is distribution. NVSR is trained on real band-limited
speech, and stock ZipVoice's output is not that: its high band is not a clean
resampling roll-off but a generated near-floor region, so the restorer is asked
to extend something outside its training distribution. It gets the level right
and the fine structure wrong, which is exactly what a high level with a low
swing and a depressed UTMOS looks like.

## Bearing on the vocoder-swap idea

This is the released-weights, no-training probe of that idea, and it separates
the two claims cleanly. The *architecture* argument holds: putting the band
extension after the mel, on the output side, keeps identity essentially at stock
and costs almost nothing, which no mel-domain arm managed. The *off-the-shelf*
shortcut does not: no released extender is trained on generated mel, and the
UTMOS cost of that mismatch is larger than the UTMOS spread between every
trained arm measured so far.

It is also still a cascade -- a second model after the solve -- so it stands as
a baseline, not as a candidate system.

## The other two rates, which change the verdict

| condition | system | SIM-o | UTMOS | 4-6k | 6-8k | 8-12k | swing 8-12k |
|---|---|---|---|---|---|---|---|
| reference | - | - | - | -35.2 | -40.3 | -44.3 | 40.4 |
| sr16000 (8 kHz cutoff) | stock | 0.730 | 4.08 | -37.3 | -45.6 | -60.5 | 35.3 |
| sr16000 | NVSR on the output | 0.726 | 4.08 | -37.1 | -43.5 | -45.3 | 32.6 |
| sr24000 (clean prompt) | stock | 0.739 | 4.20 | -37.2 | -42.5 | -47.3 | 42.7 |
| sr24000 | NVSR on the output | 0.738 | 4.20 | -37.2 | -42.0 | -41.7 | 32.4 |

**At sr16000 the UTMOS deficit disappears entirely.** 8-12 kHz rises 15.2 dB,
-60.5 to -45.3, landing within 1.0 dB of the reference, while UTMOS is identical
at 4.08 and SIM-o moves 0.004. Swing falls 35.3 to 32.6, a small cost. This is
the rate the adapter arm failed at, managing +4.6 dB, and it is the reason the
dual design was built; output-side restoration gets +15.2 dB there for no
measurable quality or identity cost and RTF 0.01.

So the sr8000 UTMOS deficit is specific to the hardest condition, where 8 kHz of
spectrum has to be invented from 4 kHz of evidence. It is not a general property
of putting the extender after the solve.

**On a clean prompt it is not harmless, though the metrics do not catch it.**
With nothing missing, it still adds 5.6 dB at 8-12 kHz, -47.3 to -41.7, which
overshoots the reference's -44.3 by 2.6 dB, and drops swing from 42.7 to 32.4.
UTMOS and SIM-o are unchanged to within 0.01, so both would call this free; the
band figures say it brightens the output and flattens its dynamics. Anything
that ships this unconditionally needs a cutoff detector, or needs the
restoration to be a no-op when the input is already full-band.
