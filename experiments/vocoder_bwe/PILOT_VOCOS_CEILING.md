# Pilot: what the pretrained Vocos does with a band-limited mel

Measured 2026-09-30 on the 76-utterance `bwe_eval` reference set. No TTS model
involved: reference audio -> mel -> `vocos-mel-24khz` -> audio. Scripts are in
the session scratchpad (`pilot_vocoder_ceiling.py`, `pilot_resample.py`).

Band figures are `band_frames.py` speech rows, in dB. Each file is
level-normalised by that script, so the rows are relative, and absolute RMS is
quoted separately where it matters.

| system | 4-6k | 6-8k | 8-12k | swing 8-12k | RMS |
|---|---|---|---|---|---|
| reference | -35.2 | -40.3 | -44.3 | 40.4 | -24.4 |
| vocos on the reference's own full-band mel | -35.2 | -40.3 | -43.6 | 38.2 | -24.5 |
| audio resampled 24k -> 8k -> 24k (the sr8000 input) | -50.3 | -75.8 | -78.9 | 10.0 | -24.6 |
| vocos on that resampled audio's mel | -48.6 | -74.9 | -74.7 | 20.9 | -25.6 |

Three things follow.

**The vocoder is transparent when the mel has content.** On a real full-band mel
it lands within 0.7 dB of the reference in every band and holds the swing at
38.2 against 40.4. None of the missing high band measured in any of the TTS arms
is the vocoder's doing, and a vocoder swap cannot gain anything on clean input.

**The pretrained vocoder fills almost nothing.** Given genuinely band-limited
audio it adds 4.2 dB at 8-12 kHz and lifts the swing from 10.0 to 20.9 -- some
content, correlated with speech, but 30 dB short of the reference. That 30 dB is
the headroom a trained upsampling vocoder has to claim, and it confirms the
obvious: every released vocoder is trained for faithfulness, so no off-the-shelf
checkpoint will do this. The experiment necessarily requires training.

**Vocos cannot decode the synthetic mel floor at all.** Band-limiting in the mel
domain the way `bwe.band_limit` does -- upper bins clamped to `log(1e-7)` --
makes the output near-silent: RMS -92.1 dB against the reference's -24.4. Half
the input channels shifted that far below the training distribution collapses
the prediction globally. This does not invalidate the adapter or dual arms,
where the clamp is applied to the TTS *prompt* and never to the vocoder input,
but it does fix one design constraint: a vocoder-side experiment has to be
trained and evaluated on resample-based band limiting (`bwe.resample_band_limit`),
never on the clamp.
