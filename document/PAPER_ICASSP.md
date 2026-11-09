# ICASSP submission plan: sampling-rate-agnostic prompts via a jointly-trained mel bandwidth extender

This distils `BWE_RESULTS.md` (five prompt rates, two corpora, seven arms, n = 76 per
condition) into the subset that fits four ICASSP pages plus one page of references. It
states the claims that survive, the tables that must appear, what to cut, and the
objections a reviewer will raise.

`BWE_RESULTS.md` stays the technical record. This file is the paper's scope decision.

## 1. The claim to build the paper around

Zero-shot TTS conditions on a prompt recording. ZipVoice extracts that prompt as a 100-bin
mel at 24 kHz, so a prompt captured at 8 kHz arrives with everything above 4 kHz empty --
a mel the model never saw in training. Resampling cannot recreate it. The proposal is to
insert an NVSR-style ResUNet that fills the missing band *in the mel prompt*, and to train
it jointly with the TTS so that the TTS's own flow-matching loss supervises it.

The headline result is rate invariance. With a 4 kHz prompt, where 83% of the mel band has
to be synthesised, the system scores UTMOS 4.27 and DNSMOS P.808 3.94, against 4.22 and
3.95 at an 8 kHz prompt and 4.16 and 3.95 at a clean 24 kHz prompt. Stock ZipVoice falls
from 3.89 to 2.80 UTMOS over the same range. Prompt bandwidth stops mattering.

The reason to care is deployment. The target application is voice-preserving speech
translation on edge hardware, where the prompt arrives at whatever rate the capture path
produced: 8 kHz from a telephony leg, 16 kHz from an embedded microphone stack, 44.1 or
48 kHz from consumer hardware. A cloud service can require a 24 kHz recording. A device in
the field takes what the hardware in front of it gives, and cannot switch to a larger model
when that turns out to be 8 kHz. Invariance to prompt rate therefore has to be built in,
and it has to be cheap: the extender adds 7.46M parameters to a 122.66M-parameter TTS
model, a 6.1% overhead, running once over the prompt rather than once per synthesised
frame, with no vocoder in the added path because it operates on the mel the TTS already
consumes.

Recommended title: *Sampling-rate-agnostic prompting for zero-shot TTS via a jointly
trained mel bandwidth extender.*

### Contributions, in the order they should be listed

1. A mel-domain bandwidth extender applied to the *prompt* of a flow-matching zero-shot
   TTS rather than to output audio, trained jointly so the TTS's flow-matching gradient
   acts as a task-aware reconstruction loss.
2. Rate invariance: synthesis quality from a 4 kHz prompt matches quality from a clean
   24 kHz prompt, where the unmodified model loses 1.47 UTMOS.
3. Transferability: the trained extender bolted onto the *unmodified released checkpoint*
   recovers roughly three quarters of the gain at 8 kHz, so deployment needs no TTS
   retraining and no new weights to distribute.
4. A negative result that simplifies the recipe: the bandwidth curriculum and every direct
   loss on the extender output can be removed with no measurable effect, and two of the
   three discriminators never train at all. What remains -- the flow-matching gradient plus
   one spectrogram discriminator -- trains 1.7x faster.

Contribution 4 is what separates this from an engineering report, and it should not be
buried in a final subsection. Reviewers reward an ablation that removes machinery.

## 2. Section budget for four pages

Two columns, 10 pt. Roughly 950 words fills a page of pure text; a half-column float costs
about 200 words of space. The budget below lands at about 3100 words, two figures and two
tables.

| Section | Words | Floats |
|---|---|---|
| Abstract | 150 | -- |
| 1. Introduction | 500 | -- |
| 2. Method | 650 | Fig. 1 (architecture) |
| 3. Experimental setup | 350 | -- |
| 4. Results | 850 | Fig. 2 (rate ladder), Table 1 |
| 5. Ablations | 450 | Table 2 |
| 6. Conclusion | 130 | -- |

## 3. Method: what has to be said, and what does not

Three things are load-bearing and must survive the cut.

**The insertion point.** The extender runs on the prompt features *before* the speech
condition is built. The flow-matching target stays the clean full-band mel and the
flow-matching loss is untouched. That single design choice is why joint training works: the
existing loss becomes a task-aware bandwidth-extension loss at no additional cost, and it
teaches the extender to produce a mel the *TTS* finds useful rather than one that merely
looks right.

**The hard composite.** The network predicts a residual over the masked band only, and the
output is `(1 - m) * M_hat + m * M_tilde`, so observed bins are copied through bit-exact.
This makes the extender provably the identity when the cutoff equals Nyquist, which the
paper then uses as a free null condition (Section 4).

**The degradation model.** Band-limiting happens in the mel domain: bins above the cutoff
are set to the log floor, which for this feature configuration is -1.6118 rather than zero,
with a randomised 2-4 bin cosine transition, jittered floor and low-level noise. State the
floor value; a reader reproducing this will otherwise fill with zero and get a different
model.

Compress to one line each: the mel-bin-centre mapping, the ResUNet channel widths, the
cutoff-sampling equation, and the F0 estimator. They are implementation detail and the
reader can be pointed at the released code.

Do not spend space defending the deterministic ResUNet against a flow-matching extender.
One sentence: a flow generator would need a differentiable multi-step rollout nested inside
the TTS's own flow step, for a tensor that is never vocoded.

### Figure 1

A single column-width diagram: prompt mel, band-limit block, ResUNet with two input
channels (masked mel, binary mask), hard composite, then into the speech condition with the
flow-matching loss against the clean mel. The three discriminators hang off the composite.
Mark the gradient path from the flow-matching loss back through the extender -- that is the
paper's central idea and it should be visible in the figure.

### Draft text for Section 2

What follows is drafted at roughly the section budget and is meant to be edited down rather
than written from scratch. Numbers in it are the ones the released recipe actually uses.

#### 2.1 Bandwidth extension inside the conditioning path

ZipVoice is a flow-matching text-to-speech model that clones a voice from a speech prompt.
The prompt enters the model only as a log-mel spectrogram: an 80 ms window, 256-sample hop,
100 mel filters spanning 0 to 12 kHz, extracted at 24 kHz. A prompt recorded at a lower rate is
resampled to 24 kHz before extraction, which restores the sample rate but not the content, so
every mel bin above the original Nyquist frequency sits at the empty-bin floor. For this feature
configuration that floor is the natural log of the magnitude clamp scaled by the feature scale,
`log(1e-7) * 0.1 = -1.6118`, not zero; a reader reproducing the degradation must use that value.
The model has never seen conditioning of that shape, and speaker similarity and perceived quality
degrade accordingly.

We insert a bandwidth extender into the conditioning path, at the mel level. Let `M` be the clean
prompt mel and `m` a per-bin mask that is one where the band is missing and zero where it is
observed. The band-limited prompt is `M_tilde = (1 - m) * M + m * floor`. The extender is an
NVSR-style ResUNet with four encoder stages of residual convolution blocks at widths 32, 64, 128
and 256, a bottleneck, and a mirrored decoder with skip connections. It takes two input channels,
the band-limited mel and the mask itself, and predicts a residual `R`. The output is the hard
composite

    M_fb = (1 - m) * M_tilde + m * (M_tilde + R),

so observed bins are copied through bit-exact and the network only ever writes into the band it
is asked to invent. Two properties follow. The extender is exactly the identity when the cutoff
equals Nyquist, which the evaluation uses as a free null condition; and the observed evidence
cannot be corrupted by the network, which is the same replacement trick NVSR applies in the
audio domain.

The insertion point is the contribution. `M_fb` replaces the prompt mel in the speech condition,
and nothing else in ZipVoice changes: the flow-matching target remains the clean full-band mel
`M` and the flow-matching loss is written exactly as before. Because the condition is now a
function of the extender's parameters, that untouched loss backpropagates into the extender and
becomes a task-aware bandwidth-extension objective at no additional cost. The extender is not
trained to produce a mel that looks correct; it is trained to produce a mel the text-to-speech
model can use. Nothing in the added path runs at waveform rate. There is no analysis-synthesis
round trip and no second vocoder, because the extender consumes and produces the representation
the model already computes -- unlike an audio-domain super-resolution front end, which must
synthesise a full-band waveform that the feature extractor immediately discards down to 100 mel
bins. The extender adds 7.46 M parameters to ZipVoice's 122.66 M, a 6.1% overhead, and one
forward pass over a 6.2-second prompt costs 0.43% of the synthesis it precedes, because it runs
once over the prompt while the solver runs sixteen times over the output.

#### 2.2 Loss design: structure from reconstruction, detail from discriminators

The extender is trained under three kinds of supervision, and each is responsible for a
different scale of the problem.

Reconstruction fixes the coarse structure. We take a mean absolute error between `M_fb` and the
clean mel `M`, weighted by the mask so it applies only to the restored band. A per-bin regression
loss of this kind is what makes the restored band land in the right place spectrally, and on its
own it is also what makes it oversmooth: the conditional mean of a set of plausible high bands is
a blurred high band with no harmonic structure. (We also implement the log-spectral-distance,
inter-band level and neighbour-dissimilarity trio of HRTFformer as an alternative; the reported
systems use the L1.)

Adversarial supervision fixes the fine structure. Three discriminators, all operating on the mel
grid rather than on audio, attack the three ways an extended band looks wrong. A multi-scale
spectrogram discriminator sees the composite mel time-pooled by factors of one, two and four, and
is the general-purpose realism critic. A cross-band discriminator divides the 100 bins into 20
subbands of 5, forms each subband's temporal energy envelope and correlates it against the
envelope of the observed low band, so it judges whether the invented band moves *coherently* with
the band that was actually measured -- a spectrally plausible high band that fails to follow the
low band's dynamics is exactly the artifact a per-bin loss cannot see. A harmonic discriminator
samples the mel at the first twelve multiples of an estimated F0 by bilinear interpolation and
judges the harmonic comb directly, so that voiced structure is criticised where it lives rather
than averaged over the spectrum. All three use a hinge objective with feature matching against
the real activations.

The total generator objective is

    L = L_fm + 1.0 * L_recon + 0.1 * sum_q L_adv^q + 2.0 * sum_q L_fmatch^q,

with `q` over the three discriminators. Adversarial gradients reach the extender only, since they
enter through `M_fb`, so the discriminators cannot destabilise the text-to-speech weights
directly. Three optimizers are used: ScaledAdam for the text-to-speech parameters, ScaledAdam for
the extender, and plain AdamW at 2e-4 for the discriminators, whose clipping behaviour is a poor
fit for ScaledAdam's per-parameter heuristics.

#### 2.3 Curriculum: easy cutoffs first

One cutoff is drawn per utterance, in the range 1 to 12 kHz. Sampling it uniformly from the start
makes training unstable, because a 1 kHz cutoff asks the network to invent eleven twelfths of the
spectrum before it has learned to reproduce any of it. We therefore adopt the easy-to-balanced
schedule of AnyBand: the cutoff distribution is a mixture of an exponential branch biased towards
high cutoffs and a uniform branch, with the uniform weight annealed by

    lambda_s = [1 - cos(pi * min(s / (rho * S), 1))] / 2,

so training begins on nearly full-band prompts, where the task is close to the identity the hard
composite already provides, and reaches a uniform distribution over the whole cutoff range at
`rho * S` steps. We use `beta = 4` for the exponential branch and `rho = 0.7`.

Training proceeds in three stages. A pretraining stage trains the extender alone on
reconstruction, with the text-to-speech model absent from the objective, which gives the joint
run a sane initialisation. A frozen stage adds the flow-matching loss with all text-to-speech
weights frozen, so the gradient through the frozen model is what teaches the extender to serve
the task; the curriculum runs in both of these. A joint stage then unfreezes the text-to-speech
model at a low learning rate and enables the three discriminators, and samples cutoffs uniformly,
the curriculum having done its work. The three stages in the reported recipe are 8k, 30k and 15k
iterations.

Finally, the degradation model itself is randomised, because a mel-domain brick wall is not what
a real resampler produces. The mask edge is a raised cosine whose width is drawn per utterance
around three mel bins, the floor carries a uniform jitter, and low-level Gaussian noise is added
in the dead band. This is mitigation for the train/test mismatch, not a solution to it, and it is
listed as such in the limitations.

## 4. Experimental setup: the parts that buy credibility

Compress hard, but keep these.

Training: three stages on LibriTTS `train-clean-360` at 24 kHz, single GPU, starting from
the released Emilia ZipVoice checkpoint. Reconstruction-only pretrain (8000 steps), then
the extender against a frozen TTS with the flow-matching gradient live (30000 steps), then
joint adversarial fine-tuning (15000 steps).

Evaluation: 76 utterances per condition. LibriTTS dev-clean is in-domain for the
fine-tuning; **VCTK 0.92, 76 unseen speakers, is the out-of-domain replication and every
headline number reproduces there.** Say this early -- it is the difference between a result
and an artefact of evaluating on the training corpus.

Metrics: UTMOS and DNSMOS P.808 for naturalness, SIM-o (WavLM-large speaker verification)
for identity, Seed-TTS WER (whisper-large-v3) for intelligibility. Prompt rates 4, 8, 16,
22.05 and 24 kHz, each resampled back to 24 kHz before it reaches the model.

**Report the natural-speech topline in the same table.** Scoring the 76 natural recordings
of the target sentences with the same evaluators against the same prompts gives SIM-o 0.740,
UTMOS 4.11 and P.808 3.849 on LibriTTS, and 0.726, 4.06 and 3.748 on VCTK. Two things need
that row. SIM-o has a ceiling near 0.74, because it compares two different recordings of one
speaker, so the 0.739 the unmodified model scores at a 24 kHz prompt is saturation rather
than a weak result and a 0.01 difference cannot be read as a large regression. And both MOS
predictors sit at or below the synthesis they are scoring, which is the honest way to say
that they carry no information above roughly 4.1 UTMOS and only the degraded conditions
separate systems at all. One extra row, and it pre-empts the reviewer who asks whether 0.74
is any good. The same row retires WER: Whisper transcribes the natural recordings at
5.13% against 2.5 to 3.2% for every synthetic arm, so intelligibility on this set is
already better than the human speech being imitated and no arm can separate on it.

**The null condition is worth three sentences of its own.** At a 24 kHz prompt the cutoff
equals Nyquist, so the extender is the identity by construction and any difference from
stock is pure sampling noise. Measured on paired per-utterance differences this puts the
noise floor at about 0.011 P.808 on LibriTTS, 0.05 UTMOS, and 0.6 pp WER. Every claim in
the paper is then stated against a measured floor rather than an assumed one. Reviewers
notice this, and it costs almost nothing to include.

Drop entirely: DNSMOS P.835 (SIG/BAK/OVRL moved by essentially nothing and the space is not
available to explain why), and the `bypass_resunet` arm.

## 5. Results: two figures' worth of story, one table

### Figure 2 -- the rate ladder

Plot DNSMOS P.808 against recovered bandwidth (975, 4000, 8000, 10000 Hz) for four arms:
stock, extender-off fine-tuned, plug-and-play, full system. Two panels, LibriTTS and VCTK.
This figure carries the whole result and replaces three tables of the source document.

The shape to point out in the caption: the extender contributes nothing until roughly
8000 Hz of band is missing, and above that threshold its contribution scales with how much
is missing. Full system minus stock, P.808:

| recovered | prompt rate | P.808 LibriTTS | P.808 VCTK | UTMOS LibriTTS | UTMOS VCTK |
|---|---|---|---|---|---|
| 0 Hz | 24000 | -0.035 | -0.069 | -0.04 | +0.01 |
| 975 Hz | 22050 | -0.024 | -0.070 | 0.00 | -0.05 |
| 4000 Hz | 16000 | -0.032 | -0.061 | +0.11 | +0.10 |
| 8000 Hz | 8000 | +0.410 | +0.338 | +0.33 | +0.33 |
| 10000 Hz | 4000 | +0.730 | +0.701 | +1.47 | +1.31 |

Both predictors were run over all four arms at all five rates, so the figure can be drawn
from measurements rather than interpolated, and the two predictors agree on the curve's
shape in every cell. The 0 Hz row is the exact null: at a 24 kHz prompt the composite makes
the extender inert, and two arms carrying different extenders on identical TTS weights
produce bit-identical output, so the residual there is entirely the cost of fine-tuning on
undegraded prompts.

Nothing is sampled between 4000 and 8000 Hz recovered, so the threshold's location is
unmeasured. Say so.

### Table 1 -- attribution at the two rates that matter

The four arms form a 2x2 over {extender on, off} x {TTS fine-tuned, released}. The
extender-off fine-tuned arm receives byte-identical input to what stock receives, so it
isolates the fine-tuning; the plug-and-play arm is the trained extender merged onto
released weights, so it isolates the extender. Report 4 kHz and 8 kHz only; 16 kHz and
above is in Figure 2.

| corpus | arm | UTMOS 8k | P.808 8k | SIM-o 8k | UTMOS 4k | P.808 4k | SIM-o 4k |
|---|---|---|---|---|---|---|---|
| LibriTTS | stock | 3.89 | 3.539 | 0.610 | 2.80 | 3.210 | 0.368 |
| LibriTTS | fine-tuning only | 4.02 | 3.648 | 0.609 | 3.44 | 3.331 | 0.370 |
| LibriTTS | extender only | 4.14 | 3.860 | 0.589 | 3.87 | 3.600 | 0.377 |
| LibriTTS | both | **4.22** | **3.949** | 0.586 | **4.27** | **3.940** | 0.374 |
| VCTK | stock | 3.92 | 3.488 | 0.514 | 2.96 | 3.094 | 0.322 |
| VCTK | fine-tuning only | -- | 3.565 | 0.502 | 3.62 | 3.386 | 0.307 |
| VCTK | extender only | 4.15 | 3.726 | 0.511 | 3.91 | 3.577 | 0.317 |
| VCTK | both | **4.25** | **3.826** | 0.478 | **4.27** | **3.795** | 0.310 |

Three findings come out of this table and each deserves a short paragraph.

**The extender, not the fine-tuning, does the work.** At 8 kHz the +0.33 UTMOS headline
splits into +0.13 from LibriTTS fine-tuning and +0.20 from the extender, the latter twenty
times the noise floor.

**Plug-and-play transfers.** The extender merged onto unmodified released weights recovers
78% of the P.808 gain on LibriTTS and 70% on VCTK at 8 kHz, at no significant
speaker-similarity cost out of domain (-0.003, p = 0.73). This is the deployment story: no
retraining, no redistributed TTS weights, and the system degrades gracefully rather than
catastrophically if the extender is switched off, because released weights never learned to
depend on it.

**The identity cost is real, and it disappears for an uninteresting reason at 4 kHz.** At
8 kHz the full system gives up 0.036 SIM-o on VCTK, of which about a third is the
fine-tuning and two thirds the extender: it synthesises a high band that sounds plausible
and belongs to no one in particular. At 4 kHz no contrast between arms is significant, but
that is a floor effect rather than a benefit -- speaker identity has already collapsed for
every arm, stock included (0.322 on VCTK against 0.514 at 8 kHz). Write both halves of that
sentence together or the result reads as a win.

### MCD -- the signal-level check the reviewer will ask for

Both MOS predictors resample to 16 kHz internally, so neither observes the band being
restored. A reviewer will notice. Mel-cepstral distortion against the natural rendition of each
target sentence closes that hole: order-34 WORLD mel-cepstra, coefficient zero dropped, DTW
aligned, at 24 kHz. Full system minus stock, in dB, negative is better, n = 76 per cell:

| recovered | prompt rate | MCD LibriTTS | MCD VCTK |
|---|---|---|---|
| 0 Hz | 24000 | -0.13 | -0.12 |
| 975 Hz | 22050 | -0.25 | -0.25 |
| 4000 Hz | 16000 | -2.93 | -2.99 |
| 8000 Hz | 8000 | -6.41 | -5.44 |
| 10000 Hz | 4000 | -9.36 | -8.64 |

Two things to say about it. First, the ordering stock, reconstruction-only, plug-and-play, full
system holds in every row where anything is being recovered, so a metric that cannot be fooled by
plausible-sounding output agrees with the predictors. Second, MCD resolves the 16 kHz condition
that the predictors could not: about 3 dB on both corpora at p below 1e-28, in a band the
predictors are structurally blind to.

Report absolute MCD in a footnote, not the body. Every arm sits near 7.8 dB at a 24 kHz prompt
because the reference is a different rendition of the same sentence by the same speaker, so
delivery differences dominate the floor. The numbers rank arms; they must not be compared against
MCD figures from a resynthesis task.

### One sentence each, no table

WER carries no signal at any rate for any arm: everything lands inside the 0.6 pp measured
noise floor. And the fine-tuned checkpoint removes the failure tail -- counting utterances
below stock's own 10th percentile at an 8 kHz prompt, stock has 8 of 76, plug-and-play 1,
the full system 0, on both corpora. That line is worth including; it is the strongest
argument for joint fine-tuning over plug-and-play and it costs two sentences.

## 6. Ablations: the section that makes this a paper

### Table 2

| variant | P.808 8k (LibriTTS / VCTK) | SIM-o 8k VCTK | steps/h |
|---|---|---|---|
| full recipe | 3.949 / 3.826 | 0.478 | 1730 |
| no cutoff curriculum | 3.935 / 3.850 | +0.0122 (p = 0.043) | 2930 |
| no direct loss on extender output | 3.917 / 3.819 | +0.0156 (p = 0.029) | 2930 |
| extender from frozen stage, no joint GAN | 3.782 / 3.675 | +0.011 (p = 0.089) | -- |
| extender trained on reconstruction only | 3.745 / 3.721 | 0.514 | -- |

Three results, in this order.

**The cutoff curriculum can be removed.** Sampling cutoffs uniformly from step zero changes
nothing measurable in naturalness, intelligibility or the failure tail on either corpus.

**Every direct loss on the extender output can be removed.** With reconstruction, all three
discriminators, the adversarial and feature-matching terms set to zero, leaving the
flow-matching gradient as the extender's only supervision in the joint stage, the result is
again indistinguishable. Two of the three discriminators never learned anything in the
first place: the cross-band and harmonic discriminators sat at hinge losses of 0.98 and
0.96 against a trivial baseline of 1.0 across three independent runs, while only the
spectrogram discriminator reached 0.64.

Scope this claim carefully or a reviewer will overstate it and then reject it. Both
variants start from a frozen stage that *did* use reconstruction for 30000 steps. The
finding is that direct supervision is unnecessary *in the joint stage*, not that the
extender could be trained from scratch on the flow-matching gradient alone.

**But the joint adversarial stage itself is not removable.** Merging the extender taken
from the end of the frozen stage instead gives up 0.079 P.808 on LibriTTS and 0.051 on VCTK
at 8 kHz (p = 0.000016 and 0.014), with nothing at any other rate. This is the cleanest
measurement in the study: the two arms carry byte-identical TTS weights, and at a 24 kHz
prompt they produce bit-identical waveforms, so the null is exactly zero rather than merely
small. The joint stage therefore does two separable things -- it co-adapts the TTS to the
extender, and it improves the extender itself by a comparable amount -- and the second
survives transplant onto frozen released weights.

**And reconstruction alone is not enough.** The last row is the same ResUNet trained
only on L1 reconstruction of the missing band, with no flow-matching gradient and no
discriminators, then merged onto the released weights: NVSR's objective at our insertion
point. It beats stock at every rate -- +0.086 and +0.123 P.808 at 4 kHz on LibriTTS and
VCTK, +0.207 and +0.233 at 8 kHz, all p < 1e-4 -- so a mel-domain extender helps even
when trained the conventional way. But the flow-matching-supervised extender at the same
insertion point gains 0.390 and 0.483 at 4 kHz, four times as much, and UTMOS separates
them by 0.74 and 0.69. The advantage grows with the bandwidth that has to be invented,
which is what the mechanism predicts: reconstruction barely constrains a band that is
two thirds missing, and only the downstream loss says which plausible high band the TTS
can actually use. This is the load-bearing ablation for the paper's central claim.

The conclusion to state: the working recipe reduces to the flow-matching gradient plus one
spectrogram discriminator, and it trains 1.7x faster than the published objective it was
derived from.

## 7. What to cut, explicitly

All of this is in `BWE_RESULTS.md` and none of it belongs in four pages.

- DNSMOS P.835 in full. One clause at most.
- The `bypass_resunet` arm and the dependence failure it exposes. Interesting, but it needs
  half a column to explain and supports no claim in the paper.
- The three separate noise-floor derivations. Keep one number per metric.
- Per-utterance Wilcoxon tests alongside the paired t-tests.
- The 16 kHz and 22.05 kHz rows of every table. Figure 2 carries them.
- The training-instability history. Not a result.
- The tail analysis as its own section. Two sentences in Results.

## 8. Objections a reviewer will raise

These are real gaps, not presentation problems. Decide before submitting how each is
answered.

**No audio-domain super-resolution baseline.** This is the most likely cause of rejection.
Note that it is not a novelty objection.
Applying a bandwidth extender to the mel *prompt*, supervised by the TTS's own
flow-matching gradient, is new; no prior work places one there. The objection is about
necessity rather than originality. The motivating problem -- a low-rate prompt degrading
zero-shot synthesis -- has an obvious off-the-shelf fix that needs no research at all:
run an audio super-resolution model over the prompt waveform and hand the result to the
unmodified TTS. Nobody has published that combination either, because it is two library
calls. An unbeaten obvious alternative sinks a paper faster than a thin novelty claim does,
so the comparison has to be in the table.

The intended baseline was NVSR at its released 44.1 kHz configuration, because our
extender is the same architecture, which would turn the comparison into an ablation over
*where the network sits*. That could not be run: both `ssr_eval`, which distributes the
NVSR checkpoint, and AudioSR pin dependencies that no longer build on current Python, and
neither has a maintained release. Reviving either means pinning an old interpreter, which
is worth doing before a camera-ready but was not worth blocking on here.

What was run instead splits the same question along a different axis, and is arguably the
more direct test. The `recon-only` arm is the *same* ResUNet, in the same place in the
pipeline, trained only on L1 reconstruction of the missing band -- NVSR's objective, with
no TTS gradient and no discriminators -- then merged onto the released ZipVoice weights.
Comparing it to the plug-and-play arm holds the architecture, the insertion point and the
TTS weights fixed and varies only the supervision. It therefore isolates the paper's actual
claim, that the flow-matching gradient is a better training signal for a prompt extender
than reconstruction is, rather than the incidental claim that mel beats waveform.

Say plainly in the limitations that a published audio-domain system was not run, and why.
An acknowledged gap with a substituted experiment that tests the same mechanism is a much
weaker target than an unacknowledged one.

The substituted arm was run in full: 76 utterances at 4 and 8 kHz prompts on both
corpora, scored on SIM-o, WER, UTMOS and both DNSMOS predictors, with paired
per-utterance tests. Section 6 carries the result. The proposed supervision wins by a
wide margin at 4 kHz and a narrower one at 8 kHz, which is the placement-and-supervision
argument demonstrated rather than assumed.

The cost argument is now measured rather than asserted, and should be reported with
numbers. The extender adds 7.46M parameters to ZipVoice's 122.66M, a 6.1% overhead, and
one forward pass over a 6.19-second prompt takes 3.58 ms on an RTX 5070 Ti or 66.71 ms
on CPU, against 873.62 ms for the 16-step synthesis it precedes -- 0.41% of one
utterance. End to end the extender is invisible: aggregate RTF over 76 LibriTTS
utterances at an 8 kHz prompt is 0.147 for the full system against 0.150 for stock,
with a run-to-run spread of 0.0004 -- the extender arm measures marginally faster,
because it generates slightly longer utterances and RTF divides by output duration.
Quote that as an upper bound, not as a speed-up. The structural reason matters more
than the figure: the extender runs once
over the prompt while the solver runs sixteen times over the output, and there is no
vocoder anywhere in the added path, because it consumes and produces the mel the TTS
already computes. An audio-domain front end puts a full-band waveform model and its
neural vocoder into the prompt path to produce a signal the feature extractor then
immediately discards down to 100 mel bins. Under the edge deployment this work targets,
parity on quality at that cost ratio would still be a win. `local/cost_bench.py`
reproduces the measurement.

**Listening test.** Every naturalness number in the study is a predictor, and both UTMOS
and DNSMOS run at 16 kHz internally, so neither can see what the extender synthesises above
8 kHz. MCD covers the signal-fidelity half of that gap and confirms the
ordering, but it says nothing about whether the synthesised band sounds right. A subjective test is therefore not optional confirmation of the objective numbers --
it is the only measurement that observes the band the method is about. A design that fits
the paper is given below.

### Listening test design

Naturalness MOS on a 5-point scale, plus a separate similarity MOS against the prompt,
since the objective results show the two move in opposite directions and a single score
would hide that.

Conditions, six in total: stock and the full system at 4 kHz and at 8 kHz prompts, the
plug-and-play arm at 8 kHz, and stock at a 24 kHz prompt as an upper anchor. Add the
audio-domain super-resolution baseline at both rates if that experiment lands in time.
Twelve utterances per condition, drawn from VCTK so the speakers are unseen, gives 72 to 96
stimuli. Fifteen to twenty listeners with every listener rating every condition supports a
within-subject test, which is what the paired objective analysis already uses and what the
small effect sizes need.

Two contrasts must reach significance for the paper's claims to hold subjectively: the full
system against stock at 4 kHz, which the objective numbers put at 1.47 UTMOS and should be
unmissable, and the full system against plug-and-play at 8 kHz, which is small and is the
one at genuine risk. Report the similarity scores whether or not they favour the method;
the objective SIM-o cost at 8 kHz is already in the paper and a subjective result that
contradicts it is itself worth a sentence.

**The degradation model is idealised.** Mel-domain band-limiting is not what a real
resampler or codec produces; the randomised transition slope, floor jitter and noise blunt
the mismatch rather than removing it. One sentence in the conclusion, offering noise,
reverberation and codec artefacts as the natural extension of the same joint-training
argument, is the right amount of space for this. Do not widen the motivation to cover them.

**One TTS model, one language, one seed.** The claim is about flow-matching TTS in general
but the evidence is one system, English, single run. Scope the abstract accordingly.

**n = 76 per condition.** Adequate for the effects reported -- the main results sit above
t = 10 -- but state it plainly rather than letting a reviewer discover it.

## 9. Abstract draft

> Zero-shot text-to-speech systems condition on a prompt recording, and their acoustic
> front-ends assume that prompt was captured at the model's own sampling rate. A prompt
> recorded at 8 kHz, resampled upward, arrives with an empty upper mel band that the model
> never saw in training, and synthesis quality degrades sharply. We insert an NVSR-style
> ResUNet bandwidth extender into the mel prompt of a flow-matching zero-shot TTS system
> and train it jointly with the model, so that the model's own flow-matching loss
> supervises the extender toward a prompt representation the TTS finds useful rather than
> one that merely looks plausible. On LibriTTS and on 76 unseen VCTK speakers, the
> resulting system is close to invariant to prompt bandwidth: from a 4 kHz prompt, where
> 83% of the mel band must be synthesised, it scores UTMOS 4.27 against the unmodified
> model's 2.80, and matches its own score on clean 24 kHz prompts. The trained extender
> also transfers to the unmodified released checkpoint, recovering roughly three quarters
> of the gain with no retraining. Ablations show that the bandwidth curriculum and all
> direct supervision of the extender output can be removed without measurable effect,
> reducing the objective to the flow-matching gradient and a single spectrogram
> discriminator and training 1.7x faster.

About 195 words. Trim the transfer sentence first if space is needed.
