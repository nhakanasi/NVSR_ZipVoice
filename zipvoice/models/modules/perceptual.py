# Copyright    2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speaker-identity and deep-feature losses measured on vocoded audio.

The band adapter writes the mel bins a band-limited prompt does not observe.
Neither of its two training losses can say whose voice those bins belong to:
flow matching scores the frozen backbone's velocity, and an L1 against the
target mel is minimised by whatever high band is closest on average, which is
nobody's in particular. Measured on LibriTTS that cost 0.122 SIM-o at a 4 kHz
cutoff against the released model, while raising UTMOS -- a plausible high band
attached to the wrong speaker.

This module supplies the missing gradient. The endpoint estimate and the target
are vocoded by the same vocoder and embedded by the same frozen speaker encoder,
and the branch is charged for the distance between them: a cosine distance on
the utterance embedding, and an L1 between the encoder's intermediate
activations. Vocoding both sides rather than comparing against the original
recording is deliberate -- it removes the vocoder's own artifacts from the
comparison, leaving the model's mel error as the only thing the loss can see.

The speaker encoder here is WeSpeaker ResNet34. The SIM-o reported in the
results is a WavLM-based ECAPA-TDNN, a different architecture trained on
different data, so an improvement in the reported metric is transfer rather than
the objective scoring itself. That separation only holds as long as the two stay
distinct; do not point this at the evaluation checkpoint.

Nothing here is trainable and nothing here belongs to the model. The training
script owns the instance, so a saved checkpoint stays exactly the released
tensors plus the branch, and remains mergeable back onto the released model.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio

# Vocos analyses at 24 kHz with a 256-sample hop, matching VocosFbank in
# zipvoice/utils/feature.py. Frame indices are converted to sample indices with
# this, so the two have to agree.
HOP_LENGTH = 256
VOCOS_RATE = 24000

# The speaker encoder's rate. Resampling to it discards everything above 8 kHz,
# so at a 4 kHz cutoff this loss never sees the top octave the branch invents.
# That is a real limit, and an aligned one: the SIM-o evaluator resamples to the
# same rate and cannot see that band either.
SPEAKER_RATE = 16000

# Below roughly a quarter second the encoder has too few frames to pool over and
# the embedding stops meaning anything. Shorter spans are skipped rather than
# contributing noise.
MIN_SPEAKER_SAMPLES = 4000


class SpeakerPerceptual(nn.Module):
    """Frozen vocoder and speaker encoder, used only to produce losses.

    Args:
      speaker_checkpoint: path to the pyannote WeSpeaker ResNet34 ``.bin``.
      vocos_local_path: directory holding ``config.yaml`` and
        ``pytorch_model.bin``. None downloads ``charactr/vocos-mel-24khz``.
      feat_scale: the scale training features carry, undone before vocoding.
      fmatch_layers: which ResNet stages to match activations on.
    """

    def __init__(
        self,
        speaker_checkpoint: str,
        vocos_local_path: Optional[str] = None,
        feat_scale: float = 0.1,
        fmatch_layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
    ):
        super().__init__()
        from pyannote.audio import Model as PyannoteModel

        self.feat_scale = feat_scale
        # Built on first use, not here. The vocoder exists only to turn a mel
        # back into a waveform for the speaker encoder, and the waveform
        # degradation path has the real band-limited recording in hand, so
        # ``embed_audio`` never needs one -- neither does a run with the
        # identity and feature-matching weights at zero. Constructing it
        # eagerly would make every such run reach for
        # ``charactr/vocos-mel-24khz``, which is the only thing standing
        # between adapter training and an offline machine.
        self._vocoder: Optional[nn.Module] = None
        self._vocos_local_path = vocos_local_path

        speaker = PyannoteModel.from_pretrained(speaker_checkpoint)
        self.speaker = speaker.eval()

        self.resample = torchaudio.transforms.Resample(VOCOS_RATE, SPEAKER_RATE)

        self.requires_grad_(False)

        # The embedding width, probed rather than assumed. It sizes the branch's
        # projection, and ``embed_prompt`` needs it to return a correctly shaped
        # zero when every prompt in a batch is too short. Probed here, before the
        # hooks below exist, so the throwaway forward captures nothing.
        with torch.no_grad():
            self.speaker_dim = int(
                self.speaker(torch.zeros(1, 1, SPEAKER_RATE)).reshape(-1).numel()
            )

        # Activations are captured by hook rather than by reimplementing the
        # encoder's forward, so this keeps working if the checkpoint's internals
        # differ from what is assumed here.
        self._captured: List[torch.Tensor] = []
        # The hooks below stay registered for the life of the module, but only
        # ``forward`` wants what they collect. See ``_capture``.
        self._capturing = False
        self._hooks = []
        resnet = self.speaker.resnet
        for name in fmatch_layers:
            module = getattr(resnet, name, None)
            if module is None:
                logging.warning(f"Speaker encoder has no layer {name}; not matched")
                continue
            self._hooks.append(module.register_forward_hook(self._capture))
        assert self._hooks, "No feature matching layers found on the speaker encoder"

    def _capture(self, module, inputs, output) -> None:
        # Guarded, because the hooks fire on every forward of the speaker
        # encoder and only ``forward`` consumes them. ``embed_audio`` and
        # ``embed_prompt`` run that encoder once per batch for the pretrained
        # speaker stream, whatever the feature-matching weight is, so without
        # this an adapter run that never computes a perceptual loss still
        # appended four ResNet34 activations per utterance to a list nothing
        # cleared -- about 17 MB a batch, which reached 29 GiB and killed the
        # run some fifteen hundred batches later.
        if self._capturing:
            self._captured.append(output)

    def train(self, mode: bool = True) -> "SpeakerPerceptual":
        # Frozen in both senses: this never enters training mode, even when the
        # model around it does.
        return super().train(False)

    @property
    def vocoder(self) -> nn.Module:
        """The mel vocoder, built on first access.

        It is frozen and moved to whatever device the speaker encoder is on, so
        a lazily-built one behaves like the eagerly-built one it replaced even
        though it missed the ``.to(device)`` the caller did at construction.
        """
        if self._vocoder is None:
            from vocos import Vocos

            if self._vocos_local_path:
                vocoder = Vocos.from_hparams(f"{self._vocos_local_path}/config.yaml")
                state = torch.load(
                    f"{self._vocos_local_path}/pytorch_model.bin",
                    map_location="cpu",
                    weights_only=True,
                )
                vocoder.load_state_dict(state, strict=False)
            else:
                vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            device = next(self.speaker.parameters()).device
            self._vocoder = vocoder.eval().requires_grad_(False).to(device)
        return self._vocoder

    def _vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """(1, T, F) scaled mel -> (1, samples) waveform at 16 kHz.

        ``Vocos.decode`` is decorated ``@torch.inference_mode()``, which silently
        detaches the result; its two submodules are called directly so that the
        gradient survives.
        """
        mel = mel.permute(0, 2, 1) / self.feat_scale  # (1, F, T)
        audio = self.vocoder.head(self.vocoder.backbone(mel))
        return self.resample(audio)

    @torch.no_grad()
    def embed_prompt(
        self, condition: torch.Tensor, prompt_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Speaker embedding of the prompt, for conditioning rather than loss.

        The identity loss above could only ever charge the branch for identity
        it had already failed to produce; it never told the branch *whose* band
        to write, and measured as a clean null. This supplies that instead: the
        prompt's embedding under a pretrained speaker encoder, as an input.

        Args:
          condition: the band-limited mel, (B, T, F). This must be the masked
            tensor, never the clean features -- see the leak note below.
          prompt_mask: True at frames carrying prompt, (B, T).

        Returns:
          The embeddings, (B, D), and a validity flag, (B, 1), that is zero for
          an utterance whose prompt was too short to embed. The flag is not
          decoration: ``condition_time_mask`` draws the training prompt as
          0-30% of the utterance and regularly leaves it empty, and without the
          flag a zero embedding is indistinguishable from a speaker the encoder
          happens to place near the origin.

        On leakage. Three things keep this honest, and all three have to hold:
        the mel is the band-limited one, so nothing above the cutoff is in the
        input; only prompt frames are read, never the span being generated; and
        the encoder runs at 16 kHz, so it is deaf above 8 kHz regardless. The
        prompt is vocoded here rather than read from disk so that training and
        inference embed the same kind of signal -- at inference the prompt is a
        real band-limited recording, and embedding that directly would put a
        vocoder's worth of difference between the two.
        """
        device = condition.device
        embeddings, valid = [], []
        min_frames = MIN_SPEAKER_SAMPLES * (VOCOS_RATE // SPEAKER_RATE) // HOP_LENGTH

        # fp32, not merely fp32 inputs. This is called from inside the model's
        # forward, so it inherits the training autocast, and Vocos's ISTFT
        # overflows in half precision -- torch says so itself ("ComplexHalf
        # support is experimental"). An inf embedding then reaches the branch's
        # projection and the whole loss goes NaN within tens of steps.
        with torch.amp.autocast(device.type, enabled=False):
            for i in range(condition.size(0)):
                idx = prompt_mask[i].nonzero()
                if idx.numel() == 0 or (int(idx[-1]) - int(idx[0]) + 1) < min_frames:
                    embeddings.append(None)
                    valid.append(0.0)
                    continue
                begin, stop = int(idx[0]), int(idx[-1]) + 1
                wav = self._vocode(condition[i : i + 1, begin:stop].float())
                emb = self.speaker(wav.unsqueeze(1)).reshape(-1)
                # A prompt can still vocode to something the encoder cannot
                # handle; such an utterance is dropped rather than allowed to
                # poison the batch, using the same flag as a too-short prompt.
                if not torch.isfinite(emb).all():
                    embeddings.append(None)
                    valid.append(0.0)
                    continue
                embeddings.append(emb)
                valid.append(1.0)

        dim = next((e.numel() for e in embeddings if e is not None), None)
        if dim is None:
            return (
                torch.zeros(condition.size(0), self.speaker_dim, device=device),
                torch.zeros(condition.size(0), 1, device=device),
            )
        stacked = torch.stack(
            [e if e is not None else torch.zeros(dim, device=device) for e in embeddings]
        )
        return stacked, torch.tensor(valid, device=device).unsqueeze(-1)

    @torch.no_grad()
    def embed_audio(
        self, audio: torch.Tensor, prompt_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Speaker embedding of the prompt, read from the waveform directly.

        ``embed_prompt`` below vocodes the band-limited mel because that is all
        it has. When the degradation is done by really resampling the waveform,
        the band-limited signal itself is in hand, and this reads that instead:
        no vocoder in the conditioning path, and the encoder sees the same kind
        of signal it will see at inference, where the prompt is a real
        band-limited recording rather than a rendering of its mel.

        Args:
          audio: the degraded waveform at ``VOCOS_RATE``, of shape ``(B, S)``.
          prompt_mask: True at frames carrying prompt, of shape ``(B, T)``.

        Returns:
          The embeddings, ``(B, D)``, and a validity flag, ``(B, 1)``, zero for
          an utterance whose prompt was too short to embed -- see
          ``embed_prompt`` for why the flag is not decoration.
        """
        device = audio.device
        embeddings, valid = [], []

        with torch.amp.autocast(device.type, enabled=False):
            wavs = self.resample(audio.float())
            scale = SPEAKER_RATE / VOCOS_RATE
            for i in range(audio.size(0)):
                idx = prompt_mask[i].nonzero()
                if idx.numel() == 0:
                    embeddings.append(None)
                    valid.append(0.0)
                    continue
                begin = int(int(idx[0]) * HOP_LENGTH * scale)
                stop = int((int(idx[-1]) + 1) * HOP_LENGTH * scale)
                span = wavs[i : i + 1, begin:stop]
                if span.size(-1) < MIN_SPEAKER_SAMPLES:
                    embeddings.append(None)
                    valid.append(0.0)
                    continue
                emb = self.speaker(span.unsqueeze(1)).reshape(-1)
                if not torch.isfinite(emb).all():
                    embeddings.append(None)
                    valid.append(0.0)
                    continue
                embeddings.append(emb)
                valid.append(1.0)

        dim = next((e.numel() for e in embeddings if e is not None), None)
        if dim is None:
            return (
                torch.zeros(audio.size(0), self.speaker_dim, device=device),
                torch.zeros(audio.size(0), 1, device=device),
            )
        stacked = torch.stack(
            [e if e is not None else torch.zeros(dim, device=device) for e in embeddings]
        )
        return stacked, torch.tensor(valid, device=device).unsqueeze(-1)

    def forward(
        self,
        x1_hat: torch.Tensor,
        features: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Losses over the generated span of each utterance handed in.

        Args:
          x1_hat: the endpoint estimate, (B, T, F), already restricted to the
            utterances that are to contribute.
          features: the target mel, same shape.
          starts, ends: the generated span of each utterance, in frames.

        Returns:
          The identity loss, the feature matching loss, and how many utterances
          actually contributed.
        """
        device = x1_hat.device
        zero = torch.zeros((), device=device)
        identity_terms: List[torch.Tensor] = []
        fmatch_terms: List[torch.Tensor] = []

        for i in range(x1_hat.size(0)):
            begin, stop = int(starts[i]), int(ends[i])
            if (stop - begin) * HOP_LENGTH < MIN_SPEAKER_SAMPLES * (
                VOCOS_RATE // SPEAKER_RATE
            ):
                continue
            # fp32 throughout: backprop through the vocoder's ISTFT overflows
            # readily in half precision, and this runs under --use-fp16.
            gen = x1_hat[i : i + 1, begin:stop].float()
            ref = features[i : i + 1, begin:stop].float()

            self._captured = []
            self._capturing = True
            try:
                gen_wav = self._vocode(gen)
                gen_emb = self.speaker(gen_wav.unsqueeze(1))
                gen_feats = self._captured

                self._captured = []
                with torch.no_grad():
                    ref_wav = self._vocode(ref)
                    ref_emb = self.speaker(ref_wav.unsqueeze(1))
                    ref_feats = self._captured
            finally:
                self._capturing = False

            identity_terms.append(
                1.0 - torch.nn.functional.cosine_similarity(gen_emb, ref_emb, dim=-1)
            )

            # Each stage is normalised by the reference activation's own scale,
            # so a layer does not dominate merely by being larger.
            layer_terms = []
            for a, b in zip(gen_feats, ref_feats):
                if a.shape != b.shape:
                    continue
                layer_terms.append((a - b).abs().mean() / b.abs().mean().clamp(min=1e-5))
            if layer_terms:
                fmatch_terms.append(torch.stack(layer_terms).mean())

        self._captured = []
        if not identity_terms:
            return zero, zero, 0
        identity = torch.cat(identity_terms).mean()
        fmatch = torch.stack(fmatch_terms).mean() if fmatch_terms else zero
        return identity, fmatch, len(identity_terms)
