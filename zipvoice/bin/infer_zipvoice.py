#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

"""
This script generates speech with our pre-trained ZipVoice or
    ZipVoice-Distill models. If no local model is specified,
    Required files will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.
    `zipvoice_bwe` and `zipvoice_adapter` are locally trained models that
    accept prompts at any sampling rate; pass either together with
    `--model-dir`.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.


(3) Inference with TensorRT:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav \
    --trt-engine-path models/zipvoice_distill_onnx_trt/fm_decoder.fp16.plan
"""

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter, adapter_kwargs
from zipvoice.models.zipvoice_bwe import ZipVoiceBWE
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)
from zipvoice.utils.tensorrt import load_trt

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=[
            "zipvoice",
            "zipvoice_distill",
            "zipvoice_bwe",
            "zipvoice_adapter",
        ],
        help="The model used for inference. zipvoice_bwe extends the prompt "
        "mel above the prompt's own Nyquist frequency before conditioning on "
        "it; zipvoice_adapter instead leaves the backbone untouched and adds a "
        "side branch that writes only the mel bins the prompt does not observe. "
        "Either accepts a prompt recorded at any sampling rate. Neither has a "
        "pre-trained release, so both require --model-dir.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="The model directory that contains model checkpoint, configuration "
        "file model.json, and tokens file tokens.txt. Will download pre-trained "
        "checkpoint from huggingface if not specified.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model.pt",
        help="The name of model checkpoint.",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="The vocoder checkpoint. "
        "Will download pre-trained vocoder from huggingface if not specified.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesizein the format of "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="""
        Path name of the generated wavs dir,
        used when test-list is not None
        """,
    )

    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="""
        Path name of the generated wav path,
        used when test-list is None
        """,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="The number of sampling steps.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )

    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )

    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value, set to 0 to disable normalization",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    parser.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="Number of threads to use for PyTorch on CPU.",
    )

    parser.add_argument(
        "--raw-evaluation",
        type=str2bool,
        default=False,
        help="Whether to use the 'raw' evaluation mode where provided "
        "prompts and text are fed to the model without pre-processing",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=100,
        help="Maximum duration (seconds) in a single batch, including "
        "durations of the prompt and generated wavs. You can reduce it "
        "if it causes CUDA OOM.",
    )

    parser.add_argument(
        "--remove-long-sil",
        type=str2bool,
        default=False,
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    )

    parser.add_argument(
        "--bwe-bypass",
        type=str,
        default="none",
        choices=["none", "resunet", "full"],
        help="Ablate the bandwidth extender of a zipvoice_bwe model. "
        "'none' extends the prompt mel as usual. 'resunet' re-imposes the "
        "training-time dead-band floor but does not predict into it, which "
        "measures the ResUNet alone. 'full' leaves the prompt mel untouched, "
        "which is the same input stock ZipVoice sees and so isolates the "
        "extender from the fine-tuning that trained it. Ignored by other "
        "models.",
    )

    parser.add_argument(
        "--use-speaker-cond",
        type=str2bool,
        default=None,
        help="Must match the zipvoice_adapter checkpoint being loaded: whether "
        "its branch was trained with the prompt speaker-statistics conditioning "
        "stream. None keeps whatever the model config says.",
    )

    parser.add_argument(
        "--use-rate-cond",
        type=str2bool,
        default=None,
        help="Must match the zipvoice_adapter checkpoint being loaded: whether "
        "its branch was trained with the cutoff-frequency conditioning stream.",
    )

    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="zipvoice_adapter only: scale the branch's band residual at "
        "inference. 0.0 reproduces the frozen released model, 1.0 the branch "
        "as trained. Intermediate values walk the identity / naturalness trade "
        "the branch makes, without retraining.",
    )

    parser.add_argument(
        "--residual-scale-end",
        type=float,
        default=-1.0,
        help="zipvoice_adapter only: the dose at t=1, with --residual-scale "
        "becoming the dose at t=0 and the two interpolated linearly along the "
        "solve. -1.0 keeps the flat dose. The branch's writes accumulate in "
        "the state the next step reads, so the amount by which it lands short "
        "is not constant along the trajectory and a flat correction is either "
        "too weak early or too strong late.",
    )

    parser.add_argument(
        "--residual-scale-hi",
        type=float,
        default=-1.0,
        help="zipvoice_adapter only: a separate dose above "
        "--residual-scale-split-hz, with --residual-scale applying below it. "
        "-1.0 keeps one dose for the whole band. Measured per octave, the "
        "level deficit is spread over 3-12 kHz while the identity cost of "
        "raising the dose is concentrated in the octave just above the cutoff, "
        "where the branch is continuing formant structure the prompt "
        "half-shows; splitting the dial separates the two.",
    )

    parser.add_argument(
        "--residual-scale-split-hz",
        type=float,
        default=0.0,
        help="zipvoice_adapter only: where --residual-scale-hi takes over. 0.0 "
        "means one octave above each utterance's own cutoff, which is what the "
        "rate sweep wants; a positive value pins it to a fixed frequency.",
    )

    parser.add_argument(
        "--residual-scale-top-hz",
        type=float,
        default=0.0,
        help="zipvoice_adapter only: the frequency at which "
        "--residual-scale-top applies, with the dose held flat above it. 0.0 "
        "means the top mel bin. Only meaningful together with "
        "--residual-scale-top.",
    )

    parser.add_argument(
        "--residual-scale-top",
        type=float,
        default=-1.0,
        help="zipvoice_adapter only: the raised dose at "
        "--residual-scale-top-hz, with --residual-scale-hi applying at the "
        "split and the two interpolated log-linearly in frequency and held "
        "flat beyond. -1.0 makes the raised dose a flat step. The same gain "
        "moves the top of the spectrum far more than the octave above the "
        "cutoff, so correcting an even deficit needs a dose that falls with "
        "frequency.",
    )

    parser.add_argument(
        "--decouple-residual",
        type=str2bool,
        default=False,
        help="zipvoice_adapter only: keep the branch's writes out of what the "
        "frozen backbone reads, so the backbone follows the trajectory it would "
        "have followed alone and the branch's contribution is added on top. "
        "Inference-only, and closer to how training posed the problem.",
    )

    parser.add_argument(
        "--band-fill",
        type=str2bool,
        default=False,
        help="zipvoice_adapter only: the branch emits the missing band as a mel "
        "rather than a velocity. Its output is kept out of the solver and added "
        "to the solved sample once, and it carries its own state across the "
        "steps. Set this iff the checkpoint was trained with --band-fill; on "
        "any other checkpoint it silently discards the branch.",
    )

    parser.add_argument(
        "--speaker-encoder-checkpoint",
        type=str,
        default=None,
        help="zipvoice_adapter only, and required when the branch was trained "
        "with --use-pretrained-spk 1: the pyannote WeSpeaker ResNet34 .bin the "
        "prompt is embedded with. Without it the model is handed a zero "
        "embedding flagged invalid, which is off-distribution for such a "
        "branch -- so speaker similarity measured that way is not the model's.",
    )

    parser.add_argument(
        "--trt-engine-path",
        type=str,
        default=None,
        help="The path to the TensorRT engine file.",
    )
    return parser


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def extend_prompt_band(
    model: torch.nn.Module,
    prompt_features: torch.Tensor,
    prompt_sampling_rate: int,
    sampling_rate: int,
) -> torch.Tensor:
    """
    Restore the mel band that a prompt recorded below `sampling_rate` never had.

    Resampling a 16 kHz prompt up to 24 kHz leaves everything above 8 kHz empty,
    which is a mel the model never saw in training. A ZipVoiceBWE fills it in;
    any other model is left alone. ``model.bwe_bypass`` ablates the fill.

    Args:
        model: the loaded model.
        prompt_features: the prompt mel, shape (1, T, F), already scaled by
            feat_scale.
        prompt_sampling_rate: the sampling rate of the prompt file itself,
            before resampling.
        sampling_rate: the model's sampling rate.

    Returns:
        The prompt mel, band-extended if the model supports it.
    """
    if not isinstance(model, ZipVoiceBWE) or model.bwe_bypass == "full":
        return prompt_features

    cutoff_hz = min(prompt_sampling_rate, sampling_rate) / 2
    if model.bwe_bypass == "resunet":
        logging.info(
            f"Band-limiting the prompt mel at {cutoff_hz:.0f} Hz without "
            f"running the extender (--bwe-bypass resunet)"
        )
    else:
        logging.info(
            f"Extending the prompt mel above {cutoff_hz:.0f} Hz "
            f"(prompt recorded at {prompt_sampling_rate} Hz)"
        )
    cutoff = torch.full(
        (prompt_features.size(0),),
        cutoff_hz,
        device=prompt_features.device,
        dtype=prompt_features.dtype,
    )
    restored, _ = model.restore(
        prompt_features,
        cutoff,
        randomize=False,
        use_extender=model.bwe_bypass != "resunet",
    )
    return restored


def band_sample_kwargs(
    model: torch.nn.Module,
    prompt_sampling_rate: int,
    sampling_rate: int,
    prompt_wav: Optional[torch.Tensor] = None,
    num_prompt_frames: Optional[int] = None,
    speaker_perceptual: Optional[torch.nn.Module] = None,
) -> dict:
    """
    Extra ``sample`` arguments describing where the prompt's band ends.

    ZipVoiceAdapter takes the cutoff rather than a pre-extended mel: it derives
    the band mask itself, so that the branch sees which bins are observed and,
    equivalently, which bins it is allowed to write. Every other model gets
    nothing extra and is unaffected.

    A branch trained with ``--use-pretrained-spk 1`` also reads an embedding of
    the prompt under a frozen speaker encoder. Training takes that embedding
    from the *degraded* waveform, and the prompt here is a real band-limited
    recording, so the same encoder is run on the prompt directly -- the
    ``embed_audio`` path, no vocoder involved. Leaving it out is not neutral:
    the model then gets a zero vector flagged invalid, a case training only ever
    saw for prompts too short to embed.
    """
    if not isinstance(model, ZipVoiceAdapter):
        return {}
    cutoff_hz = min(prompt_sampling_rate, sampling_rate) / 2
    logging.info(
        f"Conditioning on a prompt band-limited at {cutoff_hz:.0f} Hz "
        f"(prompt recorded at {prompt_sampling_rate} Hz)"
    )
    kwargs = {"cutoff_hz": cutoff_hz}
    if (
        model.pretrained_spk_proj is not None
        and speaker_perceptual is not None
        and prompt_wav is not None
    ):
        # Every frame of the prompt is prompt; the generated span does not exist
        # yet and contributes nothing to the embedding at inference.
        device = next(speaker_perceptual.parameters()).device
        wav = prompt_wav.to(device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        prompt_mask = torch.ones(1, num_prompt_frames, dtype=torch.bool, device=device)
        embedding, valid = speaker_perceptual.embed_audio(wav, prompt_mask)
        kwargs["speaker_embedding"] = torch.cat([embedding, valid], dim=-1)
    return kwargs


def generate_sentence_raw_evaluation(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    speaker_perceptual: Optional[torch.nn.Module] = None,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function directly feed the prompt_text, prompt_wav and text to the model.
        It is not efficient and can have poor results for some inappropriate inputs.
        (e.g., prompt wav contains long silence, text to be generated is too long)
        This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and process prompt wav
    prompt_wav, prompt_sampling_rate = load_prompt_wav(
        prompt_wav, sampling_rate=sampling_rate
    )
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features = extend_prompt_band(
        model, prompt_features, prompt_sampling_rate, sampling_rate
    )
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
        **band_sample_kwargs(
            model,
            prompt_sampling_rate,
            sampling_rate,
            prompt_wav=prompt_wav,
            num_prompt_frames=prompt_features.size(1),
            speaker_perceptual=speaker_perceptual,
        ),
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics


def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
    speaker_perceptual: Optional[torch.nn.Module] = None,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function will do the following to improve the generation quality:
        1. chunk the text according to punctuations.
        2. process chunked texts in batches.
        3. remove long silences in the prompt audio.
        4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        max_duration (float, optional): The maximum duration to process in each
            batch. Used to control memory consumption when generating long audios.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and process prompt wav
    prompt_wav, prompt_sampling_rate = load_prompt_wav(
        prompt_wav, sampling_rate=sampling_rate
    )

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (1-3 seconds is recommended)."
        )
    elif prompt_duration > 10:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features = extend_prompt_band(
        model, prompt_features, prompt_sampling_rate, sampling_rate
    )

    # Add punctuation in the end if there is not
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    # chunk text so that each len(prompt wav + generated wav) is around 25 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Batchify chunked texts for faster processing
    tokens_batches, chunked_index = batchify_tokens(
        chunked_tokens, max_duration, prompt_duration, token_duration
    )

    # Start predicting features
    chunked_features = []
    start_t = dt.datetime.now()

    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)

        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full(
            (len(batch_tokens),), prompt_features.size(1), device=device
        )

        # Generate features
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
            **band_sample_kwargs(
                model,
                prompt_sampling_rate,
                sampling_rate,
                prompt_wav=prompt_wav,
                num_prompt_frames=prompt_features.size(1),
                speaker_perceptual=speaker_perceptual,
            ),
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append((pred_features, pred_features_lens))

    # Start vocoder processing
    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()

    for pred_features, pred_features_lens in chunked_features:
        batch_wav = []
        for i in range(pred_features.size(0)):

            wav = (
                vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]])
                .squeeze(1)
                .clamp(-1, 1)
            )
            # Adjust wav volume if necessary
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    # Finish model generation
    t = (dt.datetime.now() - start_t).total_seconds()

    # Merge chunked wavs
    indexed_chunked_wavs = [
        (index, wav) for index, wav in zip(chunked_index, chunked_wavs)
    ]
    sequential_indexed_chunked_wavs = sorted(indexed_chunked_wavs, key=lambda x: x[0])
    sequential_chunked_wavs = [
        sequential_indexed_chunked_wavs[i][1]
        for i in range(len(sequential_indexed_chunked_wavs))
    ]
    final_wav = cross_fade_concat(
        sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )

    # Calculate processing time metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


def generate_list(
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    raw_evaluation: bool = False,
    max_duration: float = 100,
    remove_long_sil: bool = False,
    speaker_perceptual: Optional[torch.nn.Module] = None,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    with open(test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{res_dir}/{wav_name}.wav"

        common_params = {
            "save_path": save_path,
            "prompt_text": prompt_text,
            "prompt_wav": prompt_wav,
            "text": text,
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "device": device,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "t_shift": t_shift,
            "target_rms": target_rms,
            "feat_scale": feat_scale,
            "sampling_rate": sampling_rate,
            "speaker_perceptual": speaker_perceptual,
        }

        if raw_evaluation:
            metrics = generate_sentence_raw_evaluation(**common_params)
        else:
            metrics = generate_sentence(
                **common_params,
                max_duration=max_duration,
                remove_long_sil=remove_long_sil,
            )
        logging.info(f"[Sentence: {i}] Saved to: {save_path}")
        logging.info(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    logging.info(f"Average RTF: {np.sum(total_t) / np.sum(total_wav_seconds):.4f}")
    logging.info(
        f"Average RTF w/o vocoder: "
        f"{np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds):.4f}"
    )
    logging.info(
        f"Average RTF vocoder: "
        f"{np.sum(total_t_vocoder) / np.sum(total_wav_seconds):.4f}"
    )


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_num_threads(args.num_thread)
    torch.set_num_interop_threads(args.num_thread)

    params = AttributeDict()
    params.update(vars(args))
    fix_random_seed(params.seed)

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
        "zipvoice_bwe": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_adapter": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
    }

    model_specific_defaults = model_defaults.get(params.model_name, {})

    for param, value in model_specific_defaults.items():
        if getattr(params, param) is None:
            setattr(params, param, value)
            logging.info(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

    local_only = {
        "zipvoice_bwe": "zipvoice.bin.train_zipvoice_bwe",
        "zipvoice_adapter": "zipvoice.bin.train_zipvoice_adapter",
    }
    if params.model_name in local_only and params.model_dir is None:
        raise ValueError(
            f"{params.model_name} has no pre-trained release on Huggingface; "
            f"train it with {local_only[params.model_name]} and pass "
            f"--model-dir."
        )

    if params.model_dir is not None:
        params.model_dir = Path(params.model_dir)
        if not params.model_dir.is_dir():
            raise FileNotFoundError(f"{params.model_dir} does not exist")
        for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
            if not (params.model_dir / filename).is_file():
                raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
        model_ckpt = params.model_dir / params.checkpoint_name
        model_config = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
        logging.info(
            f"Using {params.model_name} in local model dir {params.model_dir}, "
            f"checkpoint {params.checkpoint_name}"
        )
    else:
        logging.info(f"Using pretrained {params.model_name} model from the Huggingface")
        model_ckpt = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.pt"
        )
        model_config = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json"
        )

        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt"
        )

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_bwe":
        model = ZipVoiceBWE(
            **model_config["model"],
            **model_config.get("bwe", {}),
            **tokenizer_config,
            feat_scale=params.feat_scale,
            sampling_rate=model_config["feature"]["sampling_rate"],
        )
    elif params.model_name == "zipvoice_adapter":
        model = ZipVoiceAdapter(
            **model_config["model"],
            **adapter_kwargs(
                model_config,
                use_speaker_cond=params.use_speaker_cond,
                use_rate_cond=params.use_rate_cond,
            ),
            **tokenizer_config,
            feat_scale=params.feat_scale,
            sampling_rate=model_config["feature"]["sampling_rate"],
        )
    else:
        assert params.model_name == "zipvoice_distill"
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )

    if params.model_name == "zipvoice_adapter":
        model.residual_scale = params.residual_scale
        # Negative means unset on the command line; the model wants None there.
        model.residual_scale_end = (
            None if params.residual_scale_end < 0.0 else params.residual_scale_end
        )
        model.residual_scale_hi = (
            None if params.residual_scale_hi < 0.0 else params.residual_scale_hi
        )
        model.residual_scale_split_hz = params.residual_scale_split_hz
        model.residual_scale_top_hz = params.residual_scale_top_hz
        model.residual_scale_top = (
            None if params.residual_scale_top < 0.0 else params.residual_scale_top
        )
        model.decouple_residual = params.decouple_residual
        model.band_fill = params.band_fill

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    if torch.cuda.is_available():
        params.device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        params.device = torch.device("mps")
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    if isinstance(model, ZipVoiceBWE):
        model.bwe_bypass = params.bwe_bypass
    elif params.bwe_bypass != "none":
        logging.warning(
            f"--bwe-bypass {params.bwe_bypass} has no effect on "
            f"{params.model_name}, which has no bandwidth extender."
        )

    model = model.to(params.device)
    model.eval()

    if params.trt_engine_path:
        load_trt(model, params.trt_engine_path)

    # The branch's pretrained speaker stream, if the checkpoint has one. It is
    # not a submodule of the model -- training owns it the same way -- so it is
    # built here and handed to the generation functions.
    speaker_perceptual = None
    if isinstance(model, ZipVoiceAdapter) and model.pretrained_spk_proj is not None:
        if params.speaker_encoder_checkpoint:
            from zipvoice.models.modules.perceptual import SpeakerPerceptual

            speaker_perceptual = (
                SpeakerPerceptual(
                    speaker_checkpoint=params.speaker_encoder_checkpoint,
                    feat_scale=params.feat_scale,
                )
                .to(params.device)
                .eval()
            )
            logging.info(
                "Conditioning the branch on the prompt's speaker embedding "
                f"({speaker_perceptual.speaker_dim}-dim)"
            )
        else:
            logging.warning(
                "This checkpoint's branch was trained with a pretrained speaker "
                "stream, but --speaker-encoder-checkpoint was not given. The "
                "model will be handed a zero embedding flagged invalid, which "
                "training only ever saw for unusably short prompts; speaker "
                "similarity measured this way understates the model."
            )

    vocoder = get_vocoder(params.vocoder_path)
    vocoder = vocoder.to(params.device)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params.sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    if params.test_list:
        res_dir = params.res_dir
        os.makedirs(res_dir, exist_ok=True)
        generate_list(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            raw_evaluation=params.raw_evaluation,
            max_duration=params.max_duration,
            remove_long_sil=params.remove_long_sil,
            speaker_perceptual=speaker_perceptual,
        )
    else:
        assert (
            not params.raw_evaluation
        ), "Raw evaluation is only valid with --test-list"
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            max_duration=params.max_duration,
            remove_long_sil=params.remove_long_sil,
            speaker_perceptual=speaker_perceptual,
        )
        logging.info(f"Saved to: {params.res_wav_path}")
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
