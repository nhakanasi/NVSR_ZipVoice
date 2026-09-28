#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang,
#                                                       Han Zhu)
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
This script trains the high-band branch of a dual-diffusion ZipVoice.

The backbone is a released ZipVoice checkpoint and is frozen: it is run under
no_grad and in eval mode to produce the trajectory and hidden state the branch
cross-attends to. Only the branch and its conditioning encoders are optimized,
so --checkpoint is required rather than optional.

The branch predicts the clean endpoint directly. Its masked endpoint MSE is
evaluated partly on teacher-forced flow states and partly on detached rollout
states produced by the same coupled solver used at inference. An optional L1
term and adversarial stage can be added; see zipvoice.models.zipvoice_dual.

Usage:

python3 -m zipvoice.bin.train_zipvoice_dual \
    --world-size 8 \
    --use-fp16 1 \
    --num-epochs 11 \
    --max-duration 500 \
    --lr-hours 30000 \
    --model-config conf/zipvoice_dual.json \
    --checkpoint exp/zipvoice/epoch-11-avg-4.pt \
    --tokenizer libritts \
    --token-file "data/tokens_libritts.txt" \
    --dataset libritts \
    --manifest-dir data/fbank \
    --exp-dir exp/zipvoice_dual
"""

import argparse
import copy
import json
import logging
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut, CutSet
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

import zipvoice.utils.diagnostics as diagnostics
from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.modules.bwe_discriminators import (
    BweDiscriminators,
    feature_matching_loss,
    hinge_d_loss,
    hinge_g_loss,
)
from zipvoice.models.zipvoice_dual import ZipVoiceDual
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import (
    remove_checkpoints,
    resume_checkpoint,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from zipvoice.utils.common import (
    AttributeDict,
    GradScaler,
    MetricsTracker,
    cleanup_dist,
    create_grad_scaler,
    get_adjusted_batch_count,
    get_env_info,
    get_parameter_groups_with_lrs,
    prepare_input,
    set_batch_count,
    setup_dist,
    setup_logger,
    str2bool,
    torch_autocast,
)
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.hooks import register_inf_check_hooks
from zipvoice.utils.lr_scheduler import (
    CosineDecayLRScheduler,
    Eden,
    FixedLRScheduler,
    LRScheduler,
)
from zipvoice.utils.optim import ScaledAdam

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12356,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=11,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--num-iters",
        type=int,
        default=0,
        help="Number of iter to train, will ignore num_epochs if > 0.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="""Checkpoints of pre-trained models, will load it if not None
        """,
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="""Resume a run from a mid-epoch `checkpoint-{batch}.pt`, which no
        epoch number can name. Restores the branch, the optimizer, the
        discriminators and `batch_idx_train`, so training continues from the
        step the file records rather than from scratch.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/zipvoice",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.02, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--lr-hours",
        type=float,
        default=0,
        help="""If positive, --epoch is ignored and it specifies the number of hours
        that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=50,
        help="""Reference batch duration for purposes of adjusting batch counts for"
        setting various schedules inside the model".
        """,
    )

    parser.add_argument(
        "--finetune",
        type=str2bool,
        default=False,
        help="Whether to use the fine-tuning mode, will used a fixed learning rate "
        "schedule and skip the large dropout phase.",
    )

    parser.add_argument(
        "--lr-decay-start",
        type=int,
        default=-1,
        help="""Fine-tuning only: hold the learning rate until this batch
        index, then decay it with a cosine to --lr-final-ratio of the base rate
        by --num-iters. Checkpoints taken at a fixed rate are snapshots of a
        run that is still moving; the decay lets it settle. A negative value
        keeps the rate fixed for the whole run.""",
    )

    parser.add_argument(
        "--lr-final-ratio",
        type=float,
        default=0.1,
        help="Learning rate at --num-iters, relative to --base-lr, when decaying.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--scan-oom",
        type=str2bool,
        default=False,
        help="Scan pessimistic batches to see whether they cause OOMs.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=5000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--valid-by-epoch",
        type=str2bool,
        default=False,
        help="""Whether to validate after each epoch. If False, will validate
        after every save_every_n iterations.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    # The backbone is frozen and always fully conditioned here, so stock
    # training's --condition-drop-ratio has nothing to act on and is gone.

    parser.add_argument(
        "--fm-space",
        type=str,
        default="endpoint",
        choices=["endpoint", "velocity"],
        help="""Where the branch's MSE is measured. "endpoint" compares the
        predicted and clean endpoints, which is the velocity error weighted by
        (1-t)^2 and so barely trains the late steps. "velocity" compares the
        implied velocity (x1_hat - y)/(1-t) with the one that reaches the clean
        endpoint, (x1 - y)/(1-t): uniform in t, as the backbone was trained.
        Requires --t-max below 1.""",
    )

    parser.add_argument(
        "--t-max",
        type=float,
        default=1.0,
        help="""Training flow times are drawn uniformly from [0, t-max), and
        validation's fixed grid is scaled to the same range. Below 1 it keeps
        the velocity-space loss's 1/(1-t)^2 factor bounded without clamping
        it, which would make the weighting non-uniform near t-max.""",
    )

    parser.add_argument(
        "--recon-loss-weight",
        type=float,
        default=0.0,
        help="""Weight of the L1 term on the endpoint estimate, added to the
        endpoint MSE. This term is minimized by the per-bin conditional
        median, which in log-mel sits 9-10 dB below the energy of a real
        sample, so a large weight reproduces the adapter arm's hedge; the
        adversarial stage is what removes it. 0 disables the term.""",
    )

    parser.add_argument(
        "--band-loss-weight",
        type=float,
        default=0.0,
        help="""Weight of the band-level term: per utterance and per sub-band
        of the missing range, the L1 distance between the endpoint estimate's
        and the target's band energy, both its mean over the generated frames
        (level and tilt) and its standard deviation over them (how far it
        follows the voice). These are near-deterministic given the utterance,
        so unlike the per-bin terms their optimum is not a hedge below the
        real level. 0 disables the term.""",
    )

    parser.add_argument(
        "--band-t-min",
        type=float,
        default=0.5,
        help="""Only utterances whose flow time is at least this enter the
        band-level term. Below it the endpoint estimate is largely noise, and
        its band statistics say nothing about the band.""",
    )

    parser.add_argument(
        "--solve-loss-weight",
        type=float,
        default=0.0,
        help="""Weight of the band level, spread and envelope terms on an
        evaluation-matched solve: --solve-utts utterances per batch are solved
        from pure noise as inference solves them (16 steps, t_shift 0.5, the
        backbone under guidance 1.0), with gradients through the final branch
        call, whose endpoint is the output's high band. 0 disables it.""",
    )

    parser.add_argument(
        "--solve-utts",
        type=int,
        default=2,
        help="Utterances per batch that get the evaluation-matched solve.",
    )

    parser.add_argument(
        "--rollout-prob",
        type=float,
        default=0.5,
        help="""Per-utterance probability of training at a detached state
        generated by the branch and frozen backbone instead of at the exact
        teacher-forced flow marginal. 0 disables rollout training.""",
    )

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2,
        help="""Number of no-gradient Euler steps used to construct a rollout
        state from pure noise at t=0 to the supervised time. More steps improve
        fidelity but add two frozen-backbone forwards (velocity on the stock
        state, memory on the branch's) and one branch forward each.""",
    )

    parser.add_argument(
        "--rollout-start-step",
        type=int,
        default=0,
        help="""Rollout is off before this step. A fresh branch's rollouts are
        noise, and training on them teaches nothing.""",
    )

    parser.add_argument(
        "--rollout-ramp-steps",
        type=int,
        default=0,
        help="""After --rollout-start-step, --rollout-prob is reached linearly
        over this many steps. 0 switches it on at once.""",
    )

    parser.add_argument(
        "--resample-degrade",
        type=str2bool,
        default=False,
        help="""Band-limit the prompt by resampling the waveform, as the
        evaluation prompts are, instead of by flooring the mel. The mel floor
        sits 9-13 log-units below a real 24k->8k->24k file above the cutoff,
        so a branch conditioned on it meets a different prompt at evaluation.
        Needs the waveform, so it turns --return-audio on.""",
    )

    # Stage 2. Off unless a start step is given, so stage 1 is exactly the
    # recipe above and the discriminators are not even constructed.
    #
    # What they are for: both stage-1 terms can be satisfied by a band that is
    # merely plausible on average, and the L1 term is satisfied best by one
    # that is not plausible at all, the conditional median. A discriminator
    # scores whether *this* band is a plausible continuation of *this* observed
    # low band, which is the first term in the objective that a speaker-generic
    # band loses on.
    parser.add_argument(
        "--gan-start-step",
        type=int,
        default=-1,
        help="""Enable the discriminators once batch_idx_train reaches this
        value. A negative value disables them for the whole run, which is the
        default and leaves the objective as stage 1 alone.""",
    )

    parser.add_argument(
        "--adv-loss-weight",
        type=float,
        default=0.1,
        help="Weight of each discriminator view's hinge adversarial loss.",
    )

    parser.add_argument(
        "--feat-match-loss-weight",
        type=float,
        default=2.0,
        help="Weight of each discriminator view's feature-matching loss.",
    )

    parser.add_argument(
        "--disc-lr",
        type=float,
        default=2e-4,
        help="""AdamW learning rate of the discriminators. Plain AdamW rather
        than ScaledAdam: the per-tensor scale clipping is a poor fit for a
        discriminator, and these parameters never enter the saved model.""",
    )

    parser.add_argument(
        "--gan-t-min",
        type=float,
        default=0.5,
        help="""Only utterances whose flow time is at least this are
        discriminated. Below it the endpoint estimate is largely noise, and a
        discriminator trained to call noise fake learns the noise level rather
        than the band. Only used with --gan-source endpoint.""",
    )

    parser.add_argument(
        "--gan-source",
        type=str,
        default="endpoint",
        choices=["endpoint", "solve"],
        help="""What the discriminators judge as fake. "endpoint" is the
        branch's one-shot endpoint estimate at the training state, above
        --gan-t-min. "solve" is the evaluation-matched solve of the first
        --solve-utts utterances, the mel inference produces, with the target's
        low band put under it so the real and the fake differ only in the band
        the branch generates. It shares the solve --solve-loss-weight pays for;
        keep that weight above zero so the band level stays anchored while the
        discriminators shape its texture.""",
    )

    parser.add_argument(
        "--cutoff-curriculum-steps",
        type=int,
        default=30000,
        help="""Number of steps over which the sampled prompt cutoff walks from
        easy (low cutoff, little to generate) to the full range. After this
        many steps the cutoff is drawn from the full range.""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "custom"],
        help="The used training dataset",
    )

    parser.add_argument(
        "--train-manifest",
        type=str,
        help="Path of the training manifest",
    )

    parser.add_argument(
        "--dev-manifest",
        type=str,
        help="Path of the validation manifest",
    )

    parser.add_argument(
        "--min-len",
        type=float,
        default=1.0,
        help="The minimum audio length used for training",
    )

    parser.add_argument(
        "--max-len",
        type=float,
        default=30.0,
        help="The maximum audio length used for training",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/zipvoice_base.json",
        help="The model configuration file.",
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
        "--token-file",
        type=str,
        default="data/tokens_emilia.txt",
        help="The file that contains information that maps tokens to ids,"
        "which is a text file with '{token}\t{token_id}' per line.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - env_info:  A dict containing information about the environment.

    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "env_info": get_env_info(),
        }
    )

    return params


def gan_enabled(params: AttributeDict) -> bool:
    """Whether the discriminators take part at the current training step."""
    return params.gan_start_step >= 0 and params.batch_idx_train >= (
        params.gan_start_step
    )


def discriminator_states(
    discriminators: Optional[Union[nn.Module, DDP]],
    disc_optimizer: Optional[Optimizer],
) -> Optional[dict]:
    """The discriminator state to store in a checkpoint, so that a resumed run
    picks the adversarial game back up rather than restarting it against a
    branch that has moved on."""
    if discriminators is None:
        return None
    module = (
        discriminators.module if isinstance(discriminators, DDP) else discriminators
    )
    return {
        "discriminators": module.state_dict(),
        "disc_optimizer": (
            disc_optimizer.state_dict() if disc_optimizer is not None else None
        ),
    }


def run_discriminators(
    discriminators: Union[nn.Module, DDP],
    mel: Tensor,
    ctx: dict,
) -> Tuple[dict, dict]:
    """Run every discriminator view on one mel, with the shared conditioning.

    Frames inside the prompt and inside the padding are zeroed, on the real and
    the fake mel alike: the prompt is ground truth in both, so leaving it in
    would let a discriminator score the half of the sample that is never
    generated.
    """
    return discriminators(
        mel=mel * ctx["frame_mask"],
        mask=ctx["band_mask"],
        cutoff_hz=ctx["cutoff_hz"],
        centers=ctx["centers"],
        f0=None,
        voiced=None,
    )


def gan_context(
    params: AttributeDict,
    terms: dict,
    features: Tensor,
    num_frames: int,
    centers: Tensor,
) -> Optional[dict]:
    """The subset of the batch the discriminators see, or None if it is empty.

    Only utterances far enough along the flow qualify, so the tensors are
    sliced rather than masked: a discriminator has no per-utterance weight, and
    feeding it near-noise endpoints at a weight of zero is not available.

    With --gan-source solve the subset is the solved utterances instead, and
    the fake is their solved high band over the target's own low band. The
    solve's low band is the backbone's, which the branch cannot change; judging
    it would spend the discriminators on a difference no gradient can close.
    """
    # The model reports which frames it had to generate as (B, T) bools; the
    # discriminators multiply a (B, T, F) mel by it.
    frame_mask = terms["frame_mask"].to(features.dtype).unsqueeze(-1)
    if params.gan_source == "solve":
        if "solved_x1" not in terms:
            return None
        k = terms["solved_x1"].size(0)
        real = features[:k]
        band_mask = terms["band_mask"][:k]
        return {
            "features": real,
            "x1_hat": real + (terms["solved_x1"] - real) * band_mask,
            "band_mask": band_mask,
            "cutoff_hz": terms["cutoff_hz"][:k],
            "frame_mask": frame_mask[:k],
            "centers": centers,
            "num_frames": num_frames,
        }

    keep =(terms["t"].flatten() >= params.gan_t_min).nonzero().flatten()
    if keep.numel() == 0:
        return None
    return {
        "features": features[keep],
        "x1_hat": terms["x1_hat"][keep],
        "band_mask": terms["band_mask"][keep],
        "cutoff_hz": terms["cutoff_hz"][keep],
        "frame_mask": frame_mask[keep],
        "centers": centers,
        "num_frames": num_frames,
    }


# Sub-band edges of the band-level term, in Hz. Each band only counts the bins
# a given utterance is missing, so the bands below its cutoff drop out.
BAND_EDGES_HZ = (2000.0, 3000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0)


def band_level_loss(
    params: AttributeDict,
    terms: dict,
    features: Tensor,
    centers: Tensor,
) -> Tensor:
    """L1 between the endpoint estimate's and the target's per-band energy
    statistics over the generated frames: the mean, which carries level and
    spectral tilt, and the standard deviation, which carries how far the band
    rises and falls with the voice.

    A band's energy in one frame is the mask-weighted mean of its log-mel bins.
    Pairs of (utterance, band) with no missing bin or no generated frame, and
    utterances below --band-t-min, count for nothing.
    """
    keep = (terms["t"].view(-1) >= params.band_t_min).to(features.dtype)  # (B,)
    frames = terms["frame_mask"].to(features.dtype)  # (B, T)
    num_frames = frames.sum(dim=1).clamp(min=1.0)  # (B,)
    mask = terms["band_mask"].squeeze(1).to(features.dtype)  # (B, F)
    # Statistics in float32: fp16 variance over long utterances is too coarse.
    pred = terms["x1_hat"].float()
    target = features.float()

    edges = torch.tensor(BAND_EDGES_HZ, device=centers.device, dtype=centers.dtype)
    loss = features.new_zeros((), dtype=torch.float32)
    count = features.new_zeros((), dtype=torch.float32)
    for low, high in zip(edges[:-1], edges[1:]):
        in_band = ((centers >= low) & (centers < high)).to(mask.dtype)  # (F,)
        weight = (mask * in_band).float()  # (B, F)
        weight_sum = weight.sum(dim=1)  # (B,)
        valid = keep * (weight_sum > 0).to(keep.dtype) * (frames.sum(dim=1) > 0)
        if valid.sum() == 0:
            continue
        norm = weight_sum.clamp(min=1e-6).unsqueeze(1)  # (B, 1)
        stats = []
        for mel in (pred, target):
            energy = (mel * weight.unsqueeze(1)).sum(dim=-1) / norm  # (B, T)
            mean = (energy * frames).sum(dim=1) / num_frames
            var = (((energy - mean.unsqueeze(1)) ** 2) * frames).sum(
                dim=1
            ) / num_frames
            stats.append((mean, (var + 1e-6).sqrt()))
        (pred_mean, pred_std), (target_mean, target_std) = stats
        per_utt = (pred_mean - target_mean).abs() + (pred_std - target_std).abs()
        loss = loss + (per_utt * valid).sum()
        count = count + valid.sum()
    return loss / count.clamp(min=1.0)


def band_envelope_loss(
    terms: dict,
    features: Tensor,
    centers: Tensor,
    kernel: int = 5,
) -> Tensor:
    """L1 between the predicted and target per-band energy traces over time,
    each smoothed over ``kernel`` frames, on the generated frames.

    The band-level term only constrains each band's mean and spread over the
    utterance, which a constant layer of noise or a horizontal ridge can meet.
    This one asks the band to rise and fall where the target's does. The traces
    are smoothed because the band's frame-to-frame detail is not predictable
    from the low band, and an L1 on it would regress to the quiet median again.
    """
    frames = terms["frame_mask"].float()  # (B, T)
    mask = terms["band_mask"].squeeze(1).float()  # (B, F)
    pred = terms["x1_hat"].float()
    target = features.float()

    edges = torch.tensor(BAND_EDGES_HZ, device=centers.device, dtype=centers.dtype)
    loss = pred.new_zeros(())
    count = pred.new_zeros(())
    for low, high in zip(edges[:-1], edges[1:]):
        in_band = ((centers >= low) & (centers < high)).float()  # (F,)
        weight = mask * in_band  # (B, F)
        weight_sum = weight.sum(dim=1)  # (B,)
        valid = (weight_sum > 0).float()
        if valid.sum() == 0:
            continue
        norm = weight_sum.clamp(min=1e-6).unsqueeze(1)
        traces = [
            nn.functional.avg_pool1d(
                ((mel * weight.unsqueeze(1)).sum(dim=-1) / norm).unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                count_include_pad=False,
            ).squeeze(1)
            for mel in (pred, target)
        ]  # (B, T) each
        error = ((traces[0] - traces[1]).abs() * frames).sum(dim=1)
        error = error / frames.sum(dim=1).clamp(min=1.0)
        loss = loss + (error * valid).sum()
        count = count + valid.sum()
    return loss / count.clamp(min=1.0)


def solve_output_loss(
    params: AttributeDict,
    terms: dict,
    features: Tensor,
    centers: Tensor,
) -> Tensor:
    """Band level, spread and envelope of the evaluation-matched solve.

    ``terms["solved_x1"]`` is the mel inference would produce for the first few
    utterances, so these terms judge what is scored rather than a one-shot
    endpoint estimate from a training state.
    """
    k = terms["solved_x1"].size(0)
    solved = {
        "x1_hat": terms["solved_x1"],
        "t": torch.ones(k, device=features.device),  # past any --band-t-min
        "frame_mask": terms["frame_mask"][:k],
        "band_mask": terms["band_mask"][:k],
    }
    return band_level_loss(
        params, solved, features[:k], centers
    ) + band_envelope_loss(solved, features[:k], centers)


def compute_disc_loss(
    params: AttributeDict,
    discriminators: Union[nn.Module, DDP],
    ctx: dict,
) -> Tuple[Tensor, MetricsTracker]:
    """Hinge discriminator loss: the target mel against the branch's endpoint
    estimate, detached so no gradient reaches the branch.

    Real and fake go through in a single batched forward. DDP's reducer only
    tolerates one forward per backward, and none of the discriminator stacks
    carry batch statistics (they are weight-normalized), so batching the two
    together is numerically the same as two separate passes.
    """
    real = ctx["features"]
    fake = ctx["x1_hat"].detach()
    batch_size = real.size(0)

    paired = dict(ctx)
    for key in ("band_mask", "cutoff_hz", "frame_mask"):
        paired[key] = torch.cat([ctx[key], ctx[key]], dim=0)
    paired_logits, _ = run_discriminators(
        discriminators, torch.cat([real, fake], dim=0), paired
    )

    # "frames" is deliberately not set: this tracker is added to the branch's,
    # which already carries the frame count MetricsTracker averages by.
    info = MetricsTracker()
    loss = torch.zeros((), device=real.device)
    for view, logits in paired_logits.items():
        view_loss = hinge_d_loss(
            [logit[:batch_size] for logit in logits],
            [logit[batch_size:] for logit in logits],
        )
        loss = loss + view_loss
        info[f"d_{view}"] = view_loss.detach().cpu().item() * ctx["num_frames"]
    info["d_loss"] = loss.detach().cpu().item() * ctx["num_frames"]
    return loss, info


_FBANKS = {}


def _resample_fbank(device: torch.device) -> nn.Module:
    """The feature extractor's mel module, the one the features were made with."""
    if device not in _FBANKS:
        _FBANKS[device] = VocosFbank().fbank.to(device)
    return _FBANKS[device]


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
    audio: Optional[Tensor] = None,
    audio_lens: Optional[Tensor] = None,
) -> Tuple[Tensor, MetricsTracker, Optional[dict]]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      features:
        The target acoustic feature.
      features_lens:
        The number of frames of each utterance.
      tokens:
        Input tokens that representing the transcripts.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
      discriminators:
        The discriminator set, or None when stage 2 is off.

    Returns the loss, the metrics, and the context the caller needs for the
    discriminator's own step, which is None when stage 2 is off, when no
    utterance in the batch cleared the flow-time gate, or in validation.
    """

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape

    noise = torch.randn_like(features)  # (B, T, F)

    # Sampling t from uniform distribution
    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (
            (torch.arange(batch_size, device=device) / batch_size)
            .unsqueeze(1)
            .unsqueeze(2)
        )
    t = t * params.t_max
    rollout_prob = 0.0
    if is_training and params.batch_idx_train >= params.rollout_start_step:
        ramp = params.batch_idx_train - params.rollout_start_step + 1
        rollout_prob = params.rollout_prob * min(
            1.0, ramp / max(params.rollout_ramp_steps, 1)
        )

    fbank = None
    if audio is not None:
        fbank = _resample_fbank(device)
        audio = audio.to(device)
        audio_lens = audio_lens.to(device)

    # One solve serves both consumers. Validation never solves: a 16-step
    # solve on every dev batch costs more than the validation itself, so with
    # --gan-source solve the validation log carries no adversarial terms.
    gan_on_solve = (
        discriminators is not None
        and gan_enabled(params)
        and params.gan_source == "solve"
    )
    want_solve = is_training and (params.solve_loss_weight > 0 or gan_on_solve)

    with torch.set_grad_enabled(is_training):

        terms = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            step=params.batch_idx_train,
            total_steps=params.cutoff_curriculum_steps,
            rollout_prob=rollout_prob,
            rollout_steps=params.rollout_steps,
            audio=audio,
            audio_lens=audio_lens,
            fbank=fbank,
            solve_utts=params.solve_utts if want_solve else 0,
            fm_space=params.fm_space,
        )
        fm_loss = terms["fm_loss"]
        recon_loss = terms["recon_loss"]
        loss = fm_loss + params.recon_loss_weight * recon_loss

        inner = model.module if isinstance(model, DDP) else model
        band_loss = None
        if params.band_loss_weight > 0:
            band_loss = band_level_loss(
                params, terms, features, inner.mel_centers
            )
            loss = loss + params.band_loss_weight * band_loss
        solve_loss = None
        if "solved_x1" in terms and params.solve_loss_weight > 0:
            solve_loss = solve_output_loss(
                params, terms, features, inner.mel_centers
            )
            loss = loss + params.solve_loss_weight * solve_loss

    num_frames = features_lens.sum().item()
    ctx = None
    adv_terms = {}
    # Validation runs this too, under no-grad: the adversarial and
    # feature-matching terms are the only measure of what stage 2 is actually
    # optimizing, and without them the validation loss is not comparable to the
    # training loss it is meant to track.
    with torch.set_grad_enabled(is_training):
        if discriminators is not None and gan_enabled(params):
            ctx = gan_context(
                params, terms, features, num_frames, inner.mel_centers
            )
        if ctx is not None:
            # The unwrapped module. These forwards exist to push gradients back
            # into the branch; the gradients they leave on the discriminator
            # parameters are discarded before the discriminator's own step, and
            # running them through the DDP wrapper would break its
            # one-forward-per-backward contract.
            disc_module = (
                discriminators.module
                if isinstance(discriminators, DDP)
                else discriminators
            )
            fake_logits, fake_features = run_discriminators(
                disc_module, ctx["x1_hat"], ctx
            )
            with torch.no_grad():
                real_logits, real_features = run_discriminators(
                    disc_module, ctx["features"], ctx
                )

            for view in fake_logits:
                adv = hinge_g_loss(fake_logits[view])
                fmatch = feature_matching_loss(
                    real_features[view], fake_features[view]
                )
                loss = (
                    loss
                    + params.adv_loss_weight * adv
                    + params.feat_match_loss_weight * fmatch
                )
                adv_terms[f"adv_{view}"] = adv.detach().cpu().item()
                adv_terms[f"fmatch_{view}"] = fmatch.detach().cpu().item()
            adv_terms["gan_utts"] = float(ctx["features"].size(0))

            if not is_training:
                # Training logs d_* from the discriminator's own step, which
                # validation never takes. Recomputing them from the forwards
                # just done costs nothing and keeps the key set the same on
                # both sides, so the two curves stay comparable.
                d_total = 0.0
                for view in fake_logits:
                    d_view = hinge_d_loss(
                        real_logits[view], fake_logits[view]
                    ).item()
                    d_total += d_view
                    adv_terms[f"d_{view}"] = d_view
                adv_terms["d_loss"] = d_total
                ctx = None

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    info["frames"] = num_frames
    # MetricsTracker averages by dividing by "frames", so every value is
    # pre-multiplied by the frame count here.
    info["loss"] = loss.detach().cpu().item() * num_frames
    info["fm_loss"] = fm_loss.detach().cpu().item() * num_frames
    info["recon_loss"] = recon_loss.detach().cpu().item() * num_frames
    if band_loss is not None:
        info["band_loss"] = band_loss.detach().cpu().item() * num_frames
    if solve_loss is not None:
        info["solve_loss"] = solve_loss.detach().cpu().item() * num_frames
    info["rollout_fraction"] = (
        terms["rollout_fraction"].detach().cpu().item() * num_frames
    )
    for key, value in adv_terms.items():
        info[key] = value * num_frames

    return loss, info, ctx


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
    disc_optimizer: Optional[Optimizer] = None,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches or every epochs.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer.
      scheduler:
        The learning rate scheduler, we call step() every epoch.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            extra_states=discriminator_states(discriminators, disc_optimizer),
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):

        if batch_idx % 10 == 0:
            if params.finetune:
                set_batch_count(model, get_adjusted_batch_count(params) + 100000)
            else:
                set_batch_count(model, get_adjusted_batch_count(params))

        if (
            params.valid_by_epoch and batch_idx == 0 and not params.print_diagnostics
        ) or (
            not params.valid_by_epoch
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                discriminators=discriminators,
                world_size=world_size,
            )
            model.train()
            if discriminators is not None:
                discriminators.train()
            logging.info(
                f"Epoch {params.cur_epoch}, global_batch_idx: {params.batch_idx_train},"
                f" validation: {valid_info}"
            )
            logging.info(
                f"Maximum memory allocated so far is "
                f"{torch.cuda.max_memory_allocated() // 1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        params.batch_idx_train += 1

        batch_size = len(batch["text"])

        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )

        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                loss, loss_info, ctx = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                    audio=batch.get("audio"),
                    audio_lens=batch.get("audio_lens"),
                    discriminators=discriminators,
                )
                if ctx is not None:
                    disc_loss, disc_info = compute_disc_loss(
                        params=params,
                        discriminators=discriminators,
                        ctx=ctx,
                    )
                    loss_info = loss_info + disc_info

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()
            if ctx is not None:
                # The branch's backward has just left gradients on the
                # discriminators as well, from the generator-side forwards.
                # They are the wrong sign for the discriminator's own
                # objective, so they are discarded rather than accumulated.
                disc_optimizer.zero_grad()
                scaler.scale(disc_loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            # Use the number of hours of speech to adjust the learning rate
            if params.lr_hours > 0:
                scheduler.step_epoch(
                    params.batch_idx_train
                    * params.max_duration
                    * params.world_size
                    / 3600
                )
            scaler.step(optimizer)
            if ctx is not None:
                scaler.step(disc_optimizer)
            # One update for both: the scale is a property of the fp16 range,
            # not of either optimizer, and calling update twice per step would
            # halve it twice on a single inf.
            scaler.update()
            optimizer.zero_grad()
            if ctx is not None:
                disc_optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception : {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                extra_states=discriminator_states(discriminators, disc_optimizer),
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it. The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have
            # different behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 1024.0 or (
                cur_grad_scale < 4096.0 and params.batch_idx_train % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, "
                f"batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    if discriminators is not None:
        discriminators.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )

        loss, loss_info, _ = compute_fbank_loss(
            params=params,
            model=model,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            is_training=False,
            audio=batch.get("audio"),
            audio_lens=batch.get("audio_lens"),
            discriminators=discriminators,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    features = batch["features"]
    tokens = batch["tokens"]

    logging.info(f"features shape: {features.shape}")
    num_tokens = sum(len(i) for i in tokens)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )
        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):

                loss, loss_info, _ = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                    audio=batch.get("audio"),
                    audio_lens=batch.get("audio_lens"),
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is "
            f"{torch.cuda.max_memory_allocated() // 1000000}MB"
        )


def tokenize_text(c: Cut, tokenizer):
    if hasattr(c.supervisions[0], "tokens"):
        tokens = tokenizer.tokens_to_token_ids([c.supervisions[0].tokens])
    else:
        tokens = tokenizer.texts_to_token_ids([c.supervisions[0].text])
    c.supervisions[0].tokens = tokens[0]
    return c


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    params.valid_interval = params.save_every_n
    # Set epoch to a large number to ignore it.
    if params.num_iters > 0:
        params.num_epochs = 1000000
    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    params.update(model_config["model"])
    params.update(model_config["feature"])

    if not 0.0 <= params.rollout_prob <= 1.0:
        raise ValueError(
            f"--rollout-prob must be in [0, 1], got {params.rollout_prob}"
        )
    if params.rollout_steps < 1:
        raise ValueError(
            f"--rollout-steps must be positive, got {params.rollout_steps}"
        )
    if not 0.0 < params.t_max <= 1.0:
        raise ValueError(f"--t-max must be in (0, 1], got {params.t_max}")
    if params.fm_space == "velocity" and params.t_max >= 1.0:
        raise ValueError("--fm-space velocity needs --t-max below 1")

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    if torch.cuda.is_available():
        params.device = torch.device("cuda", rank)
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=params.token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=params.token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=params.token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=params.token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    params.update(tokenizer_config)

    logging.info(params)

    logging.info("About to create model")

    model = ZipVoiceDual(
        **model_config["model"],
        **model_config.get("dual", {}),
        feat_scale=params.feat_scale,
        sampling_rate=params.sampling_rate,
        **tokenizer_config,
    )

    assert params.checkpoint is not None, (
        "the branch is trained beside a frozen backbone, so a pre-trained "
        "backbone checkpoint is required"
    )
    logging.info(f"Loading the frozen backbone from {params.checkpoint}")
    # The branch is not in a backbone checkpoint, but nothing else may be
    # missing: a renamed key would otherwise leave part of the backbone at its
    # random initialization, and a random backbone still produces a hidden
    # state and a trajectory for the branch to be fitted against.
    # Not load_checkpoint: it hands back the checkpoint's remaining fields, not
    # the key report this needs, and it loads non-strictly without saying what
    # it skipped.
    state = torch.load(params.checkpoint, map_location="cpu", weights_only=False)
    state = state["model"]
    if next(iter(state)).startswith("module."):
        state = {k[len("module.") :]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("branch.") for k in missing), missing

    model.freeze_backbone()
    num_param = sum([p.numel() for p in model.parameters()])
    num_trainable = sum([p.numel() for p in model.branch.parameters()])
    logging.info(
        f"Number of parameters : {num_param}, of which {num_trainable} "
        "are trainable (the branch)"
    )

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = None
    if params.resume_checkpoint is not None:
        logging.info(f"Resuming from {params.resume_checkpoint}")
        checkpoints = resume_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            filename=params.resume_checkpoint,
        )
    elif params.start_epoch > 1:
        checkpoints = resume_checkpoint(params=params, model=model, model_avg=model_avg)

    model = model.to(params.device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Stage 2's discriminators live outside the model: they are a training
    # signal, not part of what is exported, so they stay out of the state dict
    # an inference run loads.
    discriminators: Optional[Union[nn.Module, DDP]] = None
    disc_optimizer: Optional[Optimizer] = None
    if params.gan_start_step >= 0:
        discriminators = BweDiscriminators(
            n_mels=model_config["model"]["feat_dim"],
            # The harmonic view needs an F0 track, which nothing here computes;
            # the spectral and cross-band views judge the same mel without one.
            use_harmonic=False,
            **model_config.get("dual_disc", {}),
        ).to(params.device)
        num_disc_param = sum([p.numel() for p in discriminators.parameters()])
        logging.info(f"Number of discriminator parameters : {num_disc_param}")
        if world_size > 1:
            discriminators = DDP(discriminators, device_ids=[rank])
        disc_optimizer = torch.optim.AdamW(
            discriminators.parameters(),
            lr=params.disc_lr,
            betas=(0.8, 0.99),
        )

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            # The branch only. Handing ScaledAdam the frozen backbone as well
            # would still build parameter groups for it and still apply its
            # weight decay, which shrinks a weight whether or not a gradient
            # ever arrives for it.
            model.module.branch if isinstance(model, DDP) else model.branch,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    assert params.lr_hours >= 0

    if params.finetune and params.lr_decay_start >= 0:
        scheduler = CosineDecayLRScheduler(
            optimizer,
            decay_start=params.lr_decay_start,
            decay_steps=params.num_iters - params.lr_decay_start,
            final_ratio=params.lr_final_ratio,
        )
    elif params.finetune:
        scheduler = FixedLRScheduler(optimizer)
    elif params.lr_hours > 0:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_hours)
    else:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    scaler = create_grad_scaler(enabled=params.use_fp16)

    if checkpoints is not None and (
        params.start_epoch > 1 or params.resume_checkpoint is not None
    ):
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

        if "grad_scaler" in checkpoints:
            logging.info("Loading grad scaler state dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

        if discriminators is not None and "discriminators" in checkpoints:
            logging.info("Loading discriminator state dict")
            disc_module = (
                discriminators.module
                if isinstance(discriminators, DDP)
                else discriminators
            )
            disc_module.load_state_dict(checkpoints["discriminators"])
            if checkpoints.get("disc_optimizer") is not None:
                disc_optimizer.load_state_dict(checkpoints["disc_optimizer"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    def remove_short_and_long_utt(c: Cut, min_len: float, max_len: float):
        if c.duration < min_len or c.duration > max_len:
            return False
        return True

    _remove_short_and_long_utt = partial(
        remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len
    )

    datamodule = TtsDataModule(args)
    if params.dataset == "emilia":
        train_cuts = CutSet.mux(
            datamodule.train_emilia_EN_cuts(),
            datamodule.train_emilia_ZH_cuts(),
            weights=[46000, 49000],
        )
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = CutSet.mux(
            datamodule.dev_emilia_EN_cuts(),
            datamodule.dev_emilia_ZH_cuts(),
            weights=[0.5, 0.5],
        )
    elif params.dataset == "libritts":
        train_cuts = datamodule.train_libritts_cuts()
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = datamodule.dev_libritts_cuts()
    else:
        assert params.dataset == "custom"
        train_cuts = datamodule.train_custom_cuts(params.train_manifest)
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = datamodule.dev_custom_cuts(params.dev_manifest)
        # To avoid OOM issues due to too long dev cuts
        dev_cuts = dev_cuts.filter(_remove_short_and_long_utt)

    if params.tokenizer in ["emilia", "espeak", "dialog"]:
        if not hasattr(train_cuts[0].supervisions[0], "tokens") or not hasattr(
            dev_cuts[0].supervisions[0], "tokens"
        ):
            logging.warning(
                f"Using {params.tokenizer} tokenizer but tokens are not prepared,"
                f"will tokenize on-the-fly, which can slow down training significantly."
            )
    _tokenize_text = partial(tokenize_text, tokenizer=tokenizer)
    train_cuts = train_cuts.map(_tokenize_text)
    dev_cuts = dev_cuts.map(_tokenize_text)

    train_dl = datamodule.train_dataloaders(train_cuts)

    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    if params.scan_oom:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    logging.info("Training started")

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        if params.lr_hours == 0:
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            discriminators=discriminators,
            disc_optimizer=disc_optimizer,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            extra_states=discriminator_states(discriminators, disc_optimizer),
            rank=rank,
        )

        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    if args.resample_degrade:
        # Forced rather than asserted: the two flags are one decision, and a
        # run that silently fell back to the mel floor would look like a
        # resampling run in the log.
        args.return_audio = True

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
