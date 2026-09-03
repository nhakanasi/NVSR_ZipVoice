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
This script trains a ZipVoice model jointly with a mel bandwidth extender, so
that speech prompts recorded at any sampling rate can condition the model.

Training runs in three stages, selected by --bwe-joint-start-step and
--bwe-gan-start-step:

(0) BWE pretrain: --bwe-freeze-zipvoice 1 --bwe-fm-loss-weight 0
    --bwe-gan-start-step -1. The extender learns to reconstruct the missing band
    from the reconstruction loss alone; the TTS is not involved.

python3 -m zipvoice.bin.train_zipvoice_bwe \
    --world-size 8 \
    --use-fp16 1 \
    --num-epochs 2 \
    --max-duration 500 \
    --finetune 1 \
    --base-lr 0.001 \
    --bwe-freeze-zipvoice 1 \
    --bwe-fm-loss-weight 0.0 \
    --bwe-gan-start-step -1 \
    --checkpoint download/zipvoice/model.pt \
    --model-config conf/zipvoice_bwe.json \
    --tokenizer emilia \
    --token-file "data/tokens_emilia.txt" \
    --dataset emilia \
    --manifest-dir data/fbank \
    --exp-dir exp/zipvoice_bwe_pretrain

(A) Frozen TTS: --bwe-freeze-zipvoice 1, flow-matching loss on, GAN still off.
    The extender additionally learns to produce a mel the TTS finds useful.

(B) Joint adversarial: --bwe-freeze-zipvoice 0 --bwe-gan-start-step 0, low LR.
    All three discriminators are enabled and the TTS is fine-tuned alongside.

python3 -m zipvoice.bin.train_zipvoice_bwe \
    --world-size 8 \
    --use-fp16 1 \
    --num-epochs 4 \
    --max-duration 500 \
    --finetune 1 \
    --base-lr 0.0001 \
    --bwe-freeze-zipvoice 0 \
    --bwe-gan-start-step 0 \
    --bwe-curriculum-rho 0.0 \
    --checkpoint exp/zipvoice_bwe_stage_a/epoch-2.pt \
    --model-config conf/zipvoice_bwe.json \
    --tokenizer emilia \
    --token-file "data/tokens_emilia.txt" \
    --dataset emilia \
    --manifest-dir data/fbank \
    --exp-dir exp/zipvoice_bwe_joint
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
from zipvoice.models.modules.bwe import ild_loss, lsd_loss, ndl_loss, sample_cutoff
from zipvoice.models.modules.bwe_discriminators import (
    BweDiscriminators,
    f0_from_audio,
    f0_from_mel,
    feature_matching_loss,
    hinge_d_loss,
    hinge_g_loss,
)
from zipvoice.models.zipvoice_bwe import ZipVoiceBWE
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import (
    load_checkpoint,
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
    make_pad_mask,
    prepare_input,
    set_batch_count,
    setup_dist,
    setup_logger,
    str2bool,
    torch_autocast,
)
from zipvoice.utils.hooks import register_inf_check_hooks
from zipvoice.utils.lr_scheduler import Eden, FixedLRScheduler, LRScheduler
from zipvoice.utils.optim import ScaledAdam
from zipvoice.utils.wandb_logger import FanoutWriter, init_wandb

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
        "--wandb",
        type=str2bool,
        default=False,
        help="Mirror the tensorboard metrics to Weights & Biases. Authenticate "
        "with the WANDB_API_KEY environment variable or a prior `wandb login`.",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="zipvoice-bwe",
        help="Weights & Biases project to log to.",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Name of the Weights & Biases run. Defaults to the exp-dir name.",
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
        "--resume-from",
        type=str,
        default=None,
        help="""Resume an interrupted run from this checkpoint, e.g.
        exp-dir/checkpoint-10000.pt. Restores the model, model_avg, both
        optimizers, the scheduler, the grad scaler, the discriminators and
        batch_idx_train, so a run killed mid-epoch picks up its step count and
        its adversarial game rather than restarting them. Unlike --start-epoch
        this can name a mid-epoch checkpoint, which is the only kind an
        iteration-bounded stage ever writes. The dataloader position is not
        restored: the epoch replays from its first batch, so the remaining
        steps see a different slice of the epoch than they originally would
        have. Takes precedence over --checkpoint, which is for seeding a fresh
        run from someone else's weights.
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
        "--valid-interval",
        type=int,
        default=None,
        help="""Compute the validation loss every this many batches. Defaults
        to --save-every-n, which is usually what you want. Set it higher to
        checkpoint often without paying for validation as often: validation
        here runs the discriminators too, so it costs several times what a
        reconstruction-only pass would.
        """,
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
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="""Half precision format, when --use-fp16 is on. bf16 carries
        float32's exponent range, so it needs no loss scaling and cannot hit the
        "grad_scale is too small" failure that float16 reaches when gradients
        underflow; it costs a little mantissa precision in exchange. Requires
        Ampere or newer.
        """,
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--condition-drop-ratio",
        type=float,
        default=0.2,
        help="The drop rate of text condition during training.",
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

    parser.add_argument(
        "--bwe-freeze-zipvoice",
        type=str2bool,
        default=True,
        help="Freeze all ZipVoice parameters and train only the bandwidth "
        "extender (and the discriminators). Stages 0 and A.",
    )

    parser.add_argument(
        "--bwe-gan-start-step",
        type=int,
        default=-1,
        help="Enable the discriminators once batch_idx_train reaches this value. "
        "A negative value disables them for the whole run.",
    )

    parser.add_argument(
        "--bwe-cutoff-min",
        type=float,
        default=1000.0,
        help="Minimum simulated band-limiting cutoff, in Hz.",
    )

    parser.add_argument(
        "--bwe-cutoff-max",
        type=float,
        default=12000.0,
        help="Maximum simulated band-limiting cutoff, in Hz. Defaults to the "
        "Nyquist frequency of the 24 kHz mel.",
    )

    parser.add_argument(
        "--bwe-curriculum-beta",
        type=float,
        default=4.0,
        help="Exponential bias towards high cutoffs at the start of the "
        "Easy-to-Balanced curriculum.",
    )

    parser.add_argument(
        "--bwe-curriculum-rho",
        type=float,
        default=0.7,
        help="Fraction of --bwe-curriculum-steps over which the curriculum "
        "anneals to uniform cutoff sampling. Set to 0 for uniform sampling "
        "throughout, which is what the adversarial refinement stage uses.",
    )

    parser.add_argument(
        "--bwe-curriculum-steps",
        type=int,
        default=200000,
        help="Total number of steps the curriculum is scheduled against. The "
        "anneal to uniform completes at --bwe-curriculum-rho times this.",
    )

    parser.add_argument(
        "--bwe-recon-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the whole reconstruction objective on the restored band.",
    )

    parser.add_argument(
        "--bwe-recon-loss-type",
        type=str,
        default="l1",
        choices=["l1", "spectral"],
        help="Reconstruction objective on the restored band. 'l1' is the mean "
        "absolute error on the scaled log-mel. 'spectral' replaces it with the "
        "LSD + ILD + NDL combination of HRTFformer (arXiv 2510.01891), which "
        "that paper's ablation found to beat a plain squared error. The spectral "
        "terms are measured in dB and are one to two orders of magnitude larger "
        "than the L1, so --bwe-recon-loss-weight has to come down with them.",
    )

    parser.add_argument(
        "--bwe-lsd-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the log-spectral-distance term of --bwe-recon-loss-type "
        "spectral.",
    )

    parser.add_argument(
        "--bwe-ild-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the inter-band level difference term of "
        "--bwe-recon-loss-type spectral.",
    )

    parser.add_argument(
        "--bwe-ndl-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the neighbor-dissimilarity term of --bwe-recon-loss-type "
        "spectral.",
    )

    parser.add_argument(
        "--bwe-fm-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the ZipVoice flow-matching loss. Set to 0 to pretrain "
        "the extender on reconstruction alone.",
    )

    parser.add_argument(
        "--bwe-adv-loss-weight",
        type=float,
        default=0.1,
        help="Weight of each discriminator's adversarial loss.",
    )

    parser.add_argument(
        "--bwe-feat-match-loss-weight",
        type=float,
        default=2.0,
        help="Weight of each discriminator's feature-matching loss.",
    )

    parser.add_argument(
        "--bwe-disc-lr",
        type=float,
        default=2e-4,
        help="Learning rate of the discriminators' AdamW optimizer.",
    )

    parser.add_argument(
        "--bwe-f0-source",
        type=str,
        default="mel",
        choices=["mel", "audio"],
        help="Where the harmonic discriminator gets its F0 track. 'audio' is "
        "more accurate but forces audio loading in the dataloader.",
    )

    parser.add_argument(
        "--bwe-use-harmonic-disc",
        type=str2bool,
        default=True,
        help="Enable the harmonic-consistency discriminator.",
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
    return (
        params.bwe_gan_start_step >= 0
        and params.batch_idx_train >= params.bwe_gan_start_step
    )


def discriminator_states(
    discriminators: Optional[Union[nn.Module, DDP]],
    disc_optimizer: Optional[Optimizer],
) -> Optional[dict]:
    """The discriminator state to store in a checkpoint, so a resumed run picks
    the adversarial game back up rather than restarting it against a generator
    that has moved on."""
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
    """Run every discriminator view on one mel, with the shared conditioning."""
    return discriminators(
        mel=mel * ctx["frame_mask"],
        mask=ctx["band_mask"],
        cutoff_hz=ctx["cutoff_hz"],
        centers=ctx["centers"],
        f0=ctx["f0"],
        voiced=ctx["voiced"],
    )


def compute_disc_loss(
    params: AttributeDict,
    discriminators: Union[nn.Module, DDP],
    ctx: dict,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Hinge discriminator loss on the clean mel (real) against the restored mel
    (fake, detached so no gradient reaches the extender).

    Real and fake go through in a single batched forward: DDP's reducer only
    tolerates one forward per backward, and none of the discriminator stacks
    carry batch statistics (they use weight norm), so batching them together is
    numerically identical to two separate passes.
    """
    real = ctx["features"]
    fake = ctx["mel_fb"].detach()
    batch_size = real.size(0)

    paired_ctx = dict(ctx)
    for key in ("band_mask", "cutoff_hz", "frame_mask", "f0", "voiced"):
        value = ctx[key]
        if value is not None:
            paired_ctx[key] = torch.cat([value, value], dim=0)

    paired_logits, _ = run_discriminators(
        discriminators, torch.cat([real, fake], dim=0), paired_ctx
    )
    real_logits = {
        view: [logit[:batch_size] for logit in logits]
        for view, logits in paired_logits.items()
    }
    fake_logits = {
        view: [logit[batch_size:] for logit in logits]
        for view, logits in paired_logits.items()
    }

    # "frames" is deliberately not set here: this tracker is added to the
    # generator's, which already carries the frame count.
    info = MetricsTracker()
    loss = torch.zeros((), device=ctx["features"].device)
    for view in real_logits:
        view_loss = hinge_d_loss(real_logits[view], fake_logits[view])
        loss = loss + view_loss
        info[f"d_{view}"] = view_loss.detach().cpu().item() * ctx["num_frames"]
    info["d_loss"] = loss.detach().cpu().item() * ctx["num_frames"]
    return loss, info


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    discriminators: Optional[Union[nn.Module, DDP]],
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
    audio: Optional[Tensor] = None,
) -> Tuple[Tensor, MetricsTracker, dict]:
    """
    Compute the generator loss given the model and its inputs.

    The loss is the ZipVoice flow-matching loss against the clean full-band mel,
    plus the bandwidth extender's reconstruction loss and, once the
    discriminators are enabled, its adversarial and feature-matching losses.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      discriminators:
        The discriminator set, or None when the GAN terms are disabled.
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
      audio:
        The waveform, needed only when --bwe-f0-source is "audio".

    Returns:
      A tuple of the loss, the metrics, and a context dict that
      :func:`compute_disc_loss` needs for the discriminator step.
    """

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    inner = model.module if isinstance(model, DDP) else model

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

    # One band-limiting cutoff per utterance, drawn from the Easy-to-Balanced
    # curriculum during training and uniformly during validation.
    cutoff_hz = sample_cutoff(
        batch_size=batch_size,
        step=params.batch_idx_train,
        total_steps=params.bwe_curriculum_steps if is_training else 0,
        f_min=params.bwe_cutoff_min,
        f_max=params.bwe_cutoff_max,
        beta=params.bwe_curriculum_beta,
        rho=params.bwe_curriculum_rho if is_training else 0.0,
        device=device,
    )

    with torch.set_grad_enabled(is_training):

        fm_loss, mel_fb, band_mask = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            cutoff_hz=cutoff_hz,
            condition_drop_ratio=params.condition_drop_ratio,
        )

        frame_mask = (
            (~make_pad_mask(features_lens, max_len=features.size(1)))
            .unsqueeze(-1)
            .to(features.dtype)
        )  # (B, T, 1)

        # Reconstruction is scored only on the restored band and on real frames,
        # and normalized by their count so that its magnitude stays comparable
        # across cutoff frequencies.
        recon_weight = band_mask * frame_mask  # (B, T, F)
        recon_terms = {}
        if params.bwe_recon_loss_type == "l1":
            recon_loss = (mel_fb - features).abs().mul(recon_weight).sum() / (
                recon_weight.sum().clamp(min=1.0)
            )
        else:
            recon_terms["bwe_lsd"] = lsd_loss(
                mel_fb, features, recon_weight, params.feat_scale
            )
            recon_terms["bwe_ild"] = ild_loss(
                mel_fb, features, band_mask, frame_mask, params.feat_scale
            )
            recon_terms["bwe_ndl"] = ndl_loss(
                mel_fb, features, recon_weight, params.feat_scale
            )
            recon_loss = (
                params.bwe_lsd_loss_weight * recon_terms["bwe_lsd"]
                + params.bwe_ild_loss_weight * recon_terms["bwe_ild"]
                + params.bwe_ndl_loss_weight * recon_terms["bwe_ndl"]
            )

        loss = params.bwe_recon_loss_weight * recon_loss
        if params.bwe_fm_loss_weight > 0:
            loss = loss + params.bwe_fm_loss_weight * fm_loss

    total_frames = features_lens.sum().item()
    info = MetricsTracker()
    info["frames"] = total_frames
    info["fm_loss"] = fm_loss.detach().cpu().item() * total_frames
    info["bwe_recon"] = recon_loss.detach().cpu().item() * total_frames
    for name, term in recon_terms.items():
        info[name] = term.detach().cpu().item() * total_frames

    ctx = {
        "features": features,
        "mel_fb": mel_fb,
        "band_mask": band_mask,
        "cutoff_hz": cutoff_hz,
        "centers": inner.mel_centers,
        "frame_mask": frame_mask,
        "num_frames": total_frames,
        "f0": None,
        "voiced": None,
    }

    # Validation runs this too, under no-grad: the adversarial and
    # feature-matching terms are the only measure of the thing the GAN is
    # actually optimising, and without them valid_loss is not comparable to the
    # training loss it is meant to track.
    with torch.set_grad_enabled(is_training):
        if discriminators is not None and gan_enabled(params):
            if params.bwe_use_harmonic_disc:
                if params.bwe_f0_source == "audio":
                    assert audio is not None, (
                        "--bwe-f0-source audio requires the dataloader to return "
                        "audio; pass --return-audio 1"
                    )
                    ctx["f0"], ctx["voiced"] = f0_from_audio(
                        audio.to(device), num_frames=features.size(1)
                    )
                else:
                    ctx["f0"], ctx["voiced"] = f0_from_mel(
                        features, inner.mel_centers
                    )

            # The unwrapped module: in training these forwards exist to push
            # gradients back into the extender, the gradients they leave on the
            # discriminator parameters are discarded before the discriminator
            # step, and running them through the DDP wrapper would break its
            # one-forward-per-backward contract.
            disc_module = (
                discriminators.module
                if isinstance(discriminators, DDP)
                else discriminators
            )
            fake_logits, fake_features = run_discriminators(
                disc_module, mel_fb, ctx
            )
            with torch.no_grad():
                real_logits, real_features = run_discriminators(
                    disc_module, features, ctx
                )

            for view in fake_logits:
                adv = hinge_g_loss(fake_logits[view])
                fmatch = feature_matching_loss(
                    real_features[view], fake_features[view]
                )
                loss = loss + params.bwe_adv_loss_weight * adv
                loss = loss + params.bwe_feat_match_loss_weight * fmatch
                info[f"adv_{view}"] = adv.detach().cpu().item() * total_frames
                info[f"fmatch_{view}"] = (
                    fmatch.detach().cpu().item() * total_frames
                )

            if not is_training:
                # Training logs d_* from the discriminator's own step, which
                # validation never takes. Recomputing them from the forwards
                # just done costs nothing and keeps the key set the same on
                # both sides, so the two curves are comparable.
                d_total = 0.0
                for view in fake_logits:
                    d_view = hinge_d_loss(
                        real_logits[view], fake_logits[view]
                    ).item()
                    d_total += d_view
                    info[f"d_{view}"] = d_view * total_frames
                info["d_loss"] = d_total * total_frames

    assert loss.requires_grad == is_training
    info["loss"] = loss.detach().cpu().item() * total_frames

    return loss, info, ctx


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    discriminators: Optional[Union[nn.Module, DDP]],
    optimizer: Optimizer,
    disc_optimizer: Optional[Optimizer],
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
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
      discriminators:
        The discriminator set, or None when the GAN terms are disabled for this
        run.
      optimizer:
        The optimizer of the generator (ZipVoice plus the bandwidth extender).
      disc_optimizer:
        The AdamW optimizer of the discriminators, or None.
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
    if discriminators is not None:
        discriminators.train()
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
            with torch_autocast(dtype=params.amp_torch_dtype, enabled=params.use_fp16):
                loss, loss_info, ctx = compute_fbank_loss(
                    params=params,
                    model=model,
                    discriminators=discriminators,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                    audio=batch.get("audio", None),
                )

            # Discriminator step, 1:1 with the generator step.
            train_disc = discriminators is not None and gan_enabled(params)
            if train_disc:
                with torch_autocast(
                    dtype=params.amp_torch_dtype, enabled=params.use_fp16
                ):
                    disc_loss, disc_info = compute_disc_loss(
                        params=params,
                        discriminators=discriminators,
                        ctx=ctx,
                    )
                loss_info = loss_info + disc_info

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()

            if train_disc:
                # The generator's adversarial term backpropagates through the
                # discriminators and leaves gradients on their parameters.
                # Discard those before computing the discriminators' own.
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
            if train_disc:
                scaler.step(disc_optimizer)
            # A single update() per iteration, after every step().
            scaler.update()
            optimizer.zero_grad()
            if train_disc:
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
        if params.batch_idx_train % 100 == 0 and params.use_grad_scaler:
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
            cur_grad_scale = scaler._scale.item() if params.use_grad_scaler else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, "
                f"batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (
                    f"grad_scale: {scaler._scale.item()}"
                    if params.use_grad_scaler
                    else ""
                )
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_grad_scaler:
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
            discriminators=discriminators,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            is_training=False,
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
            with torch_autocast(dtype=params.amp_torch_dtype, enabled=params.use_fp16):

                loss, loss_info, _ = compute_fbank_loss(
                    params=params,
                    model=model,
                    discriminators=None,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
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
    if args.bwe_f0_source == "audio":
        # The harmonic discriminator needs the waveform to estimate F0. Set
        # before params is built so the logged and checkpointed params say so.
        args.return_audio = True

    params = get_params()
    params.update(vars(args))
    # Autocast dtype and loss scaling are one decision, resolved here so no
    # call site has to re-derive it. Only float16 needs a GradScaler.
    params.amp_torch_dtype = (
        torch.bfloat16 if params.amp_dtype == "bf16" else torch.float16
    )
    params.use_grad_scaler = params.use_fp16 and params.amp_dtype == "fp16"
    if params.valid_interval is None:
        params.valid_interval = params.save_every_n
    # Set epoch to a large number to ignore it.
    if params.num_iters > 0:
        params.num_epochs = 1000000
    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    params.update(model_config["model"])
    params.update(model_config["feature"])

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

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

    model = ZipVoiceBWE(
        **model_config["model"],
        **model_config.get("bwe", {}),
        **tokenizer_config,
        feat_scale=params.feat_scale,
        sampling_rate=params.sampling_rate,
    )

    if params.checkpoint is not None and params.resume_from is None:
        logging.info(f"Loading pre-trained model from {params.checkpoint}")
        # strict=False so that a stock ZipVoice checkpoint, which has no
        # bandwidth extender, can seed stage 0 / stage A.
        _ = load_checkpoint(filename=params.checkpoint, model=model, strict=False)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    if params.bwe_freeze_zipvoice:
        logging.info("Freezing all ZipVoice parameters; training only the extender")
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("bwe."))

    discriminators: Optional[nn.Module] = None
    if params.bwe_gan_start_step >= 0:
        discriminators = BweDiscriminators(
            n_mels=params.feat_dim,
            use_harmonic=params.bwe_use_harmonic_disc,
            **model_config.get("bwe_disc", {}),
        )
        logging.info(
            "Number of discriminator parameters : "
            f"{sum(p.numel() for p in discriminators.parameters())}"
        )

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    # Two ways in: --start-epoch names a completed epoch, --resume-from names an
    # arbitrary checkpoint. Everything downstream only cares that we are resuming.
    resuming = params.start_epoch > 1 or params.resume_from is not None
    checkpoints = None
    if resuming:
        checkpoints = resume_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            filename=params.resume_from,
        )
        if params.resume_from is not None:
            # batch_idx_train came back from the checkpoint; the epoch loop has to
            # restart on the epoch that was in flight, not on epoch 1.
            params.start_epoch = checkpoints.get("cur_epoch", params.start_epoch)
            logging.info(
                f"Resuming from {params.resume_from} at epoch "
                f"{params.start_epoch}, batch_idx_train {params.batch_idx_train}"
            )
        if discriminators is not None and checkpoints is not None:
            if "discriminators" in checkpoints:
                logging.info("Loading discriminator state dict")
                discriminators.load_state_dict(checkpoints["discriminators"])

    model = model.to(params.device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    disc_optimizer: Optional[Optimizer] = None
    if discriminators is not None:
        discriminators = discriminators.to(params.device)
        if world_size > 1:
            discriminators = DDP(discriminators, device_ids=[rank])
        # Plain AdamW rather than ScaledAdam: the paper specifies AdamW, and
        # ScaledAdam's parameter-scale clipping is a poor fit for a GAN
        # discriminator.
        disc_optimizer = torch.optim.AdamW(
            discriminators.parameters(),
            lr=params.bwe_disc_lr,
            betas=(0.8, 0.99),
        )
        if resuming and checkpoints is not None:
            if "disc_optimizer" in checkpoints:
                logging.info("Loading discriminator optimizer state dict")
                disc_optimizer.load_state_dict(checkpoints["disc_optimizer"])

    assert params.lr_hours >= 0

    if params.finetune:
        scheduler = FixedLRScheduler(optimizer)
    elif params.lr_hours > 0:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_hours)
    else:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    scaler = create_grad_scaler(enabled=params.use_grad_scaler)

    if resuming and checkpoints is not None:
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

    # Rank 0 owns both logging backends; FanoutWriter keeps every call site
    # talking to a single object regardless of which of them are enabled. This
    # runs after the checkpoint is loaded because the checkpoint carries the id
    # of the Weights & Biases run that wrote it, and a resumed run has to
    # continue that history rather than start a second one alongside it.
    if rank == 0 and (args.tensorboard or params.wandb):
        tb_writer = FanoutWriter(
            tb_writer=(
                SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
                if args.tensorboard
                else None
            ),
            wandb_run=init_wandb(
                params,
                run_id=(checkpoints or {}).get("wandb_run_id"),
            ),
        )
        if tb_writer.wandb_run is not None:
            # Rides into every checkpoint through save_checkpoint(params=params).
            params.wandb_run_id = tb_writer.wandb_run.id
            logging.info(f"Weights & Biases run id: {params.wandb_run_id}")
    else:
        tb_writer = None

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
            discriminators=discriminators,
            model_avg=model_avg,
            optimizer=optimizer,
            disc_optimizer=disc_optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
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

    if tb_writer is not None:
        tb_writer.close()

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

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
