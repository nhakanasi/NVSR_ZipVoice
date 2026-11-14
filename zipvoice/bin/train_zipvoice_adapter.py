"""
Train the bandwidth-extension side branch of ZipVoiceAdapter.

Stage 1 -- zero-shot text-to-speech -- is skipped: it is what the released
ZipVoice checkpoint already is. What is trained here is stage 2, in which the
side branch learns to supply the band a band-limited prompt does not have.

The stage runs in two parts, and the first is the one to reach for. Stage 2a
trains only the branch; every released tensor has ``requires_grad`` cleared,
which removes it from the optimiser's parameter groups entirely rather than
merely zeroing its gradient. That distinction matters: ScaledAdam factors each
update into a direction and a per-tensor scale, and an earlier experiment in
this repository measured released weights being rescaled by 0.904 despite their
gradient being zeroed throughout. A parameter the optimiser never receives
cannot be rescaled. Together with the band mask that gates the branch's output,
this makes stage 2a keep every released weight bit-identical and the full-band
arm algebraically unchanged, so the result can be merged back onto the released
checkpoint.

Stage 2b additionally releases the flow-matching decoder at a low learning rate.
It is the fallback, to be run only if 2a plateaus short, and it gives up the
bit-identical property.

Two loss terms and no others: ZipVoice's flow-matching loss over the generated
span, and an L1 spectrogram reconstruction loss on the endpoint estimate over
the bins the band-limiting removed.

Stage 2a:

python3 -m zipvoice.bin.train_zipvoice_adapter \
    --world-size 1 \
    --use-fp16 1 \
    --num-iters 30000 \
    --max-duration 72 \
    --finetune 1 \
    --base-lr 0.001 \
    --adapter-trainable adapter \
    --checkpoint download/zipvoice/model.pt \
    --model-config conf/zipvoice_adapter.json \
    --token-file download/zipvoice/tokens.txt \
    --dataset libritts \
    --exp-dir exp/adapter

Stage 2b:

python3 -m zipvoice.bin.train_zipvoice_adapter \
    --world-size 1 \
    --use-fp16 1 \
    --num-iters 10000 \
    --max-duration 72 \
    --finetune 1 \
    --base-lr 0.0005 \
    --adapter-trainable decoder \
    --curriculum-rho 0.0 \
    --checkpoint exp/adapter/checkpoint-30000.pt \
    --model-config conf/zipvoice_adapter.json \
    --token-file download/zipvoice/tokens.txt \
    --dataset libritts \
    --exp-dir exp/adapter_joint
"""

import argparse
import contextlib
import copy
import json
import logging
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

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
from zipvoice.models.modules.bwe import resample_band_limit, sample_cutoff
from zipvoice.models.modules.dmd import DMDController
from zipvoice.utils.feature import VocosFbank
from zipvoice.models.modules.perceptual import SpeakerPerceptual
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter, adapter_kwargs
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
        exp-dir/checkpoint-10000.pt. Restores the model, model_avg, the
        optimizer, the scheduler, the grad scaler and batch_idx_train, so a run
        killed mid-epoch picks up its step count rather than restarting it.
        Unlike --start-epoch this can name a mid-epoch checkpoint, which is the
        only kind an iteration-bounded stage ever writes. The dataloader
        position is not restored: the epoch replays from its first batch, so the
        remaining steps see a different slice of the epoch than they originally
        would have. Takes precedence over --checkpoint, which is for seeding a
        fresh run from someone else's weights.
        """,
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
        default="zipvoice-adapter",
        help="Weights & Biases project to log to.",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Name of the Weights & Biases run. Defaults to the exp-dir name.",
    )

    parser.add_argument(
        "--valid-interval",
        type=int,
        default=None,
        help="Compute the validation loss every this many batches. Defaults to "
        "--save-every-n, which is usually what you want. Set it higher to "
        "checkpoint often without paying for validation as often.",
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
        "--save-every-n",
        type=int,
        default=5000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train %% save_every_n == 0. The checkpoint filename
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

    add_adapter_arguments(parser)
    add_dmd_arguments(parser)

    return parser


def add_dmd_arguments(parser: argparse.ArgumentParser) -> None:
    """Distribution-matching distillation of the one-pass coupled student.

    See ``zipvoice/models/modules/dmd.py``. Off unless --dmd-loss-weight > 0.
    Needs --adapter-trainable lora (the student LoRA is what lets the backbone
    learn to read its own branch's writes in a single pass) and runs on one GPU.
    """
    parser.add_argument(
        "--dmd-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the DMD generator loss on the solved sample. 0 is off.",
    )
    parser.add_argument(
        "--dmd-fm-weight",
        type=float,
        default=1.0,
        help="Weight of the ordinary flow-matching + reconstruction loss while "
        "DMD is on. It anchors the student to real data.",
    )
    parser.add_argument(
        "--dmd-fake-rank",
        type=int,
        default=64,
        help="LoRA rank of the fake score on the decoder.",
    )
    parser.add_argument(
        "--dmd-fake-lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate of the fake score.",
    )
    parser.add_argument(
        "--dmd-start-step",
        type=int,
        default=1000,
        help="Batches of fake-score-only updates before the generator loss "
        "starts. The fake score starts as the teacher, so before it has fitted "
        "the student the only generator signal is the guidance direction.",
    )
    parser.add_argument(
        "--dmd-gen-interval",
        type=int,
        default=5,
        help="Generator update every N batches (DMD2's two time-scale rule); "
        "the fake score updates every batch.",
    )
    parser.add_argument(
        "--dmd-num-step",
        type=int,
        default=16,
        help="Solver steps of the student rollout. Train at the step count you "
        "deploy at.",
    )
    parser.add_argument(
        "--dmd-guidance",
        type=float,
        default=1.0,
        help="Guidance scale of the student rollout, i.e. the deployed one.",
    )
    parser.add_argument(
        "--dmd-real-guidance",
        type=float,
        default=1.0,
        help="Guidance scale of the clean-prompt teacher score.",
    )
    parser.add_argument(
        "--dmd-t-shift",
        type=float,
        default=0.5,
        help="t_shift of the student rollout, i.e. the deployed one.",
    )
    parser.add_argument(
        "--dmd-grad-steps",
        type=int,
        default=1,
        help="Solver steps at the end of the rollout that carry a graph. Each "
        "costs one guided backbone forward's activations.",
    )
    parser.add_argument(
        "--dmd-t-min",
        type=float,
        default=0.02,
        help="Lower end of the noise level the sample is re-noised at.",
    )
    parser.add_argument(
        "--dmd-t-max",
        type=float,
        default=0.98,
        help="Upper end of the noise level the sample is re-noised at.",
    )
    parser.add_argument(
        "--dmd-utts",
        type=int,
        default=1,
        help="Utterances per DMD rollout.",
    )
    parser.add_argument(
        "--dmd-max-seconds",
        type=float,
        default=8.0,
        help="Only utterances up to this long are rolled out (0 = no cap).",
    )
    parser.add_argument(
        "--dmd-cutoffs",
        type=str,
        default="4000,8000",
        help="Comma-separated prompt cutoffs in Hz, one drawn per utterance.",
    )
    parser.add_argument(
        "--dmd-prompt-frac",
        type=str,
        default="0.2,0.5",
        help="Range of the prompt's share of the utterance, as lo,hi.",
    )


def add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments specific to the bandwidth-extension side branch."""
    parser.add_argument(
        "--adapter-trainable",
        type=str,
        default="adapter",
        choices=["adapter", "shallow", "decoder", "all", "lora"],
        help="Which parameters the optimiser sees. 'adapter' trains only the "
        "branch and leaves every released tensor bit-identical, which is "
        "stage 2a and the default. 'shallow' also releases the decoder's 1-D "
        "parameters and its two band-facing projections, about 0.4%% of it, so "
        "the pretrained velocity field is adjusted rather than retrained. "
        "'decoder' releases the whole flow-matching decoder, which is stage 2b. "
        "'all' releases everything and exists to measure what the restraint "
        "costs. 'lora' trains the branch and the decoder's LoRA factors, which "
        "are gated off on a full-band prompt; needs --lora-rank.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Rank of the low-rank update on the decoder's in/out projections, "
        "applied only to utterances whose prompt is missing a band. 0 installs "
        "none. Overrides the model config.",
    )

    parser.add_argument(
        "--recon-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the spectrogram reconstruction loss, an L1 error on the "
        "endpoint estimate over the bins the band-limiting removed. This is the "
        "term that actually asks the model to invent a missing band; the "
        "flow-matching loss on its own only asks it to stay a text-to-speech "
        "model. Setting it to 0 is the ablation that isolates it.",
    )

    parser.add_argument(
        "--cutoff-min",
        type=float,
        default=1000.0,
        help="Lowest cutoff frequency, in Hz, of the band-limiting curriculum.",
    )

    parser.add_argument(
        "--cutoff-max",
        type=float,
        default=12000.0,
        help="Highest cutoff frequency, in Hz, of the band-limiting curriculum.",
    )

    parser.add_argument(
        "--curriculum-beta",
        type=float,
        default=4.0,
        help="Sharpness of AnyBand's Easy-to-Balanced cutoff distribution.",
    )

    parser.add_argument(
        "--curriculum-rho",
        type=float,
        default=0.7,
        help="Fraction of --curriculum-steps over which the cutoff distribution "
        "anneals to uniform. 0 samples cutoffs uniformly throughout, which is "
        "what the low-learning-rate joint stage uses.",
    )

    parser.add_argument(
        "--full-band-prob",
        type=float,
        default=0.2,
        help="Fraction of each batch left entirely un-band-limited, so the "
        "original full-band condition of the pretrained backbone is preserved "
        "and the gate sees band_frac=0 during training.",
    )

    parser.add_argument(
        "--use-speaker-cond",
        type=str2bool,
        default=None,
        help="Give the branch the prompt's per-bin mean and standard deviation, "
        "pooled over the prompt frames of the band-limited condition mel, as an "
        "extra conditioning stream. Overrides the model config; None keeps "
        "whatever the config says.",
    )

    parser.add_argument(
        "--variance-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the dispersion floor: the batch's per-bin standard "
        "deviation of the written band, charged only where the prediction is "
        "less varied than the target. It is the term that opposes writing every "
        "speaker's average high band, which both other losses accept. Always "
        "computed and logged, so a run at 0 measures its magnitude before a run "
        "spends gradient on it.",
    )

    parser.add_argument(
        "--use-rate-cond",
        type=str2bool,
        default=None,
        help="Give the branch the cutoff frequency as an embedded number rather "
        "than only as the shape of the band mask. Overrides the model config.",
    )
    parser.add_argument(
        "--mask-xt",
        type=str2bool,
        default=None,
        help="Hide the bins above the cutoff in the flow state from the "
        "network, so the branch has to synthesise them rather than copy them "
        "out of the training target. Overrides the model config.",
    )
    parser.add_argument(
        "--steer-all-bins",
        type=str2bool,
        default=None,
        help="Let the branch's velocity write every mel bin instead of only "
        "the ones the prompt is missing, so it can steer the low band its own "
        "writes disturb. Still inert on a full-band prompt. Incompatible with "
        "inference's --project-low-band. Overrides the model config.",
    )
    parser.add_argument(
        "--resample-degrade",
        type=str2bool,
        default=False,
        help="Band-limit by really resampling the waveform and re-extracting "
        "the mel, instead of pushing the mel bins above the cutoff to the "
        "empty-bin floor. The floor is 8.7 log-units away from what a resampled "
        "file actually produces, and the evaluation prompts are resampled "
        "files, so this removes a train/test mismatch. Requires audio in the "
        "batch, which it switches on.",
    )
    parser.add_argument(
        "--degrade-variants",
        type=str2bool,
        default=False,
        help="With --resample-degrade, vary the downsampler: half the batches' "
        "rate groups use the evaluation pipeline's exact resampler, the rest a "
        "different anti-alias filter, plus an optional early roll-off and a "
        "small noise floor. For deployment on devices whose front end is "
        "unknown.",
    )
    # Charging the branch for what its writes do outside the band it writes.
    #
    # The residual is added to a velocity, so the solver carries it into the
    # next state and the frozen backbone -- which reads the whole spectrum --
    # answers in every bin, including the ones the branch is masked out of.
    # That off-diagonal response is where the speaker similarity goes, and no
    # earlier arm had a gradient into it. See ``forward``'s drift block.
    parser.add_argument("--multistep-loss-weight", type=float, default=0.0)
    parser.add_argument("--multistep-steps", type=int, default=3)
    parser.add_argument("--multistep-max-utts", type=int, default=1)

    parser.add_argument(
        "--identity-rollout-steps",
        type=int,
        default=0,
        help="Number of solver steps in the identity rollout, which should be "
        "the number inference uses. Zero switches the rollout off. On a rollout "
        "batch the model runs the deployed solve on one utterance and the only "
        "term charged is the speaker identity loss on that sample; the "
        "flow-matching, reconstruction, band and envelope terms are not "
        "computed at all, so they are neither trained nor logged on it.",
    )
    parser.add_argument(
        "--identity-rollout-prob",
        type=float,
        default=0.0,
        help="Fraction of training batches that are rollout batches. The rest "
        "are ordinary flow-matching batches, unchanged.",
    )
    parser.add_argument(
        "--identity-rollout-guidance",
        type=float,
        default=0.5,
        help="Classifier-free guidance scale inside the rollout. Must match "
        "inference: guidance is part of the generator, not a decoding "
        "preference, and a rollout run at a different scale trains a model that "
        "is never deployed.",
    )
    parser.add_argument(
        "--identity-rollout-t-shift",
        type=float,
        default=0.5,
        help="Timestep shift inside the rollout. Must match inference, for the "
        "same reason as --identity-rollout-guidance.",
    )
    parser.add_argument(
        "--identity-rollout-max-seconds",
        type=float,
        default=8.0,
        help="Longest utterance a rollout may be run on. The rollout costs "
        "--identity-rollout-steps decoder forwards at twice the batch, so its "
        "cost is the utterance's length; this caps that, and the utterance is "
        "then drawn uniformly from those that fit rather than always being the "
        "shortest one, which would point identity supervision at the spans "
        "carrying the least speaker evidence. Zero removes the cap.",
    )
    parser.add_argument(
        "--identity-rollout-accumulate",
        type=str2bool,
        default=True,
        help="Fold a rollout batch's gradient into the next ordinary batch's "
        "update instead of stepping on it alone. This is not a convenience: "
        "ScaledAdam substitutes zeros for a missing gradient and applies the "
        "momentum step to every parameter it owns, so an identity-only update "
        "would still move the backbone along its last flow-matching direction, "
        "which is exactly the exclusion the rollout is built to keep. Setting "
        "this False restores separate identity-only updates and that drift.",
    )
    parser.add_argument(
        "--identity-rollout-weight",
        type=float,
        default=0.0,
        help="Weight of the speaker identity loss on rollout batches. Separate "
        "from --identity-loss-weight, which charges the same loss on the "
        "teacher-forced endpoint estimate of an ordinary batch: the two are "
        "computed on different quantities and their gradient scales differ, so "
        "one number cannot serve both.",
    )
    parser.add_argument(
        "--identity-rollout-low-band",
        type=str2bool,
        default=False,
        help="Score the rollout's identity on the band the prompt already has. "
        "The rollout's missing bins are replaced by the target's before "
        "vocoding, so the two signals differ only below the cutoff and the "
        "term charges drift in the low band's trajectory, not the invented "
        "band. Measured on a resample-degraded run: SIM fell as far on outputs "
        "low-passed at 4 kHz as on full ones, while the low-band L2 against "
        "the reference solve did not move.",
    )
    parser.add_argument(
        "--rollout-velocity-low-weight",
        type=float,
        default=0.0,
        help="Weight of a running cost along the rollout's solve, after "
        "VGG-Flow: the squared gap between the adapted and the stock velocity "
        "at the same state and with the same degraded prompt, over the observed "
        "bins of the generated span. Zero when the observed band does not "
        "drift, unlike --rollout-low-weight, whose clean-prompt reference puts "
        "a floor under it.",
    )
    parser.add_argument(
        "--rollout-velocity-low-steps",
        type=int,
        default=16,
        help="How many solver steps, drawn uniformly, carry the running cost. "
        "Each one in the ungraphed prefix costs one guided forward's "
        "activations and one stock forward without a graph.",
    )
    parser.add_argument(
        "--identity-rollout-grad-steps",
        type=int,
        default=1,
        help="How many of the rollout's last solver steps carry a graph "
        "(DRaFT-K). The solve always starts from noise at t = 0; the steps "
        "before these run without one. Above 1 the frozen backbone's "
        "activations are kept on those steps too, so the branch is credited "
        "for what its residual does to the backbone's later answers; memory "
        "grows by one guided backbone forward per step.",
    )
    parser.add_argument(
        "--rollout-low-weight",
        type=float,
        default=0.0,
        help="Weight of the L1 between the rollout's sample and the frozen "
        "model's own solve from the same noise with the clean prompt, over the "
        "bins the prompt observed. Nonzero with --rollout-high-weight or alone "
        "turns that reference solve on, which costs another "
        "--identity-rollout-steps guided forwards without a graph, and needs "
        "--adapter-trainable adapter so the backbone stays the frozen model.",
    )
    parser.add_argument(
        "--rollout-high-weight",
        type=float,
        default=0.0,
        help="As --rollout-low-weight, over the bins the prompt is missing.",
    )
    parser.add_argument(
        "--rollout-band-level-weight",
        type=float,
        default=0.0,
        help="Weight of the squared difference, per missing bin, between the "
        "rollout sample's mean log-mel over the generated span and the ground "
        "truth's. Timing drops out of a mean, so this compares against the real "
        "recording rather than the frozen model's solve, whose high band sits "
        "under it. Needs no reference solve.",
    )
    parser.add_argument(
        "--rollout-band-level-underfill",
        type=str2bool,
        default=False,
        help="Charge the band-level term only on bins where the sample is "
        "darker than the recording. The rollout L2 already covers a band that "
        "is too loud; what it rewards is a band that is uniformly too dark.",
    )
    parser.add_argument(
        "--rollout-cutoffs",
        type=str,
        default="",
        help="Comma-separated cutoffs in Hz, e.g. 4000,8000. When set, every "
        "rollout batch is band-limited at one of them, drawn uniformly per "
        "batch, instead of at the curriculum's cutoff, and its rollout terms "
        "are also logged per cutoff. Empty keeps the curriculum's draw.",
    )
    parser.add_argument(
        "--rollout-band-level-weights",
        type=str,
        default="",
        help="Comma-separated band-level weights, one per --rollout-cutoffs "
        "entry, replacing --rollout-band-level-weight for rollouts at that "
        "cutoff. A band that is darker at one cutoff than another is charged "
        "harder there without leaning on the other.",
    )
    parser.add_argument(
        "--rollout-distance",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="Distance between the rollout's sample and the reference solve. "
        "The reference is one deterministic solve, not a distribution over "
        "plausible bands, so the conditional-median argument for L1 does not "
        "apply to it.",
    )
    parser.add_argument(
        "--identity-rollout-start-step",
        type=int,
        default=0,
        help="Training batch at which rollout batches begin. Before it every "
        "batch is an ordinary flow-matching batch, so one run can grow a "
        "branch on the flow path first and then continue it on its own solve.",
    )
    parser.add_argument(
        "--drift-loss-weight",
        type=float,
        default=0.0,
        help="Weight on the low-band component of the backbone's response to "
        "the branch's own residual. 0 skips the term and its extra backbone "
        "forward entirely.",
    )
    parser.add_argument(
        "--drift-dt",
        type=float,
        default=0.5,
        help="Finite-difference step the residual is probed at. The backbone's "
        "low-band response is linear in the perturbation -- measured flat to "
        "within 1%% from 1/16 to 2 in fp32 -- so this trades no accuracy and is "
        "set by --use-fp16 instead: at 1/16 the difference is smaller than "
        "fp16's spacing at |v|~1 and rounding inflates the term 2.9x, which "
        "falls to 1.02x at 0.5.",
    )

    parser.add_argument(
        "--drift-normalize",
        type=str2bool,
        default=False,
        help="Divide the drift term by the mean square residual over the "
        "written band, making it the gain along the residual rather than the "
        "response's size. The unnormalised term is quadratic in the residual, "
        "so writing less lowers it as surely as aiming better, and writing "
        "less is the cheaper move: at matched steps the termed branch drove "
        "2.1-2.3x weaker than the untermed one and reached +0.76 dB of band "
        "energy against its +11.6 dB. Normalising is invariant to the "
        "residual's scale, so only its direction is charged: with the "
        "denominator attached the radial part of the gradient is exactly "
        "zero, so descent rotates the residual without lengthening or "
        "shortening it. The denominator is floored at 1e-4 rather than "
        "detached -- detaching restores a radial component pointing along "
        "+r, i.e. a brake with an adaptive weight, while the floor only "
        "stops 1/|r|^2 diverging if the branch collapses.",
    )

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Size of the Euler grid the training state is rolled out on. 0 "
        "keeps the flow-matching path, which is the default and every arm up "
        "to here. Above 0, the state handed to the loss is produced by running "
        "the solver from noise under no_grad, so the branch conditions on its "
        "own accumulated writes instead of on a state that already carries the "
        "target's high band. The rollout is detached -- no gradient flows "
        "through the solve -- so the cost is k extra forwards of wall clock and "
        "one activation peak of memory, not k.",
    )

    parser.add_argument(
        "--rollout-k",
        type=int,
        default=-1,
        help="How many of those steps to take before the loss is charged. -1 "
        "draws it uniformly from 1..rollout_steps-1 each batch, which is what a "
        "training arm wants; a fixed value pins the state to one point on the "
        "trajectory, which is what a diagnostic wants. Capped at "
        "rollout_steps-1 so the velocity target does not divide by zero.",
    )

    parser.add_argument(
        "--rollout-prob",
        type=float,
        default=1.0,
        help="Fraction of batches that use a rolled-out state; the rest use the "
        "flow-matching path. Below 1.0 this is scheduled sampling: a branch "
        "trained on self-generated states alone can settle into writing nothing "
        "-- the state then has nothing either, so the target is unreachable and "
        "the gradient is poor, which is how the earlier rollout arm collapsed "
        "the written band to +1.9 dB over stock.",
    )

    parser.add_argument(
        "--rollout-decouple",
        type=str2bool,
        default=False,
        help="""Roll the whole solve out with the branch decoupled: it reads the
        backbone's hidden state at every step and its writes never re-enter what
        the backbone reads. The low band is then the frozen model's exactly, and
        the losses score the sample the solver returns rather than a one-step
        estimate. Requires --rollout-steps. fm_loss is not defined in this mode
        and reads zero.""",
    )
    parser.add_argument(
        "--band-fill",
        type=str2bool,
        default=False,
        help="""Run the full solve with the branch reading the backbone's hidden
        state at every step and carrying its own state across them, then take
        its final read-out as the missing band and add it to the solved mel.
        The branch predicts no velocity and nothing it writes is integrated, so
        the dose and tilt knobs have nothing to correct. Requires
        --rollout-steps. fm_loss is not defined in this mode and reads zero.""",
    )
    parser.add_argument(
        "--band-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the multi-resolution band loss. The reconstruction "
        "term charges the invented band element by element, and because most "
        "of that band is not predictable from the low band its L1 optimum is a "
        "shrunken answer -- measured per octave against ground truth at a 4 kHz "
        "cutoff, the branch writes the right shape 9-10 dB too quietly. This "
        "term pools the error over tiles of frames and bins before taking the "
        "absolute value, so only each tile's net level offset is charged and "
        "the unpredictable fine structure is not. Charged on the near tier "
        "only -- see --band-tier-hz. Default 0.0 leaves the objective "
        "unchanged.",
    )

    parser.add_argument(
        "--band-tier-hz",
        type=float,
        default=4000.0,
        help="Where the written band splits into a near tier, whose level a "
        "regression can honestly target, and a far tier, where it cannot. "
        "Measured by ridge regression from the whole observed band with +-43 "
        "ms of context, the per-frame envelope of the first 4 kHz above the "
        "cutoff is 76%% predictable and everything beyond it 50%%. The split "
        "follows the cutoff, which is drawn per utterance.",
    )

    parser.add_argument(
        "--envelope-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the time-aligned band envelope loss. Every other term "
        "in this objective is satisfied by a band that is right on average: "
        "the flow-matching and reconstruction terms charge one element at a "
        "time, where most of the content is texture the conditioning does not "
        "determine, and the band loss charges a tile mean, which a band that "
        "is on at the wrong times matches exactly. This term reduces the tier "
        "to one number per frame and charges that against the truth frame by "
        "frame, so energy in a frame that should be silent costs and silence "
        "in a frame that should carry a fricative costs. The target is a "
        "legitimate one: the envelope is 50-76%% predictable from the observed "
        "band where the texture is not, so its L1 has a narrow distribution to "
        "hedge over. Default 0.0 leaves the objective unchanged.",
    )

    parser.add_argument(
        "--envelope-delta-weight",
        type=float,
        default=0.0,
        help="Weight of the frame-to-frame difference of the band envelope. "
        "This is where a veil is weakest: a band that is on all the time has a "
        "delta of zero everywhere, while speech steps at every fricative "
        "onset. Complements --envelope-loss-weight, which a slowly drifting "
        "offset can still partly satisfy.",
    )

    parser.add_argument(
        "--shape-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the per-frame normalised spectral shape of each band "
        "tier: the L1 between prediction and truth after each frame's mean "
        "log-mel over the tier is subtracted from both. It charges where the "
        "energy sits within the band independently of how much there is, "
        "which --envelope-loss-weight already charges. Same tiers and the same "
        "--envelope-t-min gate. Default 0.0 leaves the objective unchanged.",
    )

    parser.add_argument(
        "--envelope-t-min",
        type=float,
        default=0.5,
        help="Only charge the envelope terms on utterances at least this far "
        "along the flow. Near t=0 the endpoint estimate is mostly noise and "
        "its envelope is the envelope of noise, which says nothing about when "
        "the band should be loud. Same argument and default as "
        "--adapter-gan-t-min.",
    )

    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="A separate frozen copy of the released ZipVoice checkpoint, e.g. "
        "download/zipvoice/model.pt, used as the clean-prompt teacher. It has "
        "to be a second checkpoint and not this model under no_grad: "
        "--adapter-trainable shallow edits the backbone, so a self-teacher is "
        "a target that drifts with the student and can be matched by moving it.",
    )

    parser.add_argument(
        "--teacher-loss-weight",
        type=float,
        default=0.0,
        help="Weight on the squared error between the student's velocity and "
        "the teacher's, where the teacher is the released model reading the "
        "*clean* prompt at the same x_t, t, text and lengths. Charged over all "
        "frequencies of the generated, non-padding region, excluding the "
        "prompt. Measured on the evaluation set, the teacher's 8-12 kHz band "
        "sits 13.2 dB above what the same weights produce from an 8 kHz-limited "
        "prompt and 31.6 dB above the 4 kHz-limited case, so this is the "
        "behaviour the band losses were describing indirectly. Requires "
        "--teacher-checkpoint.",
    )

    parser.add_argument(
        "--use-pretrained-spk",
        type=str2bool,
        default=None,
        help="Condition the branch on the prompt's embedding under the frozen "
        "pretrained speaker encoder, which is also what the perceptual losses "
        "use. Requires --speaker-checkpoint. Overrides the model config.",
    )

    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=30000,
        help="Total number of steps the curriculum is scheduled against.",
    )

    # The adversarial terms. Off unless a start step is given, so the recipe is
    # unchanged by default and the discriminators are not even constructed.
    #
    # What they are for: the two base terms are both minimised by a conditional
    # mean, and the dispersion floor only asks that the written bands differ
    # from each other -- it is permutation-invariant, so swapping two speakers'
    # high bands scores perfectly. A discriminator scores whether *this* band is
    # a plausible continuation of *this* observed low band, which is the first
    # term in the objective that a speaker-generic band loses on.
    parser.add_argument(
        "--adapter-gan-start-step",
        type=int,
        default=-1,
        help="Enable the discriminators once batch_idx_train reaches this "
        "value. A negative value disables them for the whole run, which is the "
        "default and leaves the objective exactly as it was.",
    )

    parser.add_argument(
        "--adapter-adv-loss-weight",
        type=float,
        default=0.1,
        help="Weight of each discriminator view's hinge adversarial loss on the "
        "endpoint estimate.",
    )

    parser.add_argument(
        "--adapter-feat-match-loss-weight",
        type=float,
        default=2.0,
        help="Weight of each discriminator view's feature-matching loss.",
    )

    parser.add_argument(
        "--adapter-disc-lr",
        type=float,
        default=2e-4,
        help="AdamW learning rate of the discriminators. Plain AdamW rather "
        "than ScaledAdam: the per-tensor scale clipping is a poor fit for a "
        "discriminator, and these parameters never enter the merged model.",
    )

    parser.add_argument(
        "--adapter-gan-t-min",
        type=float,
        default=0.5,
        help="Only utterances with flow time at least this are discriminated. "
        "Below it the endpoint estimate is largely noise, and a discriminator "
        "trained to call noise fake learns the noise level rather than the "
        "band. Same argument as --perceptual-t-min, kept separate because the "
        "two terms can be weighted and gated independently.",
    )

    parser.add_argument(
        "--adapter-gan-tier",
        type=str,
        default="all",
        choices=["all", "far"],
        help="Which part of the written band the critics may see as generated. "
        "'all' is the whole band and the previous behaviour. 'far' confines "
        "them to bins more than --band-tier-hz above the cutoff, leaving the "
        "near tier spliced in from the truth as context. The near tier has a "
        "regression target that works -- its envelope already swings correctly "
        "and only its level is off -- so the critics are not needed there, and "
        "a critic has a fixed and measurably small amount of leverage to "
        "spend. This spends it where nothing else can reach.",
    )

    # The perceptual terms. Both default to 0, so the recipe is the two-term
    # objective unless a weight is asked for, and the module that computes them
    # is not even constructed.
    parser.add_argument(
        "--identity-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the speaker identity loss: one minus the cosine "
        "similarity between frozen speaker embeddings of the vocoded endpoint "
        "estimate and the vocoded target. Neither of the other two terms carries "
        "speaker identity -- flow matching scores the frozen backbone's velocity "
        "and the L1 is minimised by a speaker-generic high band -- so without "
        "this the branch invents a plausible band belonging to nobody.",
    )

    parser.add_argument(
        "--fmatch-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the deep feature matching loss: L1 between the speaker "
        "encoder's intermediate activations for the two vocoded signals, each "
        "stage normalised by the reference's own scale. This is a perceptual "
        "distance, not a discriminator: no GAN and no new trainable parameters.",
    )

    parser.add_argument(
        "--perceptual-t-min",
        type=float,
        default=0.5,
        help="Only utterances with flow time at least this contribute to the "
        "perceptual terms. Below it the endpoint estimate is largely noise and "
        "the distance between two embeddings of noise is not a speaker signal.",
    )

    parser.add_argument(
        "--perceptual-max-utts",
        type=int,
        default=4,
        help="Cap on utterances per step that go through the vocoder and the "
        "speaker encoder. This is what bounds the step time cost, since both run "
        "forward and backward on audio.",
    )

    parser.add_argument(
        "--perceptual-start-step",
        type=int,
        default=0,
        help="Delay the perceptual terms until this step, for a warm start.",
    )

    parser.add_argument(
        "--speaker-encoder-checkpoint",
        type=str,
        default="",
        help="Path to the pyannote WeSpeaker ResNet34 checkpoint used for the "
        "identity and feature matching losses. Deliberately not the WavLM-ECAPA "
        "the SIM-o evaluation uses: with disjoint architectures and training "
        "data a reported gain is transfer, not the objective scoring itself.",
    )

    parser.add_argument(
        "--vocos-local-path",
        type=str,
        default="",
        help="Directory holding the Vocos config.yaml and pytorch_model.bin. "
        "Empty downloads charactr/vocos-mel-24khz.",
    )


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
            "log_interval": 5,
            "reset_interval": 200,
            "env_info": get_env_info(),
        }
    )

    return params


def gan_enabled(params: AttributeDict) -> bool:
    """Whether the discriminators take part at the current training step."""
    return (
        params.adapter_gan_start_step >= 0
        and params.batch_idx_train >= params.adapter_gan_start_step
    )


def rollout_reference(params: AttributeDict) -> bool:
    """Whether rollout batches are scored against the frozen model's solve."""
    return params.rollout_low_weight > 0.0 or params.rollout_high_weight > 0.0


def rollout_cutoff_weights(params: AttributeDict) -> Dict[float, float]:
    """The fixed rollout cutoffs, each mapped to its band-level weight.

    Empty when rollouts take the curriculum's cutoff. Without per-cutoff
    weights every cutoff gets --rollout-band-level-weight.
    """
    if not params.rollout_cutoffs:
        return {}
    cutoffs = [float(c) for c in params.rollout_cutoffs.split(",")]
    if params.rollout_band_level_weights:
        weights = [float(w) for w in params.rollout_band_level_weights.split(",")]
        if len(weights) != len(cutoffs):
            raise ValueError(
                "--rollout-band-level-weights needs one weight per "
                "--rollout-cutoffs entry"
            )
    else:
        weights = [params.rollout_band_level_weight] * len(cutoffs)
    return dict(zip(cutoffs, weights))


def rollout_supervised(params: AttributeDict) -> bool:
    """Whether rollout batches carry a loss that needs no speaker encoder."""
    return (
        rollout_reference(params)
        or params.rollout_band_level_weight > 0.0
        or any(w > 0.0 for w in rollout_cutoff_weights(params).values())
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


def checkpoint_extra_states(
    discriminators: Optional[Union[nn.Module, DDP]],
    disc_optimizer: Optional[Optimizer],
    dmd: Optional[DMDController] = None,
) -> Optional[dict]:
    """Everything trained outside the model: discriminators and the DMD fake
    score. Both stay out of the model's state dict so the checkpoint still loads
    as an ordinary ``ZipVoiceAdapter``."""
    states = discriminator_states(discriminators, disc_optimizer) or {}
    if dmd is not None:
        states["dmd"] = dmd.state_dict()
    return states or None


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
        f0=None,
        voiced=None,
    )


def compute_disc_loss(
    params: AttributeDict,
    discriminators: Union[nn.Module, DDP],
    ctx: dict,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Hinge discriminator loss on the target mel (real) against the branch's
    endpoint estimate (fake, detached so no gradient reaches the branch).

    Real and fake go through in a single batched forward: DDP's reducer only
    tolerates one forward per backward, and none of the discriminator stacks
    carry batch statistics (they use weight norm), so batching them together is
    numerically identical to two separate passes.
    """
    real = ctx["features"]
    fake = ctx["x1_hat"].detach()
    batch_size = real.size(0)

    paired_ctx = dict(ctx)
    for key in ("band_mask", "cutoff_hz", "frame_mask"):
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
    loss = torch.zeros((), device=real.device)
    for view in real_logits:
        view_loss = hinge_d_loss(real_logits[view], fake_logits[view])
        loss = loss + view_loss
        info[f"d_{view}"] = view_loss.detach().cpu().item() * ctx["num_frames"]
    info["d_loss"] = loss.detach().cpu().item() * ctx["num_frames"]
    return loss, info


def gan_context(
    params: AttributeDict,
    aux: dict,
    num_frames: int,
    centers: Tensor,
) -> Optional[dict]:
    """The subset of the batch the discriminators see, or None if it is empty.

    Only utterances far enough along the flow qualify, so the tensors are sliced
    rather than masked -- a discriminator has no per-utterance weight, and
    feeding it near-noise endpoints with a zero weight is not available.
    """
    # Which bins the critics are allowed to call generated. Under
    # ``--adapter-gan-tier far`` that is only the part of the band far enough
    # above the cutoff to have no usable regression target; the near tier is
    # then spliced in from the truth alongside the low band, so it serves as
    # context and admits no gradient. The near tier's envelope already swings
    # correctly and only its level is wrong, which ``band_loss`` corrects
    # directly, so a critic there would be spending scarce leverage on a
    # problem already solved.
    write_mask = aux["band_mask"]
    if params.adapter_gan_tier == "far":
        write_mask = write_mask * aux["far_tier"]
    # Draws with nothing left to invent have real and fake identical after the
    # splice below: a pair the critic cannot lose on, which only dilutes its
    # training signal. They are dropped here rather than fed. Under the far
    # tier this also drops utterances whose cutoff is high enough that the tier
    # is empty, which a full-band test would have let through.
    missing = (write_mask > 0).flatten(1).any(dim=1)
    keep = (
        ((aux["t"] >= params.adapter_gan_t_min) & missing).nonzero().flatten()
    )
    if keep.numel() == 0:
        return None
    features = aux["features"][keep]
    band_mask = write_mask[keep]
    # The critics judge only the band the branch invents. Below the cutoff the
    # fake carries the real mel verbatim, so real and fake are bit-identical
    # there and no adversarial or feature-matching gradient can reach the low
    # band -- which is where the identity lives, and where a full-band critic
    # was found to spend its gradient (2-3 kHz and 3-4 kHz both brightened by
    # 1-2 dB over the GAN stage, costing 0.065-0.094 SIM-o). Splicing rather
    # than zeroing keeps the low band as context, which CrossBandDiscriminator
    # needs for its reference envelope and HarmonicDiscriminator for its
    # observed partials. The mask is a raised-cosine ramp, not a brick wall, so
    # this is a blend rather than a select: the gradient each bin admits is
    # exactly that bin's mask weight, and exactly zero where nothing is missing.
    x1_hat = features + band_mask * (aux["x1_hat"][keep] - features)
    return {
        "features": features,
        "x1_hat": x1_hat,
        "band_mask": band_mask,
        "cutoff_hz": aux["cutoff_hz"][keep],
        "frame_mask": aux["frame_mask"][keep],
        "centers": centers,
        "num_frames": num_frames,
    }


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
    perceptual: Optional[nn.Module] = None,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
    audio: Optional[Tensor] = None,
    audio_lens: Optional[Tensor] = None,
    fbank: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
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
      perceptual:
        The frozen vocoder and speaker encoder that produce the identity and
        feature matching losses, or None when neither is weighted. It is passed
        in rather than held on the model or on ``params``: anything on the model
        would enter the checkpoint and the averaged model, and ``params`` is
        written into the checkpoint key by key (``utils/checkpoint.py:110``).
      discriminators:
        The discriminator set, or None when the adversarial terms are off.
      teacher:
        The frozen released model that supplies the clean-prompt velocity, or
        None when that term is off. Passed in for the same reason ``perceptual``
        is: it must not enter the checkpoint or the averaged model.

    Returns the loss, the metrics, and the discriminator context the caller
    needs for the discriminator's own step -- None when the GAN is off or no
    utterance in the batch cleared the flow-time gate.
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
    num_frames = features_lens.sum().item()

    # Waveform-domain band-limiting. The cutoff has to be drawn here rather than
    # left to the model, because the degradation needs it before the forward
    # pass; the model is then handed both, and draws nothing itself.
    #
    # Whether this batch is a rollout is drawn here too, and the model is told
    # the outcome with a probability of 0 or 1. With --rollout-cutoffs a
    # rollout batch is degraded at one of those cutoffs, and that has to be
    # known before the waveform is resampled.
    #
    # Off during validation. The rollout is 16 guided forwards for one
    # utterance, and what it reports is a gradient path, not a held-out number
    # worth spending that on.
    identity_rollout_steps = (
        params.identity_rollout_steps
        if is_training
        and params.batch_idx_train >= params.identity_rollout_start_step
        and (
            rollout_supervised(params)
            or (
                perceptual is not None
                and params.batch_idx_train >= params.perceptual_start_step
            )
        )
        else 0
    )
    is_rollout_draw = (
        identity_rollout_steps > 0
        and torch.rand(()).item() < params.identity_rollout_prob
    )
    cutoff_weights = rollout_cutoff_weights(params)
    cutoff_hz = None
    if is_rollout_draw and cutoff_weights:
        cutoffs = list(cutoff_weights)
        cutoff_hz = torch.full(
            (batch_size,),
            cutoffs[int(torch.randint(len(cutoffs), ()))],
            device=device,
        )
    masked_features = None
    degraded_audio = None
    if fbank is not None:
        inner = model.module if isinstance(model, DDP) else model
        if cutoff_hz is None:
            cutoff_hz = sample_cutoff(
                batch_size=features.size(0),
                step=params.batch_idx_train,
                total_steps=params.curriculum_steps,
                f_min=inner.cutoff_min,
                f_max=inner.cutoff_max,
                beta=inner.curriculum_beta,
                rho=inner.curriculum_rho,
                full_band_prob=inner.full_band_prob,
                device=device,
            )
        with torch.no_grad():
            masked_features, degraded_audio = resample_band_limit(
                audio=audio.to(device),
                audio_lens=audio_lens.to(device),
                cutoff_hz=cutoff_hz,
                fbank=fbank,
                num_frames=features.size(1),
                sampling_rate=params.sampling_rate,
                return_audio=True,
                variants=params.degrade_variants,
            )
            masked_features = masked_features.to(features.dtype) * params.feat_scale

    # Which signal the speaker encoder reads decides whether this conditioning
    # is a leak. When the degradation ran in the waveform domain, it reads the
    # degraded waveform -- the real band-limited recording, exactly what the
    # prompt is at inference -- and never ``batch["audio"]``, which still holds
    # the band the branch is supposed to invent and which the encoder would
    # hear, since it resamples to 16 kHz and a 4 kHz cutoff leaves real content
    # from 4 to 8 kHz. Without waveform degradation it falls back to vocoding
    # the band-limited mel, which is band-limited by construction.
    speaker_embedder = None
    if perceptual is not None:
        if degraded_audio is not None:
            speaker_embedder = (
                lambda condition, prompt_mask: perceptual.embed_audio(
                    degraded_audio, prompt_mask
                )
            )
        else:
            speaker_embedder = perceptual.embed_prompt

    with torch.set_grad_enabled(is_training):

        fm_loss, recon_loss, aux = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            condition_drop_ratio=params.condition_drop_ratio,
            cutoff_hz=cutoff_hz,
            curriculum_step=params.batch_idx_train,
            curriculum_total=params.curriculum_steps,
            recon_loss_weight=params.recon_loss_weight,
            envelope_t_min=params.envelope_t_min,
            masked_features=masked_features,
            speaker_embedder=speaker_embedder,
            # A callable rather than a module, so nothing about the teacher can
            # reach the checkpoint. Validation charges it too: whether the two
            # arms agree with the teacher is exactly what the run is measuring,
            # so it has to be reported on held-out batches as well.
            teacher=(
                (
                    lambda t, xt, text_condition, speech_condition, padding_mask: (
                        teacher.forward_fm_decoder(
                            t=t,
                            xt=xt,
                            text_condition=text_condition,
                            speech_condition=speech_condition,
                            padding_mask=padding_mask,
                        )
                    )
                )
                if teacher is not None
                else None
            ),
            # The extra backbone forward is a training cost with no bearing on
            # what validation reports, so it is skipped there.
            drift_dt=(
                params.drift_dt
                if is_training and params.drift_loss_weight > 0.0
                else 0.0
            ),
            drift_normalize=params.drift_normalize,
            rollout_steps=params.rollout_steps,
            rollout_k=params.rollout_k,
            rollout_prob=params.rollout_prob,
            rollout_decouple=params.rollout_decouple,
            band_fill=params.band_fill,
            multistep_steps=(params.multistep_steps if params.multistep_loss_weight > 0 else 0),
            multistep_max_utts=params.multistep_max_utts,
            # Drawn above: rand() < 1 always holds and rand() < 0 never does.
            identity_rollout_steps=identity_rollout_steps,
            identity_rollout_prob=1.0 if is_rollout_draw else 0.0,
            identity_rollout_guidance=params.identity_rollout_guidance,
            identity_rollout_t_shift=params.identity_rollout_t_shift,
            identity_rollout_max_frames=params.identity_rollout_max_frames,
            identity_rollout_grad_steps=params.identity_rollout_grad_steps,
            identity_rollout_reference=rollout_reference(params),
            identity_rollout_distance=params.rollout_distance,
            identity_rollout_underfill=params.rollout_band_level_underfill,
            identity_rollout_velocity_steps=(
                params.rollout_velocity_low_steps
                if params.rollout_velocity_low_weight > 0.0
                else 0
            ),
        )
        # Which kind of batch this was. The model has the last word: a drawn
        # rollout with no band-limited utterance falls back to an ordinary
        # batch.
        is_rollout = bool(aux["is_rollout"])
        rollout_cutoff = (
            float(aux["cutoff_hz"].reshape(-1)[0]) if is_rollout else 0.0
        )
        variance_loss = aux["variance_loss"]
        drift_loss = aux["drift_loss"]
        band_loss = aux["band_loss"]
        band_bias = aux["band_bias"]
        envelope_loss = aux["envelope_loss"]
        envelope_delta_loss = aux["envelope_delta_loss"]
        loss = (
            fm_loss
            + recon_loss
            + params.variance_loss_weight * variance_loss
            + params.drift_loss_weight * drift_loss
            + params.band_loss_weight * band_loss
            + params.envelope_loss_weight * envelope_loss
            + params.envelope_delta_weight * envelope_delta_loss
            + params.shape_loss_weight * aux["shape_loss"]
            + params.multistep_loss_weight * aux["multistep_loss"]
            + params.teacher_loss_weight * aux["teacher_loss"]
        )
        if is_rollout:
            loss = (
                loss
                + params.rollout_low_weight * aux["rollout_low_loss"]
                + params.rollout_high_weight * aux["rollout_high_loss"]
                + cutoff_weights.get(
                    rollout_cutoff, params.rollout_band_level_weight
                )
                * aux["rollout_band_level_loss"]
                + params.rollout_velocity_low_weight
                * aux["rollout_velocity_low_loss"]
            )

        identity_loss = torch.zeros((), device=device)
        fmatch_loss = torch.zeros((), device=device)
        num_perceptual = 0
        # The weights gate the forward, not just the sum. ``perceptual`` is now
        # also built for speaker conditioning alone, and in that case these two
        # losses are dead weight: backprop through the vocoder's ISTFT
        # overflows under --use-fp16 and drove the gradient scaler to 5.8e-11
        # and the loss to NaN within 50 steps, for terms multiplied by zero.
        if (
            perceptual is not None
            and (
                params.identity_loss_weight > 0.0
                or params.fmatch_loss_weight > 0.0
                or is_rollout
            )
            and params.batch_idx_train >= params.perceptual_start_step
        ):
            # Only utterances far enough along the flow contribute. Near t=0 the
            # endpoint estimate is mostly noise, and the distance between two
            # embeddings of noise says nothing about whether the branch is
            # writing the right speaker.
            keep = (aux["t"] >= params.perceptual_t_min).nonzero().flatten()
            keep = keep[: params.perceptual_max_utts]
            # Whether anything in this batch will actually be weighted. A
            # weight-zero rollout is the control for a weighted one: it draws
            # the same batches, spends the same RNG and runs the same solve,
            # and differs only in the update. Its identity value is still
            # computed and logged -- that is the counterfactual the comparison
            # needs -- but under ``no_grad``, so no vocoder graph is built for
            # a term that would be multiplied by zero.
            weighted = (
                params.identity_rollout_weight > 0.0
                if is_rollout
                else (
                    params.identity_loss_weight > 0.0
                    or params.fmatch_loss_weight > 0.0
                )
            )
            if keep.numel() > 0:
                # fp32: the vocoder's ISTFT overflows readily in half precision,
                # and the backward pass runs through it.
                with torch_autocast(dtype=torch.float16, enabled=False), (
                    contextlib.nullcontext() if weighted else torch.no_grad()
                ):
                    x1_hat = aux["x1_hat"]
                    if is_rollout and params.identity_rollout_low_band:
                        x1_hat = torch.where(
                            aux["band_mask"] > 0, aux["features"], x1_hat
                        )
                    identity_loss, fmatch_loss, num_perceptual = perceptual(
                        x1_hat[keep],
                        aux["features"][keep],
                        aux["span_starts"][keep],
                        aux["span_ends"][keep],
                    )
                # A rollout batch carries the identity term alone. Feature
                # matching is a second opinion from the same encoder and its
                # interaction with a sampled output is unmeasured; adding it
                # here would make an unattributable change to the objective at
                # the same moment the generator changed.
                if is_rollout:
                    if params.identity_rollout_weight > 0.0:
                        loss = loss + (
                            params.identity_rollout_weight * identity_loss
                        )
                else:
                    loss = (
                        loss
                        + params.identity_loss_weight * identity_loss
                        + params.fmatch_loss_weight * fmatch_loss
                    )

    inner = model.module if isinstance(model, DDP) else model
    ctx = None
    adv_terms = {}
    # Validation runs this too, under no-grad: the adversarial and
    # feature-matching terms are the only measure of what the GAN is actually
    # optimising, and without them valid_loss is not comparable to the training
    # loss it is meant to track.
    with torch.set_grad_enabled(is_training):
        if discriminators is not None and gan_enabled(params) and not is_rollout:
            ctx = gan_context(params, aux, num_frames, inner.mel_centers)
        if ctx is not None:
            # The unwrapped module: these forwards exist to push gradients back
            # into the branch, the gradients they leave on the discriminator
            # parameters are discarded before the discriminator step, and
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
                loss = loss + params.adapter_adv_loss_weight * adv
                loss = loss + params.adapter_feat_match_loss_weight * fmatch
                info_adv = adv.detach().cpu().item()
                info_fmatch = fmatch.detach().cpu().item()
                adv_terms[f"adv_{view}"] = info_adv
                adv_terms[f"fmatch_{view}"] = info_fmatch

            adv_terms["gan_utts"] = float(ctx["features"].size(0))

            if not is_training:
                # Training logs d_* from the discriminator's own step, which
                # validation never takes. Recomputing them from the forwards
                # just done costs nothing and keeps the key set the same on both
                # sides, so the two curves are comparable.
                d_total = 0.0
                for view in fake_logits:
                    d_view = hinge_d_loss(real_logits[view], fake_logits[view]).item()
                    d_total += d_view
                    adv_terms[f"d_{view}"] = d_view
                adv_terms["d_loss"] = d_total
                # Nothing downstream takes a discriminator step in validation.
                ctx = None

    # A weight-zero rollout batch is the one training case with no graph: its
    # only term is the identity loss and that term is multiplied by zero, so
    # the batch is deliberately a no-op on the parameters while still spending
    # the same draws, the same RNG and the same solve as the weighted run it
    # controls for.
    assert loss.requires_grad == (
        is_training
        and not (
            is_rollout
            and params.identity_rollout_weight <= 0.0
            and not rollout_supervised(params)
        )
    )

    info = MetricsTracker()
    info["frames"] = num_frames
    # Per-type denominators. ``MetricsTracker`` averages ``rollout_*`` keys over
    # ``rollout_frames`` and everything else over ``fm_frames``, so a rollout
    # batch cannot make the flow-matching average look better by contributing
    # frames and a zero, and an ordinary batch cannot do the same to the
    # identity average. A run with no rollout batches sets ``fm_frames`` equal
    # to ``frames`` on every batch, which is the behaviour it always had.
    info["fm_frames"] = 0 if is_rollout else num_frames
    info["rollout_frames"] = num_frames if is_rollout else 0
    if is_rollout:
        # Only the terms this batch actually computed. Logging the rest as
        # zeros would be logging the wrong quantity, not a missing one.
        info["rollout_loss"] = loss.detach().cpu().item() * num_frames
        info["rollout_identity_loss"] = (
            identity_loss.detach().cpu().item() * num_frames
        )
        info["rollout_low_loss"] = (
            aux["rollout_low_loss"].detach().cpu().item() * num_frames
        )
        info["rollout_high_loss"] = (
            aux["rollout_high_loss"].detach().cpu().item() * num_frames
        )
        info["rollout_band_level_loss"] = (
            aux["rollout_band_level_loss"].detach().cpu().item() * num_frames
        )
        info["rollout_velocity_low_loss"] = (
            aux["rollout_velocity_low_loss"].detach().cpu().item() * num_frames
        )
        info["rollout_band_bias"] = (
            aux["rollout_band_bias"].detach().cpu().item() * num_frames
        )
        # With fixed cutoffs, the same terms again under e.g. ``rollout4k_``,
        # averaged over that cutoff's own frames. The pooled average hides a
        # band that is right at one cutoff and dark at another.
        if rollout_cutoff in cutoff_weights:
            tag = "rollout%gk" % (rollout_cutoff / 1000)
            info[tag + "_frames"] = num_frames
            for term in (
                "low_loss",
                "high_loss",
                "band_level_loss",
                "band_bias",
                "velocity_low_loss",
            ):
                info[tag + "_" + term] = (
                    aux["rollout_" + term].detach().cpu().item() * num_frames
                )
        info["rollout_utts"] = num_perceptual * num_frames
        info["rollout_residual_rms"] = aux["residual_rms"].cpu().item() * num_frames
        return loss, info, ctx
    info["fm_loss"] = fm_loss.detach().cpu().item() * num_frames
    info["recon_loss"] = recon_loss.detach().cpu().item() * num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames
    # Logged unweighted, so the agreement with the teacher can be read off
    # directly, and separately so its share of ``loss`` is visible rather than
    # inferred: multiply by --teacher-loss-weight.
    info["teacher_loss"] = aux["teacher_loss"].detach().cpu().item() * num_frames
    # How hard the branch is pushing. The previous design's failure was a gate
    # pinned at its ceiling at every bandwidth, and it was invisible in a loss
    # shared with everything else; this is the one number that would show the
    # same thing here, so it is on the training curves rather than left to a
    # post-hoc probe.
    info["residual_rms"] = aux["residual_rms"].cpu().item() * num_frames
    # Logged whatever its weight, so a run that does not pay for it still says
    # how far the written band falls short of the data's own dispersion.
    info["variance_loss"] = variance_loss.detach().cpu().item() * num_frames
    # Zero on validation batches by construction; the training curve is the one
    # that says whether the branch is learning to stay out of the low band.
    info["drift_loss"] = drift_loss.detach().cpu().item() * num_frames
    info["multistep_loss"] = aux["multistep_loss"].detach().cpu().item() * num_frames
    info["multistep_drift"] = aux["multistep_drift"].detach().cpu().item() * num_frames
    # Logged whatever its weight. It is a level offset in the feature's own log
    # units, so a run that pays for it should show this falling while
    # ``residual_rms`` rises.
    info["band_loss"] = band_loss.detach().cpu().item() * num_frames
    info["band_bias"] = band_bias.detach().cpu().item() * num_frames
    # Logged whatever their weight, and the three that follow are the ones to
    # read. ``env_bias`` is the far tier's signed level error and ``env_swing``
    # its predicted spread against ``env_swing_true``. The distinction they draw
    # is the whole point of the term: a branch buying level by lifting its floor
    # closes ``env_bias`` with ``env_swing`` flat, and one learning to turn the
    # band on closes it with ``env_swing`` rising toward ``env_swing_true``.
    # The previous run did the first, and no number on its curves said so.
    info["envelope_loss"] = envelope_loss.detach().cpu().item() * num_frames
    info["envelope_delta_loss"] = (
        envelope_delta_loss.detach().cpu().item() * num_frames
    )
    info["shape_loss"] = aux["shape_loss"].detach().cpu().item() * num_frames
    info["env_bias"] = aux["env_bias"].detach().cpu().item() * num_frames
    info["env_swing"] = aux["env_swing"].detach().cpu().item() * num_frames
    info["env_swing_true"] = aux["env_swing_true"].detach().cpu().item() * num_frames
    if perceptual is not None:
        info["identity_loss"] = identity_loss.detach().cpu().item() * num_frames
        info["fmatch_loss"] = fmatch_loss.detach().cpu().item() * num_frames
        # How many utterances actually cleared the t gate and the length floor.
        # If this sits at zero the two terms above are zero for a reason that
        # has nothing to do with the branch having learned anything.
        info["perceptual_utts"] = num_perceptual * num_frames
    for name, value in adv_terms.items():
        info[name] = value * num_frames

    return loss, info, ctx


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    discriminators: Optional[Union[nn.Module, DDP]],
    disc_optimizer: Optional[Optimizer],
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    perceptual: Optional[nn.Module] = None,
    fbank: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
    dmd: Optional[DMDController] = None,
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
            rank=0,
            extra_states=checkpoint_extra_states(discriminators, disc_optimizer, dmd),
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
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                perceptual=perceptual,
                teacher=teacher,
                discriminators=discriminators,
                fbank=fbank,
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

        tokens, features, features_lens, audio, audio_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
            return_audio=True,
        ) if fbank is not None else (
            *prepare_input(
                params=params,
                batch=batch,
                device=device,
                return_tokens=True,
                return_feature=True,
            ),
            None,
            None,
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
                    perceptual=perceptual,
                    teacher=teacher,
                    discriminators=discriminators,
                    audio=audio,
                    audio_lens=audio_lens,
                    fbank=fbank,
                )

                if dmd is not None:
                    # The generator term on the solved sample. The ordinary
                    # flow-matching loss stays on, at --dmd-fm-weight, as the
                    # anchor to real data. Off-schedule batches return None and
                    # train exactly as before.
                    dmd_loss, dmd_stats = dmd.generator_loss(
                        model,
                        step=params.batch_idx_train,
                        tokens=tokens,
                        features=features,
                        features_lens=features_lens,
                        audio=audio,
                        audio_lens=audio_lens,
                        fbank=fbank,
                    )
                    loss = params.dmd_fm_weight * loss
                    if dmd_loss is not None:
                        loss = loss + dmd_loss
                        frames = dmd_stats.pop("frames")
                        loss_info["dmd_frames"] = frames
                        for name, value in dmd_stats.items():
                            loss_info[f"dmd_{name}"] = value * frames

                train_disc = ctx is not None
                if train_disc:
                    disc_loss, disc_info = compute_disc_loss(
                        params=params,
                        discriminators=discriminators,
                        ctx=ctx,
                    )
                    loss_info = loss_info + disc_info

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # A weight-zero rollout batch's loss is a constant: there is no
            # graph to walk, and the gradient it would contribute is exactly
            # zero. ``defer_update`` below still holds the update, so the
            # control's optimizer-step count matches the weighted run's.
            if loss.requires_grad:
                scaler.scale(loss).backward()

            # A rollout batch's gradient is identity only, and it reaches the
            # branch and the conditioning encoders alone -- the backbone's
            # parameters come back with ``grad`` None, which the smoke check
            # verified. Stepping here anyway would not preserve that: ScaledAdam
            # replaces a missing gradient with zeros and still applies the
            # momentum step, so the backbone would drift along its last
            # flow-matching direction on a batch that taught it nothing. Holding
            # the gradient until the next ordinary batch keeps the exclusion
            # exact and gives the backbone precisely the update flow matching
            # asked for, while the branch gets both terms in one step.
            defer_update = bool(loss_info.get("rollout_frames", 0)) and (
                params.identity_rollout_accumulate
            )

            if train_disc:
                # The generator's loss ran the discriminators and left gradients
                # on their parameters. Discard those before computing the
                # discriminators' own.
                disc_optimizer.zero_grad()
                scaler.scale(disc_loss).backward()

            if not defer_update:
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
                # ``set_to_none``: a parameter that receives no gradient on the
                # next batch is then visibly absent rather than carrying a stale
                # zero, which is what the rollout's backbone exclusion is read
                # off. It does not by itself keep ScaledAdam from stepping such
                # a parameter -- see ``defer_update``.
                optimizer.zero_grad(set_to_none=True)
                if train_disc:
                    disc_optimizer.zero_grad(set_to_none=True)

            if dmd is not None:
                # After the student's step, so the fake score is fitted to the
                # student as it now is. Its own optimiser and scaler; its loss
                # reaches only the fake LoRA.
                fake_info = dmd.fake_update(
                    model,
                    tokens=tokens,
                    features=features,
                    features_lens=features_lens,
                    audio=audio,
                    audio_lens=audio_lens,
                    fbank=fbank,
                    autocast=partial(
                        torch_autocast, dtype=torch.float16, enabled=params.use_fp16
                    ),
                )
                fake_metrics = MetricsTracker()
                fake_metrics["dmdfake_frames"] = fake_info["fake_frames"]
                fake_metrics["dmdfake_loss"] = (
                    fake_info["fake_loss"] * fake_info["fake_frames"]
                )
                loss_info = loss_info + fake_metrics
                tot_loss = tot_loss + fake_metrics
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
                rank=rank,
                extra_states=checkpoint_extra_states(discriminators, disc_optimizer, dmd),
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
                + (f"grad_scale: {scaler._scale.item()}, " if params.use_fp16 else "")
                # On the log line rather than only at validation, because with
                # --valid-interval set high the validation line never prints and
                # the only evidence of a memory problem is the OOM itself. Both
                # numbers, because they answer different questions: `alloc` is
                # what is live right now and rises only if something is being
                # retained across batches, while `peak` is the high-water mark
                # of the forward and rises with batch shape alone.
                + (
                    f"alloc: {torch.cuda.memory_allocated() // 1000000}MB, "
                    f"peak: {torch.cuda.max_memory_allocated() // 1000000}MB"
                    if torch.cuda.is_available()
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
    world_size: int = 1,
    perceptual: Optional[nn.Module] = None,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
    fbank: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    if discriminators is not None:
        discriminators.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        tokens, features, features_lens, audio, audio_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
            return_audio=True,
        ) if fbank is not None else (
            *prepare_input(
                params=params,
                batch=batch,
                device=device,
                return_tokens=True,
                return_feature=True,
            ),
            None,
            None,
        )

        loss, loss_info, _ = compute_fbank_loss(
            params=params,
            model=model,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            is_training=False,
            perceptual=perceptual,
            teacher=teacher,
            discriminators=discriminators,
            audio=audio,
            audio_lens=audio_lens,
            fbank=fbank,
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
    perceptual: Optional[nn.Module] = None,
    fbank: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        tokens, features, features_lens, audio, audio_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
            return_audio=True,
        ) if fbank is not None else (
            *prepare_input(
                params=params,
                batch=batch,
                device=device,
                return_tokens=True,
                return_feature=True,
            ),
            None,
            None,
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
                    perceptual=perceptual,
                    teacher=teacher,
                    audio=audio,
                    audio_lens=audio_lens,
                    fbank=fbank,
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
    if params.valid_interval is None:
        params.valid_interval = params.save_every_n
    # Set epoch to a large number to ignore it.
    if params.num_iters > 0:
        params.num_epochs = 1000000
    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    # Resolve the command-line stream overrides into the config before anything
    # else reads it, so the copy written to the experiment directory describes
    # the model that was actually trained. Model averaging and inference build
    # from that copy, and a checkpoint does not load strictly into a model
    # missing one of its streams.
    model_config["adapter"] = adapter_kwargs(
        model_config,
        use_speaker_cond=params.use_speaker_cond,
        use_rate_cond=params.use_rate_cond,
        use_pretrained_spk=params.use_pretrained_spk,
        mask_xt=params.mask_xt,
        steer_all_bins=params.steer_all_bins,
        lora_rank=params.lora_rank,
    )
    params.update(model_config["model"])
    params.update(model_config["feature"])

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

    if torch.cuda.is_available():
        params.device = torch.device("cuda", rank)
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    # Built here and kept out of the model on purpose: it holds a 13.5M-parameter
    # vocoder and a 6.6M-parameter speaker encoder, none of it trainable, and a
    # checkpoint has to stay the released tensors plus the branch so it can be
    # merged back onto the released model.
    perceptual = None
    if (
        params.identity_loss_weight > 0.0
        or params.fmatch_loss_weight > 0.0
        # A rollout scored only against the reference solve needs no speaker
        # encoder; one with no reference is the identity rollout, weighted or
        # its weight-zero control, and needs it either way.
        or (
            params.identity_rollout_steps > 0
            and (params.identity_rollout_weight > 0.0 or not rollout_supervised(params))
        )
        or model_config["adapter"].get("use_pretrained_spk", False)
    ):
        assert params.speaker_encoder_checkpoint, (
            "--identity-loss-weight, --fmatch-loss-weight, "
            "--identity-rollout-weight or --use-pretrained-spk is set but "
            "--speaker-encoder-checkpoint is empty"
        )
        perceptual = SpeakerPerceptual(
            speaker_checkpoint=params.speaker_encoder_checkpoint,
            vocos_local_path=params.vocos_local_path or None,
            feat_scale=params.feat_scale,
        ).to(params.device)
        logging.info(
            f"Perceptual losses on: identity {params.identity_loss_weight}, "
            f"fmatch {params.fmatch_loss_weight}, "
            f"t >= {params.perceptual_t_min}, "
            f"at most {params.perceptual_max_utts} utterances per step"
        )
        if params.identity_rollout_steps > 0 and params.identity_rollout_prob > 0:
            logging.info(
                f"Identity rollout on: weight {params.identity_rollout_weight}, "
                f"{params.identity_rollout_steps} steps at guidance "
                f"{params.identity_rollout_guidance} and t_shift "
                f"{params.identity_rollout_t_shift}, on "
                f"{params.identity_rollout_prob:.0%} of batches, utterances up "
                f"to {params.identity_rollout_max_seconds}s, "
                + (
                    "folded into the next ordinary update"
                    if params.identity_rollout_accumulate
                    else "stepped on their own"
                )
            )

    if perceptual is not None and model_config["adapter"].get(
        "use_pretrained_spk", False
    ):
        # The embedding width is a property of the checkpoint, not a number the
        # recipe gets to choose; the config's value is overwritten with what the
        # encoder actually returns so the branch's projection cannot be built
        # the wrong size.
        model_config["adapter"]["pretrained_spk_dim"] = perceptual.speaker_dim
        logging.info(f"Speaker conditioning width {perceptual.speaker_dim}")

    # Written only now, because the block above resolves the speaker width that
    # sizes the branch. Model averaging and inference build from this copy.
    with open(f"{params.exp_dir}/model.json", "w") as f:
        json.dump(model_config, f, indent=2)


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

    model = ZipVoiceAdapter(
        **model_config["model"],
        **model_config["adapter"],
        **tokenizer_config,
        feat_scale=params.feat_scale,
        cutoff_min=params.cutoff_min,
        cutoff_max=params.cutoff_max,
        curriculum_beta=params.curriculum_beta,
        curriculum_rho=params.curriculum_rho,
        full_band_prob=params.full_band_prob,
    )

    if params.checkpoint is not None and params.resume_from is None:
        logging.info(f"Loading pre-trained model from {params.checkpoint}")
        # Not strict: a released ZipVoice checkpoint has none of the new
        # modules, which is the whole point -- they start at their
        # initialisation, which is a no-op. Anything *unexpected* is still a
        # real error and is reported. load_checkpoint returns the checkpoint
        # dictionary rather than the key report, so the load is done directly
        # here to get hold of it.
        state = torch.load(
            params.checkpoint, map_location="cpu", weights_only=False
        )["model"]
        state = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"Unexpected tensors in checkpoint: {unexpected}"
        if missing:
            roots = sorted({name.split(".")[0] for name in missing})
            logging.info(f"Modules initialised fresh: {roots}")

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    # Built outside the model, like the perceptual module and for the same
    # reason: a checkpoint has to stay the released tensors plus the branch so it
    # can be merged back onto the released model. The discriminators ride in the
    # checkpoint's extra states instead, which the averaged model never sees.
    discriminators: Optional[nn.Module] = None
    if params.adapter_gan_start_step >= 0:
        discriminators = BweDiscriminators(
            n_mels=model_config["model"]["feat_dim"],
            # No harmonic view: it needs F0, and the only F0 available here
            # would come from the band-limited mel, which is the same signal the
            # branch is already conditioned on.
            use_harmonic=False,
            **model_config.get("adapter_disc", {}),
        )
        logging.info(
            "Number of discriminator parameters : "
            f"{sum(p.numel() for p in discriminators.parameters())}"
        )

    # The clean-prompt teacher. Built here rather than held on the model, for
    # the same reason the perceptual module and the discriminators are: a
    # checkpoint has to stay the released tensors plus the branch.
    #
    # It is loaded strictly. The student loads the same released weights
    # non-strictly because its new modules are legitimately absent, but the
    # teacher is meant to *be* the released model, so a missing or unexpected
    # tensor here means the wrong file was named, and a teacher that is
    # quietly part-initialised would teach something other than the measured
    # ceiling.
    teacher: Optional[nn.Module] = None
    if params.teacher_loss_weight > 0.0:
        assert params.teacher_checkpoint, (
            "--teacher-loss-weight needs --teacher-checkpoint. The student's "
            "own backbone cannot serve: --adapter-trainable shallow edits it, "
            "and no_grad on shared, changing weights is not a fixed teacher."
        )
        logging.info(f"Loading clean-prompt teacher from {params.teacher_checkpoint}")
        # Built under a preserved RNG state. Every tensor of it is overwritten
        # by the checkpoint a moment later, so its initialisation is discarded
        # -- but it draws from the global generator all the same, and the two
        # arms of a controlled comparison have to see the same ``noise``, the
        # same ``t`` and the same cutoff draws at every step. Without this the
        # arm that builds a teacher is running on a different random stream
        # than the arm that does not, and the comparison is not paired.
        rng_state = torch.get_rng_state()
        teacher = ZipVoice(**model_config["model"], **tokenizer_config)
        torch.set_rng_state(rng_state)
        state = torch.load(
            params.teacher_checkpoint, map_location="cpu", weights_only=False
        )["model"]
        state = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        teacher.load_state_dict(state, strict=True)
        teacher.eval()
        teacher.requires_grad_(False)

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    # Two ways in: --start-epoch names a completed epoch, --resume-from names an
    # arbitrary checkpoint. Everything downstream only cares that we are resuming.
    checkpoints = None
    if params.start_epoch > 1 or params.resume_from is not None:
        checkpoints = resume_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            filename=params.resume_from,
        )
        if params.resume_from is not None:
            # batch_idx_train came back from the checkpoint; the epoch loop has
            # to restart on the epoch that was in flight, not on epoch 1.
            params.start_epoch = checkpoints.get("cur_epoch", params.start_epoch)
            logging.info(
                f"Resuming from {params.resume_from} at epoch "
                f"{params.start_epoch}, batch_idx_train {params.batch_idx_train}"
            )
        if discriminators is not None and "discriminators" in checkpoints:
            logging.info("Loading discriminator state dict")
            discriminators.load_state_dict(checkpoints["discriminators"])

    # After the model_avg deepcopy, so that the averaged copy still carries every
    # tensor, and before the optimiser is built, because
    # get_parameter_groups_with_lrs drops parameters whose requires_grad is
    # False -- which is what keeps ScaledAdam's per-tensor scale away from the
    # released weights.
    # A loss hyperparameter, not architecture: it decides which bins two
    # training terms charge and nothing the checkpoint carries, so it is set on
    # the built model rather than passed through ``adapter_kwargs`` into
    # ``model.json``, where inference would have to keep agreeing with it.
    model.band_tier_hz = params.band_tier_hz

    # The cap reaches the model in frames, because that is what it can compare
    # against; seconds are the unit the flag is worth stating in.
    params.identity_rollout_max_frames = (
        int(round(params.identity_rollout_max_seconds / VocosFbank().frame_shift))
        if params.identity_rollout_max_seconds > 0
        else 0
    )

    # A branch that writes the low band is meant to correct a frozen backbone,
    # not to share the job with one that is being moved underneath it.
    assert not (
        model_config["adapter"].get("steer_all_bins", False)
        and params.adapter_trainable != "adapter"
    ), "--steer-all-bins trains the branch only; use --adapter-trainable adapter"
    model.set_trainable(params.adapter_trainable)
    trainable =[n for n, p in model.named_parameters() if p.requires_grad]
    logging.info(
        f"Stage '{params.adapter_trainable}': training "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"parameters in {len(trainable)} tensors, roots "
        f"{sorted({n.split('.')[0] for n in trainable})}"
    )

    model = model.to(params.device)
    if teacher is not None:
        teacher = teacher.to(params.device)

    # The mel extractor, when band-limiting is done by really resampling. It is
    # the same ``MelSpectrogram`` the precomputed features were made with, so
    # the observed band of the degraded mel agrees with them closely, but not
    # exactly: measured on 65 dev utterances, the below-cutoff L1 between the
    # two is 0.066 log-units at a 4 kHz cutoff and 0.050 at 8 kHz. Of that,
    # 0.008 is not band-limiting at all -- it is what a cutoff above Nyquist,
    # which removes nothing, still costs, because this recomputes the mel from
    # the waveform while the reference was read off disk. The rest is the
    # resampler's own passband ripple, about half a dB.
    fbank = None
    if params.resample_degrade:
        fbank = VocosFbank().fbank.to(params.device)
        logging.info("Band-limiting by waveform resampling, not by mel floor")

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
        # Plain AdamW rather than ScaledAdam: ScaledAdam's per-tensor scale
        # clipping is a poor fit for a GAN discriminator.
        disc_optimizer = torch.optim.AdamW(
            discriminators.parameters(),
            lr=params.adapter_disc_lr,
            betas=(0.8, 0.99),
        )
        if checkpoints is not None and checkpoints.get("disc_optimizer") is not None:
            logging.info("Loading discriminator optimizer state dict")
            disc_optimizer.load_state_dict(checkpoints["disc_optimizer"])
        logging.info(
            f"Adversarial terms on from step {params.adapter_gan_start_step}: "
            f"adv {params.adapter_adv_loss_weight}, "
            f"fmatch {params.adapter_feat_match_loss_weight}, "
            f"t >= {params.adapter_gan_t_min}"
        )

    assert params.lr_hours >= 0

    if params.finetune:
        scheduler = FixedLRScheduler(optimizer)
    elif params.lr_hours > 0:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_hours)
    else:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    scaler = create_grad_scaler(enabled=params.use_fp16)

    dmd: Optional[DMDController] = None
    if params.dmd_loss_weight > 0.0:
        inner = model.module if isinstance(model, DDP) else model
        assert world_size == 1, (
            "DMD calls the model outside DDP's forward and runs on one GPU"
        )
        assert params.adapter_trainable == "lora", (
            "The one-pass DMD student needs --adapter-trainable lora: without "
            "the decoder LoRA nothing can learn to read the branch's writes"
        )
        assert params.identity_rollout_prob <= 0.0, (
            "DMD and the identity rollout are alternative generator losses"
        )
        assert inner.pretrained_spk_proj is None, (
            "DMD does not pass speaker embeddings"
        )
        dmd = DMDController(
            inner,
            rank=params.dmd_fake_rank,
            lr=params.dmd_fake_lr,
            loss_weight=params.dmd_loss_weight,
            start_step=params.dmd_start_step,
            gen_interval=params.dmd_gen_interval,
            num_step=params.dmd_num_step,
            guidance_scale=params.dmd_guidance,
            real_guidance=params.dmd_real_guidance,
            t_shift=params.dmd_t_shift,
            grad_steps=params.dmd_grad_steps,
            t_min=params.dmd_t_min,
            t_max=params.dmd_t_max,
            num_utts=params.dmd_utts,
            max_frames=(
                int(round(params.dmd_max_seconds / VocosFbank().frame_shift))
                if params.dmd_max_seconds > 0
                else 0
            ),
            cutoffs=[float(c) for c in params.dmd_cutoffs.split(",") if c.strip()],
            prompt_frac=tuple(float(v) for v in params.dmd_prompt_frac.split(",")),
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            use_fp16=params.use_fp16,
        )
        if checkpoints is not None and checkpoints.get("dmd") is not None:
            logging.info("Loading the DMD fake score")
            dmd.load_state_dict(checkpoints["dmd"])
        logging.info(
            f"DMD on: weight {params.dmd_loss_weight}, fake LoRA rank "
            f"{params.dmd_fake_rank} ({sum(p.numel() for p in dmd.fake.parameters())} "
            f"parameters), generator every {params.dmd_gen_interval} batches from "
            f"batch {params.dmd_start_step}, {params.dmd_num_step}-step rollout at "
            f"guidance {params.dmd_guidance}, teacher guidance "
            f"{params.dmd_real_guidance}, cutoffs {params.dmd_cutoffs}"
        )

    if checkpoints is not None:
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
    # A resumed run continues the epoch where the checkpoint left it rather than
    # replaying it from the start, so a run branched from a checkpoint sees the
    # same batches the original run saw after it. The sampler fast-forwards on
    # load; set_epoch in the loop below is a no-op for the same epoch.
    if (
        params.resume_from is not None
        and checkpoints is not None
        and checkpoints.get("sampler") is not None
    ):
        logging.info("Restoring the training sampler's position")
        train_dl.sampler.load_state_dict(checkpoints["sampler"])

    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    if params.scan_oom:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
            perceptual=perceptual,
            teacher=teacher,
            fbank=fbank,
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
            discriminators=discriminators,
            disc_optimizer=disc_optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
            perceptual=perceptual,
            teacher=teacher,
            fbank=fbank,
            dmd=dmd,
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
            rank=rank,
            extra_states=checkpoint_extra_states(discriminators, disc_optimizer, dmd),
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
        # Degrading by resampling needs the waveform, not just the precomputed
        # feature. Forced rather than asserted: the two flags are one decision,
        # and a run that silently fell back to the mel floor would look like a
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
