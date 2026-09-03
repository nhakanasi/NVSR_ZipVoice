#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../../LICENSE for clarification regarding multiple authors
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
Measure what the bandwidth extender costs on top of ZipVoice.

The deployment argument for a prompt-side extender is that it is cheap: it runs
once over the prompt, whereas the flow-matching solver runs `num_step` times
over the output, and there is no vocoder in the added path because the extender
produces the mel the TTS already consumes. That argument needs numbers rather
than adjectives, so this script reports parameter counts and wall-clock timings
for the extender against a full synthesis on the same device.

Timings use random inputs of a fixed length. Absolute numbers depend on the
hardware; the ratio is the reportable quantity.

Example::

    python3 local/cost_bench.py \
        --checkpoint exp/zipvoice_bwe/iter-15000-avg-2.pt \
        --model-config conf/zipvoice_bwe.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# The repository is used from a checkout rather than installed, and this script
# lives three directories below the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from zipvoice.models.zipvoice_bwe import ZipVoiceBWE  # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path of a trained ZipVoiceBWE checkpoint.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/zipvoice_bwe.json",
        help="Path of the model configuration used to build the checkpoint.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=360,
        help="Token vocabulary size of the checkpoint.",
    )
    parser.add_argument(
        "--prompt-seconds",
        type=float,
        default=6.19,
        help="Prompt duration to time, in seconds. The default is the median "
        "duration of the evaluation prompts.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
        help="ODE solver steps, matching the evaluation recipe.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Timed iterations for the synthesis benchmark.",
    )
    return parser


def bench(fn, device, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main():
    args = get_parser().parse_args()

    with open(args.model_config) as f:
        config = json.load(f)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)

    n_bwe = sum(v.numel() for k, v in state.items() if k.startswith("bwe."))
    n_tts = sum(v.numel() for k, v in state.items() if not k.startswith("bwe."))
    print("parameters")
    print(f"  ZipVoice            {n_tts / 1e6:8.2f} M")
    print(f"  extender            {n_bwe / 1e6:8.2f} M")
    print(f"  overhead            {100 * n_bwe / n_tts:8.1f} %")

    sampling_rate = config["feature"]["sampling_rate"]
    # The feature block names the extractor rather than its geometry, so
    # the VocosFbank hop and dimension are taken from the extractor.
    hop_length = 256
    model = ZipVoiceBWE(
        **config["model"],
        **config.get("bwe", {}),
        vocab_size=args.vocab_size,
        pad_id=0,
        feat_scale=0.1,
        sampling_rate=sampling_rate,
    )
    model.load_state_dict(state)
    model.eval()

    feat_dim = config["model"].get("feat_dim", 100)
    frames = int(args.prompt_seconds * sampling_rate / hop_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokens = [torch.randint(1, args.vocab_size - 1, (60,)).tolist()]
    prompt_features = torch.randn(1, frames, feat_dim, device=device)
    prompt_features_lens = torch.tensor([frames], device=device)
    # Half the band observed, which is the 8 kHz prompt condition.
    band_mask = torch.zeros(1, 1, feat_dim, device=device)
    band_mask[..., : feat_dim // 2] = 1.0

    def synthesise():
        return model.sample(
            tokens=tokens,
            prompt_tokens=tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            num_step=args.num_step,
            guidance_scale=1.0,
        )

    with torch.inference_mode():
        t_bwe = bench(
            lambda: model.bwe(prompt_features, band_mask), device, iters=30
        )
        output = synthesise()
        features = output[0] if isinstance(output, tuple) else output
        output_seconds = features.shape[1] * hop_length / sampling_rate
        t_synth = bench(synthesise, device, iters=args.iters)

    print()
    print(
        f"timings on {device}: prompt {args.prompt_seconds} s "
        f"({frames} frames), output {output_seconds:.2f} s, "
        f"{args.num_step} ODE steps"
    )
    print(f"  extender forward    {1000 * t_bwe:8.2f} ms")
    print(f"  ZipVoice sample()   {1000 * t_synth:8.2f} ms")
    print(f"  extender share      {100 * t_bwe / (t_synth + t_bwe):8.2f} %")
    print(f"  synthesis RTF       {t_synth / output_seconds:8.4f}")
    print()
    print(
        "The extender runs once over the prompt; the solver runs "
        f"{args.num_step} times over the output."
    )


if __name__ == "__main__":
    main()
