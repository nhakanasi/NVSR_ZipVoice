#!/usr/bin/env python3
"""Turn the scoring log and the inference logs into one CSV per sampling rate.

Reads:
  the scoring log, whose arms are delimited by "--- <set> <system> @ <rate> Hz ---"
  <results>/<set>_<system>_sr<rate>.infer.log, for the per-sentence real-time factor
  <results>/audiosr_time_<set>_<rate>.tsv, for what the cascade's extender cost

Writes <results>/testset_sr<rate>.csv.

One real-time factor is reported, `rtf`, and it is end-to-end: every stage a
system needs, over the audio that system produced. For stock and adapter that
is synthesis alone; for the two cascades it is synthesis plus the prompt
restoration their synthesis depends on.

The single column exists because the two stages are not natively on the same
denominator. Synthesis is timed against generated audio, but an extender can
only be timed against the prompt it was handed, and the prompt-to-generated
ratio measured here is 1.10 on seedtts_en, 0.76 on seedtts_zh and 0.46 on
librispeech_pc. Adding the two published ratios would therefore be wrong by up
to a factor of two, in either direction depending on the set. The extender's
wall-clock time is divided by generated seconds directly instead.

Working from wall clock also sidesteps a measurement artefact: AudioSR pads its
input to 10.24 s chunks and logs the padded length, so for the same 189 prompts
it records 1270 s of audio where NVSR records 889 s. An extender RTF normalised
by that column is divided by a denominator roughly 45% too large and flatters
AudioSR; nothing here uses it.

Every utterance is charged its own prompt restoration. The runs cached repeated
prompts -- 158 to 189 distinct prompts across 200 utterances -- which is right
for the experiment and wrong as a deployment cost, where each request restores
the prompt it was given.
"""

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Optional, Tuple

import soundfile

# The NVSR driver labels its arms "--- score <set> ... ---"; the earlier runs
# omitted the verb. Both spellings name the same thing.
ARM_RE = re.compile(
    r"^--- (?:score )?(\w+) (stock|adapter|cascade_nvsr|cascade) @ (\d+) Hz ---"
)
SIM_RE = re.compile(r"SIM-o score: ([\d.]+)")
UTMOS_RE = re.compile(r"UTMOS score: ([\d.]+)")
SEEDTTS_WER_RE = re.compile(r"Seed-TTS WER: ([\d.]+)%")
POOLED_WER_RE = re.compile(r"\]\s*WER\s*[:=]\s*([\d.]+)%")
RTF_RE = re.compile(r"\[Sentence: (\d+)\] RTF: ([\d.]+)")

SETS = ["seedtts_en", "seedtts_zh", "librispeech_pc"]
# Two cascades, one per off-the-shelf extender. They share a synthesis
# stage -- stock ZipVoice on the restored prompt -- and differ only in which
# model restored it, so the pair isolates the extender.
SYSTEMS = ["stock", "cascade", "cascade_nvsr", "adapter"]
EXTENDER = {"cascade": "audiosr", "cascade_nvsr": "nvsr"}


def mean_rtf(path: Path) -> Optional[float]:
    """Mean per-sentence synthesis RTF, discarding sentence 0.

    The first sentence carries CUDA context setup and kernel autotuning that no
    later one pays; including it would flatter whichever system ran last.
    """
    if not path.is_file():
        return None
    values = []
    with io.open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            found = RTF_RE.search(line)
            if found and int(found.group(1)) > 0:
                values.append(float(found.group(2)))
    return sum(values) / len(values) if values else None


def generated_seconds(res_dir: Path) -> Tuple[float, int]:
    """Total duration and count of the wavs an arm produced."""
    if not res_dir.is_dir():
        return 0.0, 0
    total, count = 0.0, 0
    for wav in sorted(res_dir.glob("*.wav")):
        info = soundfile.info(str(wav))
        total += info.frames / info.samplerate
        count += 1
    return total, count


def extender_rtf(path: Path, gen_secs: float, num_wavs: int) -> float:
    """Extender wall time over *generated* audio, on synthesis's denominator.

    The timing file holds one row per *distinct* prompt, because the runs
    restored each prompt once and reused it. A deployment does not get that
    discount -- every request carries its own prompt -- so the measured wall
    time is scaled by utterances over distinct prompts before dividing.
    """
    if not path.is_file() or not gen_secs or not num_wavs:
        return 0.0
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return 0.0
    wall = sum(float(r["seconds_wall"]) for r in rows) * num_wavs / len(rows)
    return wall / gen_secs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-log", type=Path, required=True)
    parser.add_argument(
        "--results", type=Path, default=Path("/c/zipvoice_data/exp/results")
    )
    parser.add_argument("--rates", type=str, default="8000,16000")
    args = parser.parse_args()

    scores: dict = {}
    current = None
    with io.open(args.score_log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            found = ARM_RE.match(line.strip())
            if found:
                current = (found.group(1), found.group(2), int(found.group(3)))
                scores.setdefault(current, {})
                continue
            if current is None:
                continue
            entry = scores[current]
            for key, regex in (
                ("sim", SIM_RE),
                ("utmos", UTMOS_RE),
                ("wer_seedtts", SEEDTTS_WER_RE),
                ("wer", POOLED_WER_RE),
            ):
                found = regex.search(line)
                # First occurrence wins: a rerun appends to the same log, and the
                # arm header resets `current` anyway.
                if found and key not in entry:
                    entry[key] = found.group(1)

    for rate in (int(r) for r in args.rates.split(",") if r.strip()):
        out = args.results / f"testset_sr{rate}.csv"
        with open(out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "testset",
                    "system",
                    "sim",
                    "wer",
                    "wer_seedtts",
                    "utmos",
                    "rtf",
                ]
            )
            for name in SETS:
                for system in SYSTEMS:
                    entry = scores.get((name, system, rate), {})
                    stem = f"{name}_{system}_sr{rate}"
                    # A ".rtf.log" is a quiet-GPU probe run on purpose and
                    # supersedes the synthesis log, which for the NVSR arm was
                    # written while three jobs shared the card and measures
                    # contention, not the model.
                    rtf = mean_rtf(args.results / f"{stem}.rtf.log")
                    if rtf is None:
                        rtf = mean_rtf(args.results / f"{stem}.infer.log")
                    if rtf is not None and system in EXTENDER:
                        gen_secs, num_wavs = generated_seconds(args.results / stem)
                        rtf += extender_rtf(
                            args.results
                            / f"{EXTENDER[system]}_time_{name}_{rate}.tsv",
                            gen_secs,
                            num_wavs,
                        )
                    writer.writerow(
                        [
                            name,
                            system,
                            entry.get("sim", ""),
                            entry.get("wer", ""),
                            entry.get("wer_seedtts", ""),
                            entry.get("utmos", ""),
                            "" if rtf is None else f"{rtf:.4f}",
                        ]
                    )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
