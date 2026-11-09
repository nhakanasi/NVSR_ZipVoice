"""Band-matched scoring, step 1: low-pass the prompts and every scored output to the 8 kHz band.

LP(x) = Resample(8k -> 24k)(resample(x, 24k -> 8k)), the op that made the 8 kHz prompts and
the control-1 references (control1_degrade.py). It is applied to every signal that is scored
band-matched, the already band-limited ones included, so the metric is one function:
SIM_m(x) = cos(E(LP x), E(LP prompt)).

Writes <OUT>/<corpus>/<name>/<uid>.wav and <OUT>/<corpus>/test_lp8.tsv (test_score.tsv with
column 3 pointing at the low-passed prompt)."""
import csv
from pathlib import Path

import torchaudio

OUT = Path(__file__).parent / "bandmatch"
OS = "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
CORPORA = {"libritts": "C:/zipvoice_data/data/bwe_eval", "vctk": "C:/zipvoice_data/data/vctk_bwe_eval"}
up = torchaudio.transforms.Resample(orig_freq=8000, new_freq=24000)


def lp(w):
    return up(torchaudio.functional.resample(w, orig_freq=24000, new_freq=8000))


for corpus, root in CORPORA.items():
    rs, c2 = f"C:/zipvoice_data/exp/results_rs/{corpus}", f"{OS}/control2/{corpus}"
    src = {"n24": f"{root}/references", "n8": f"{OS}/control1/{corpus}/sr8000",
           "z24": f"{rs}/stock_sr24000", "z8": f"{rs}/stock_sr8000",
           "f24": f"{rs}/f5_sr24000", "f8": f"{rs}/f5_sr8000", "cas": f"{rs}/cascade_nvsr_sr8000",
           "asr": f"{c2}/audiosr_sr8000_trim", "nvsr": f"{c2}/nvsr_sr8000_fit"}
    rows = [r for r in csv.reader(open(f"{root}/test_score.tsv", encoding="utf-8"), delimiter="\t",
                                  quoting=csv.QUOTE_NONE) if r]
    (OUT / corpus / "prompt").mkdir(parents=True, exist_ok=True)
    with open(OUT / corpus / "test_lp8.tsv", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            w, sr = torchaudio.load(r[2])
            assert sr == 24000, (r[2], sr)
            p = OUT / corpus / "prompt" / f"{r[0]}.wav"
            torchaudio.save(str(p), lp(w), 24000)
            f.write("\t".join([r[0], r[1], str(p), r[3]]) + "\n")
    for name, d in src.items():
        if not Path(d).is_dir():
            print(f"{corpus} {name}: missing {d}", flush=True)
            continue
        out = OUT / corpus / name
        out.mkdir(exist_ok=True)
        for r in rows:
            w, sr = torchaudio.load(f"{d}/{r[0]}.wav")
            assert sr == 24000, (d, r[0], sr)
            torchaudio.save(str(out / f"{r[0]}.wav"), lp(w), 24000)
        print(corpus, name, len(rows), flush=True)
