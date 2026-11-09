"""Control 1: band-limit the natural references exactly as the prompts were.

prepare_bwe_eval.py writes each prompt with torchaudio.functional.resample(24 kHz -> rate);
the inference path brings it back with torchaudio.transforms.Resample(rate -> 24 kHz), both
at default settings. Same here, applied to the 76 natural references per corpus.
usage: control1_degrade.py OUT_ROOT RATE [RATE ...]"""
import sys
from pathlib import Path

import torchaudio

CORPORA = {"libritts": "C:/zipvoice_data/data/bwe_eval",
           "vctk": "C:/zipvoice_data/data/vctk_bwe_eval"}
out_root, rates = Path(sys.argv[1]), [int(r) for r in sys.argv[2:]]
for corpus, root in CORPORA.items():
    uids = [l.split("\t")[0] for l in open(f"{root}/test_score.tsv", encoding="utf-8")]
    for rate in rates:
        out = out_root / corpus / f"sr{rate}"
        out.mkdir(parents=True, exist_ok=True)
        up = torchaudio.transforms.Resample(orig_freq=rate, new_freq=24000)
        for uid in uids:
            w, sr = torchaudio.load(f"{root}/references/{uid}.wav")
            assert sr == 24000, (uid, sr)
            if rate != 24000:
                w = up(torchaudio.functional.resample(w, orig_freq=24000, new_freq=rate))
            torchaudio.save(str(out / f"{uid}.wav"), w, 24000)
        print(corpus, rate, len(uids), flush=True)
