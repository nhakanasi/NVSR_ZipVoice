"""Control 2 inputs: natural references cut to 8 kHz exactly as the prompts were
(torchaudio.functional.resample 24 kHz -> 8 kHz, default settings), saved at 8 kHz, plus a
test list in audiosr_restore.py's format whose column 2 points at them."""
import sys
from pathlib import Path

import torchaudio

CORPORA = {"libritts": "C:/zipvoice_data/data/bwe_eval",
           "vctk": "C:/zipvoice_data/data/vctk_bwe_eval"}
out_root = Path(sys.argv[1])
for corpus, root in CORPORA.items():
    out = out_root / corpus / "in_sr8000"
    out.mkdir(parents=True, exist_ok=True)
    rows = [l.rstrip("\n").split("\t") for l in open(f"{root}/test_score.tsv", encoding="utf-8")]
    with open(out_root / corpus / "refs_sr8000.tsv", "w", encoding="utf-8", newline="\n") as f:
        for uid, ptext, _, text in rows:
            w, sr = torchaudio.load(f"{root}/references/{uid}.wav")
            assert sr == 24000, (uid, sr)
            path = out / f"{uid}.wav"
            torchaudio.save(str(path), torchaudio.functional.resample(w, 24000, 8000), 8000)
            f.write(f"{uid}\t{ptext}\t{path.as_posix()}\t{text}\n")
    print(corpus, len(rows), flush=True)
