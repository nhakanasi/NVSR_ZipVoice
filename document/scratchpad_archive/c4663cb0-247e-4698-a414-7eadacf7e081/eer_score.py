"""Score the EER companion (see eer_prep.py) with both verifiers.

Per speaker: target trials are all pairs within the genuine half G (45); non-target trials pair
every G file with every file of the condition's perturbed half P (100). EER is computed per
speaker and averaged, with the standard deviation over speakers, as in Carbonneau et al.'s
Table 2. 0.50 means the condition is invisible to the verifier; lower means it is detected.
Per-speaker EERs are also written to eer/per_speaker.tsv (verifier, condition, speaker, EER).
usage: eer_score.py SCRATCH CONDITION..."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import campp_sim  # noqa: E402
from zipvoice.eval.speaker_similarity.sim import SpeakerSimilarity  # noqa: E402

S = Path(sys.argv[1]) / "eer"
M = "C:/zipvoice_data/download/tts_eval_models/speaker_similarity"
split = [l.rstrip("\n").split("\t") for l in open(S / "split.tsv", encoding="utf-8")]
spk = sorted({r[0] for r in split})
G = {s: [r[2] for r in split if r[0] == s and r[1] == "G"] for s in spk}
P = {s: [r[2] for r in split if r[0] == s and r[1] == "P"] for s in spk}
wavlm = SpeakerSimilarity(f"{M}/wavlm_large_finetune.pth", f"{M}/wavlm_large/")
embedders = {
    "SIM-o": lambda p: wavlm.model([campp_sim.load_waveform(p, 16000, device=wavlm.device, max_seconds=120)])[0].cpu(),
    "CAM++": campp_sim.embed,
}


def eer(tar, non):
    tar, non = np.asarray(tar), np.asarray(non)
    best = 1.0
    for t in np.unique(np.concatenate([tar, non])):
        frr, far = np.mean(tar < t), np.mean(non >= t)
        best = min(best, max(frr, far))
    return best


out = open(S / "per_speaker.tsv", "w", encoding="utf-8", newline="\n")
with torch.no_grad():
    for vname, emb in embedders.items():
        cache = {}
        get = lambda path: cache.setdefault(path, torch.nn.functional.normalize(emb(path).flatten(), dim=0))
        for cond in sys.argv[2:]:
            per = []
            for s in spk:
                g = [get(str(S / "G" / f)) for f in G[s]]
                p = [get(str(S / cond / f)) for f in P[s]]
                tar = [float(a @ b) for a, b in combinations(g, 2)]
                non = [float(a @ b) for a in g for b in p]
                per.append(eer(tar, non))
                out.write(f"{vname}\t{cond}\t{s}\t{per[-1]:.4f}\n")
            print(f"{vname}\t{cond}\tEER {np.mean(per):.2f} ± {np.std(per):.2f}\t(n={len(spk)} speakers)", flush=True)
