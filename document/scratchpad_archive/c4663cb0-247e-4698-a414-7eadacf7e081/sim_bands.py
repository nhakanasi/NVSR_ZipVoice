"""Which band does SIM-o read? Score each output full, low-passed at the prompt
cutoff (generated band removed), and at -6 dB gain, against the clean prompt.
CPU only: the trainer holds the GPU."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(8)
from zipvoice.eval.speaker_similarity.sim import SpeakerSimilarity
from zipvoice.eval.utils import load_waveform

M = "C:/zipvoice_data/download/tts_eval_models/speaker_similarity"
R = "C:/zipvoice_data/exp/results"
D = "C:/zipvoice_data/data/bwe_eval"
sim = SpeakerSimilarity(f"{M}/wavlm_large_finetune.pth", f"{M}/wavlm_large/")
sim.model.to("cpu")
rows = [l.rstrip("\n").split("\t") for l in open(f"{D}/test_score.tsv", encoding="utf-8") if l.strip()]

def emb(x):
    with torch.no_grad():
        return sim.model([torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))])

def lowpass(x, cut):
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / 16000) >= cut] = 0
    return np.fft.irfft(X, len(x))

P = {n: emb(load_waveform(p, 16000, return_numpy=True)) for n, _, p, _ in rows}
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).item()
for rate in (4000, 8000):
    cut = rate // 2
    for sysname in ["ref"] + sys.argv[1:]:
        acc = {"full": [], "lp": [], "gain-6dB": []}
        for n, _, _, _ in rows:
            path = f"{D}/references/{n}.wav" if sysname == "ref" else f"{R}/{sysname}_sr{rate}/{n}.wav"
            if not os.path.exists(path):
                continue
            x = load_waveform(path, 16000, return_numpy=True)
            acc["full"].append(cos(P[n], emb(x)))
            acc["lp"].append(cos(P[n], emb(lowpass(x, cut))))
            acc["gain-6dB"].append(cos(P[n], emb(0.5 * x)))
        d = np.array(acc["lp"]) - np.array(acc["full"])
        print(f"sr{rate} {sysname:24s} n={len(d)} full {np.mean(acc['full']):.3f} "
              f"lp{cut} {np.mean(acc['lp']):.3f} (d {d.mean():+.3f} se {d.std(ddof=1)/np.sqrt(len(d)):.3f}) "
              f"gain-6dB {np.mean(acc['gain-6dB']):.3f}", flush=True)
