"""Low-pass each system's outputs at 4 kHz into a sibling dir, for SIM-o on the
low band alone. Also prints long-term 0-4 kHz spectrum distance to the clean
prompt, in dB, with each spectrum normalised to its own 0-4 kHz level."""
import glob, os, sys, csv, torch, torchaudio
R = "C:/zipvoice_data/exp/results"
T = "C:/zipvoice_data/data/bwe_eval/test_score.tsv"
prompt = {r[0]: r[2] for r in csv.reader(open(T, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
def ltas(w, sr):
    s = torch.stft(w[0], 1024, 256, window=torch.hann_window(1024), return_complex=True).abs().pow(2).mean(-1)
    f = torch.arange(s.numel()) * sr / 1024
    s = s[(f >= 100) & (f < 4000)]
    return 10 * torch.log10(s / s.mean() + 1e-12)
for sysname in sys.argv[1:]:
    out = f"{R}/{sysname}_lp4k"; os.makedirs(out, exist_ok=True)
    d = []
    for f in sorted(glob.glob(f"{R}/{sysname}/*.wav")):
        w, sr = torchaudio.load(f)
        torchaudio.save(f"{out}/{os.path.basename(f)}", torchaudio.functional.lowpass_biquad(torchaudio.functional.lowpass_biquad(w, sr, 4000.), sr, 4000.), sr)
        uid = os.path.basename(f)[:-4]
        if uid in prompt:
            p, psr = torchaudio.load(prompt[uid])
            a, b = ltas(w, sr), ltas(p, psr)
            n = min(a.numel(), b.numel()) if sr == psr else None
            if n: d.append(float((a[:n] - b[:n]).abs().mean()))
    print(f"{sysname}: n={len(d)} LTAS 0.1-4k |gen-prompt| {sum(d)/max(1,len(d)):.2f} dB", flush=True)
