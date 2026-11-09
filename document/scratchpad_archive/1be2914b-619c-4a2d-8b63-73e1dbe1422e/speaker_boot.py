"""Every LibriTTS interval in the paper, bootstrapped over speakers instead of items.

LibriTTS has 38 speakers with two items each, so items are not independent; VCTK has one
item per speaker, where the two bootstraps coincide and nothing changes. Each row prints the
point estimate, the item-level interval (as published, fresh draws) and the speaker-level
interval (a speaker's items drawn together). 10,000 draws, seed 0, one draw set shared by
every statistic within a block so paired quantities stay paired.

Session subsets (Sec. 4.4) draw from the speakers present in the subset; the session gap
(different minus same) draws speakers once and takes both subset means from the same draw."""
import numpy as np

S = "C:/Users/Admin/AppData/Local/Temp/claude/g--Other-computers-My-Laptop-OneDrive---Hanoi-University-of-Science-and-Technology-Viettel-GD2-ZipVoice/c4663cb0-247e-4698-a414-7eadacf7e081/scratchpad"
RS = "C:/zipvoice_data/exp/results_rs/libritts"
C2 = S + "/control2/libritts"
DIRS = {"n24": S + "/control1/libritts/sr24000", "n8": S + "/control1/libritts/sr8000",
        "z24": RS + "/stock_sr24000", "z8": RS + "/stock_sr8000", "f24": RS + "/f5_sr24000",
        "f8": RS + "/f5_sr8000", "cas": RS + "/cascade_nvsr_sr8000",
        "asr": C2 + "/audiosr_sr8000_trim", "asr16": C2 + "/audiosr_lm_eq16",
        "nvsr": C2 + "/nvsr_sr8000_fit", "nvsr16": C2 + "/nvsr_lm_eq16",
        "Ln24": S + "/norm/nat24", "Ln8": S + "/norm/nat8", "Lz24": S + "/norm/stock24", "Lz8": S + "/norm/stock8"}
V = {"W": "sim_scores.tsv", "C": "campp_scores.tsv"}
N = 10000


def load(path):
    return {u: float(v) for u, v in (l.rstrip("\n").split("\t") for l in list(open(path, encoding="utf-8"))[1:])}


sess = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in open(S + "/libritts_chapter.tsv", encoding="utf-8")}
uids = sorted(sess)
spk = np.array([u.split("_")[0] for u in uids])
a = {(k, v): np.array([load(f"{d}/{f}")[u] for u in uids]) for k, d in DIRS.items() for v, f in V.items()}
x = lambda k, v: a[(k, v)]


def draws(items, cluster):
    """Index draws over `items` (indices into uids): per item, or per speaker (all its items)."""
    rng = np.random.default_rng(0)
    if not cluster:
        return [items[rng.integers(0, len(items), len(items))] for _ in range(N)]
    groups = [items[spk[items] == s] for s in np.unique(spk[items])]
    return [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(N)]


def row(name, f, items):
    out = [f"  {name:44s} {f(items):+.3f}"]
    for cluster in (False, True):
        b = np.array([f(i) for i in draws(items, cluster)])
        out.append(f"[{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")
    print(" ".join(out) + f"   spk n={len(np.unique(spk[items]))}")


def I(nat24, nat8, s24, s8, v):
    return lambda i: (x(nat8, v)[i] - x(s8, v)[i] - x(nat24, v)[i] + x(s24, v)[i]).mean()


def diff(p, q, v):
    return lambda i: (x(p, v)[i] - x(q, v)[i]).mean()


allx = np.arange(len(uids))
print(f"LibriTTS: {len(uids)} items, {len(np.unique(spk))} speakers.  estimate  [item CI]  [speaker CI]")
for v in V:
    print(f"== verifier {v}")
    row("B = n24 - n8 (Sec 4.1, Table 2)", diff("n24", "n8", v), allx)
    row("I ZipVoice (Table 2)", I("n24", "n8", "z24", "z8", v), allx)
    row("I F5-TTS (Table 2)", I("n24", "n8", "f24", "f8", v), allx)
    row("I F5 - I ZipVoice", lambda i: (x("z8", v)[i] - x("f8", v)[i] - x("z24", v)[i] + x("f24", v)[i]).mean(), allx)
    row("I ZipVoice, -23 LUFS", I("Ln24", "Ln8", "Lz24", "Lz8", v), allx)
    row("registered share (n24-n8)/(n24-z8)",
        lambda i: (x("n24", v)[i].mean() - x("n8", v)[i].mean()) / (x("n24", v)[i].mean() - x("z8", v)[i].mean()), allx)
    row("B / D_f5", lambda i: (x("n24", v)[i].mean() - x("n8", v)[i].mean()) / (x("f24", v)[i].mean() - x("f8", v)[i].mean()), allx)
    for k in ["asr", "asr16", "nvsr", "nvsr16"]:
        row(f"{k} - n8 (Table 3)", diff(k, "n8", v), allx)
    row("cascade - ZipVoice at 8k (Table 3)", diff("cas", "z8", v), allx)
    for g in ["same", "diff"]:
        items = np.flatnonzero(np.array([sess[u] for u in uids]) == g)
        row(f"{g}-session: z24 - n24 (Sec 4.4)", diff("z24", "n24", v), items)
        row(f"{g}-session: I ZipVoice", I("n24", "n8", "z24", "z8", v), items)
    isd = np.array([sess[u] for u in uids]) == "diff"
    bonus = lambda i: (x("z24", v)[i] - x("n24", v)[i])
    row("session gap: bonus diff - same (Sec 4.4)",
        lambda i: bonus(i)[isd[i]].mean() - bonus(i)[~isd[i]].mean(), allx)
row("C - W offset of z24 - n24 (Sec 4.4)",
    lambda i: ((x("z24", "C")[i] - x("n24", "C")[i]) - (x("z24", "W")[i] - x("n24", "W")[i])).mean(), allx)
