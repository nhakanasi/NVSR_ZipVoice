"""Recover each LibriTTS eval prompt's source utterance (prepare_bwe_eval.py did not store
its id) by matching speaker and transcript in dev-clean, and write uid -> same/diff chapter."""
import glob
import sys

S = sys.argv[1]
text = {}
for f in glob.glob("C:/zipvoice_data/download/LibriTTS/dev-clean/*/*/*.normalized.txt"):
    uid = f.replace("\\", "/").rsplit("/", 1)[1][: -len(".normalized.txt")]
    text.setdefault((uid.split("_")[0], open(f, encoding="utf-8").read().strip()), []).append(uid)
out, n = open(f"{S}/libritts_chapter.tsv", "w", encoding="utf-8", newline="\n"), {}
for line in open("C:/zipvoice_data/data/bwe_eval/test_score.tsv", encoding="utf-8"):
    wav, ptext = line.split("\t")[:2]
    spk, target = wav.split("_", 1)
    hits = text.get((spk, ptext.strip()), [])
    tag = ("same" if hits[0].split("_")[1] == target.split("_")[1] else "diff") if len(hits) == 1 else f"unmatched{len(hits)}"
    n[tag] = n.get(tag, 0) + 1
    out.write(f"{wav}\t{hits[0] if hits else ''}\t{tag}\n")
print(n)
