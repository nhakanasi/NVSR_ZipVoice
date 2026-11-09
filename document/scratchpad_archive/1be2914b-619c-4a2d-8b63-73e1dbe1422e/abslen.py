"""Abstract length in characters, LaTeX comments dropped and each \\tbd counted as one dash."""
import re
import sys

t = open(sys.argv[1], encoding="utf-8").read()
a = t.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
a = re.sub(r"%.*", "", a)
a = re.sub(r"\\tbd(\[[^\]]*\])?", "-", a)
print(len(" ".join(a.split())))
