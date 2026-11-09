"""List a remote zip's members by reading only its central directory with HTTP range requests.
usage: zip_listing.py URL OUT_TSV"""
import struct
import subprocess
import sys


def get(url, start, end=None):
    # curl, not urllib: Zenodo answers urllib's range requests with 403
    rng = f"{start}-" if end is None else f"{start}-{end}"
    r = subprocess.run(["curl", "-sS", "-L", "-r", rng, "-D", "-", url], capture_output=True, check=True)
    head, body = r.stdout.split(b"\r\n\r\n", 1)
    while body.startswith(b"HTTP/"):  # redirect hops put several header blocks in front
        head, body = body.split(b"\r\n\r\n", 1)
    cr = [l for l in head.decode().splitlines() if l.lower().startswith("content-range")]
    return body, int(cr[0].split("/")[1])


url, out = sys.argv[1], sys.argv[2]
_, size = get(url, 0, 0)
tail, _ = get(url, max(0, size - 131072))
i = tail.rfind(b"PK\x05\x06")
cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64: read the zip64 end record
    j = tail.rfind(b"PK\x06\x06")
    cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
cd, _ = get(url, cd_off, cd_off + cd_size - 1)
names, p = [], 0
while cd[p:p + 4] == b"PK\x01\x02":
    usize = struct.unpack("<I", cd[p + 24:p + 28])[0]
    n, m, k = struct.unpack("<HHH", cd[p + 28:p + 34])
    names.append((cd[p + 46:p + 46 + n].decode("utf-8", "replace"), usize))
    p += 46 + n + m + k
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for nm, us in names:
        f.write(f"{nm}\t{us}\n")
print(f"{len(names)} members, archive {size / 1e9:.2f} GB")
