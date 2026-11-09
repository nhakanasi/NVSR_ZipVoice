import re

p = "egs/zipvoice/run_adapter_libritts.sh"
s = open(p, encoding="utf-8", newline="").read()
lines = s.split("\n")
out = []
for line in lines:
    if line.lstrip().startswith("--rollout-band-level-weight ${lora_band_level_weight}"):
        indent = line[: len(line) - len(line.lstrip())]
        body = line.strip()
        assert body.endswith("\\"), body
        parts = re.split(r" {2,}", body[:-1].strip())
        parts.insert(parts.index("--identity-rollout-low-band ${lora_identity_low_band}") + 1,
                     "--rollout-velocity-low-weight ${lora_velocity_low_weight}")
        parts.insert(parts.index("--rollout-velocity-low-weight ${lora_velocity_low_weight}") + 1,
                     "--rollout-velocity-low-steps ${lora_velocity_low_steps}")
        out.extend(indent + part + " \\" for part in parts)
    else:
        out.append(line)
s = "\n".join(out)
old = "lora_identity_low_band=${lora_identity_low_band:-false}\n"
assert s.count(old) == 1
s = s.replace(old, old + """# A running cost along the rollout's solve (VGG-Flow): the adapted velocity
# against the stock one at the same state and degraded prompt, on the observed
# bins. It holds the low band's trajectory at every step rather than scoring the
# endpoint, where the clean-prompt reference puts a floor under the low L2.
lora_velocity_low_weight=${lora_velocity_low_weight:-0}
lora_velocity_low_steps=${lora_velocity_low_steps:-16}
""")
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
