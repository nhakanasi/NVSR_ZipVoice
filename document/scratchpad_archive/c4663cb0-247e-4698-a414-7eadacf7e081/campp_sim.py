"""Second verifier: CAM++ (3D-Speaker, VoxCeleb, iic/speech_campplus_sv_en_voxceleb_16k).

Same pairs, loading and output format as zipvoice.eval.speaker_similarity.sim, with the
WavLM ECAPA-TDNN swapped for CAM++. The CAMPPlus class is taken from modelscope's own
source with its registry imports stripped, so modelscope's heavy deps are not needed.
Features follow SpeakerVerificationCAMPPlus: Kaldi fbank (80 bins) with per-utterance mean
removal, on 16 kHz audio.
usage: campp_sim.py TEST_LIST OUT_TSV WAV_DIR [WAV_DIR ...]   (writes WAV_DIR/OUT_TSV)"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as Kaldi

from zipvoice.eval.utils import load_waveform

SV = Path(torch.__file__).parents[1] / "modelscope/models/audio/sv"
CKPT = (Path.home() / ".cache/modelscope/models/iic--speech_campplus_sv_en_voxceleb_16k"
        / "snapshots/master/campplus_voxceleb.bin")

ns = {}
exec((SV / "DTDNN_layers.py").read_text(encoding="utf-8"), ns)
src = (SV / "DTDNN.py").read_text(encoding="utf-8")
src = src[:src.index("@MODELS.register_module")]
src = src[:src.index("from modelscope")] + src[src.index("class FCM"):]
exec(src, ns)
model = ns["CAMPPlus"](80, 512)
model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
model.eval()


@torch.no_grad()
def embed(path):
    w = load_waveform(path, 16000, max_seconds=120)
    f = Kaldi.fbank(w.unsqueeze(0), num_mel_bins=80)
    return model((f - f.mean(0, keepdim=True)).unsqueeze(0))[0]


if __name__ == "__main__":
    test_list, out_name = sys.argv[1], sys.argv[2]
    rows = [l.rstrip("\n").split("\t") for l in open(test_list, encoding="utf-8") if l.strip()]
    prompt_emb = {r[0]: embed(r[2]) for r in rows}
    for wav_dir in sys.argv[3:]:
        scores = {uid: torch.nn.functional.cosine_similarity(
            embed(f"{wav_dir}/{uid}.wav"), p, dim=0).item() for uid, p in prompt_emb.items()}
        with open(f"{wav_dir}/{out_name}", "w", encoding="utf-8", newline="\n") as f:
            f.write("wav_name\tsim\n")
            for name, v in scores.items():
                f.write(f"{name}\t{v:.6f}\n")
        print(f"{wav_dir}\t{np.mean(list(scores.values())):.4f}", flush=True)
