"""Stock F5-TTS on the four-column test list, no adapter anywhere in the path.

Mirrors f5_adapter/infer.py (RMS rule, duration rule, ref_text + text, no chunking,
no silence trimming, no cross-fade) but builds the model with upstream's own
load_model, so the result is the released checkpoint and nothing else.
usage: f5_stock_infer.py TEST_LIST RES_DIR [SEED]"""
import csv
import sys
import time
from importlib.resources import files
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model import DiT
from f5_tts.model.utils import convert_char_to_pinyin

CKPT = "C:/zipvoice_data/download/f5tts/F5TTS_v1_Base/model_1250000.safetensors"
VOCODER = "C:/zipvoice_data/download/vocos-mel-24khz"
NFE, CFG, SWAY, SPEED, TARGET_RMS = 32, 2.0, -1.0, 1.0, 0.1

test_list, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 666
res_dir.mkdir(parents=True, exist_ok=True)
device = "cuda"

cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
model = load_model(DiT, OmegaConf.to_container(cfg.model.arch), CKPT, mel_spec_type="vocos",
                   vocab_file=str(files("f5_tts").joinpath("infer/examples/vocab.txt")),
                   device=device).eval()
vocoder = load_vocoder("vocos", is_local=True, local_path=VOCODER, device=device)
sr = int(cfg.model.mel_spec.target_sample_rate)
hop = int(cfg.model.mel_spec.hop_length)

torch.manual_seed(seed)
total_audio = total_time = 0.0
with test_list.open(encoding="utf-8") as fh:
    rows = [r for r in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE) if r]
for wav_name, prompt_text, prompt_wav, text in rows:
    audio, rate = torchaudio.load(prompt_wav)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    rms = torch.sqrt(audio.square().mean())
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms
    if rate != sr:
        audio = torchaudio.transforms.Resample(rate, sr)(audio)
    audio = audio.to(device)
    ref_text = prompt_text if prompt_text.endswith(" ") else prompt_text + " "
    ref_len = audio.shape[-1] // hop
    duration = ref_len + int(ref_len / max(1, len(ref_text.encode("utf-8")))
                             * len(text.encode("utf-8")) / SPEED)
    start = time.time()
    with torch.inference_mode():
        gen, _ = model.sample(cond=audio, text=convert_char_to_pinyin([ref_text + text]),
                              duration=duration, steps=NFE, cfg_strength=CFG,
                              sway_sampling_coef=SWAY)
        mel = gen.to(torch.float32)[:, ref_len:, :].permute(0, 2, 1)
        wave = vocoder.decode(mel)
        if rms < TARGET_RMS:
            wave = wave * rms / TARGET_RMS
    total_time += time.time() - start
    wave = wave.squeeze().cpu()
    total_audio += wave.shape[-1] / sr
    torchaudio.save(str(res_dir / f"{wav_name}.wav"), wave.unsqueeze(0), sr)
print(f"{test_list} -> {res_dir}: {len(rows)} files, {total_audio:.1f}s audio, "
      f"RTF {total_time / max(total_audio, 1e-6):.4f}", flush=True)
