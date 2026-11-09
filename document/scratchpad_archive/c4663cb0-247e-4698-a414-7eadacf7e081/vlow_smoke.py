"""Smoke check for the rollout's running velocity cost.

For each checkpoint: value of the low-band velocity gap on the training rollout
(resampled prompt, 4 kHz), gradient reaching LoRA and branch, peak memory, and
the terminal low L2 against the stock solve from the same degraded prompt.
"""
import json, sys
import torch, torchaudio
from lhotse import load_manifest_lazy
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter, adapter_kwargs
from zipvoice.models.modules.bwe import resample_band_limit
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import add_punctuation

E = "C:/zipvoice_data/exp"
SR, CUT, FS = 24000, 4000.0, 0.1
dev = torch.device("cuda")
tok = EmiliaTokenizer(token_file=f"{E}/adapter_lora/tokens.txt")
cfg = json.load(open(f"{E}/adapter_lora/model.json"))
fx = VocosFbank()
fbank = VocosFbank().fbank.to(dev)
cuts = []
for c in load_manifest_lazy(sys.argv[1]):
    if 4.0 <= c.duration <= 8.0:
        cuts.append(c)
    if len(cuts) == int(sys.argv[2]):
        break

for ckpt in sys.argv[3:]:
    model = ZipVoiceAdapter(**cfg["model"], **adapter_kwargs(cfg), vocab_size=tok.vocab_size,
                            pad_id=tok.pad_id, feat_scale=FS, sampling_rate=SR)
    load_checkpoint(filename=ckpt, model=model, strict=True)
    model = model.to(dev)
    model.set_trainable("lora") if hasattr(model, "set_trainable") else None
    vals, lows = [], []
    torch.cuda.reset_peak_memory_stats()
    for i, c in enumerate(cuts):
        wav = torch.from_numpy(c.load_audio()).float()[:1]
        if c.sampling_rate != SR:
            wav = torchaudio.functional.resample(wav, c.sampling_rate, SR)
        clean = fx.extract(wav, sampling_rate=SR).to(dev) * FS
        T = clean.size(0)
        P = int(0.3 * T)
        masked = resample_band_limit(
            audio=wav.to(dev), audio_lens=torch.tensor([wav.size(1)], device=dev),
            cutoff_hz=torch.tensor([CUT], device=dev), fbank=fbank, num_frames=T,
            sampling_rate=SR)[0].to(clean.dtype) * FS
        ids = tok.tokens_to_token_ids(tok.texts_to_tokens([add_punctuation(c.supervisions[0].text)]))[0]
        text_cond, pad = model.forward_text_train(tokens=[ids], features_lens=torch.tensor([T], device=dev))
        gen = torch.zeros(1, T, dtype=torch.bool, device=dev)
        gen[:, P:] = True
        speech = torch.where(gen[..., None], 0, masked)
        _, band_mask = model._band_limit(clean[None], torch.tensor([CUT], device=dev), randomize=False)
        torch.manual_seed(1000 + i)
        noise = torch.randn(1, T, clean.size(1), device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            sample, _, ref, vlow = model.identity_rollout(
                noise=noise, text_condition=text_cond.detach(), speech_condition=speech, padding_mask=pad,
                band_mask=band_mask, cutoff_hz=torch.tensor([CUT], device=dev), prompt_mask=~gen & ~pad,
                num_step=16, guidance_scale=1.0, t_shift=0.5, grad_steps=4,
                velocity_low_steps=int(__import__("os").environ.get("VSTEPS", 16)))
        vlow.backward()
        g_lora = sum(float(m.lora_B.grad.norm()) for m in model._lora_modules if m.lora_B.grad is not None)
        g_branch = sum(float(p.grad.norm()) for n, p in model.named_parameters()
                       if n.startswith("adapter.") and p.grad is not None)
        model.zero_grad(set_to_none=True)
        vals.append(float(vlow.detach()))
        print(f"{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]} {i} vlow {float(vlow):.5f} "
              f"grad lora {g_lora:.3e} branch {g_branch:.3e}", flush=True)
    print(f"MEAN {ckpt}: vlow {sum(vals)/len(vals):.5f} peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    del model
    torch.cuda.empty_cache()
