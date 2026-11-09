"""A-D substitution test: where does the 4 kHz band go dark between the
training rollout and evaluation? Same checkpoint, same noise per utterance.

A  training prefix, training degradation, identity_rollout (training path)
B  training prefix, training degradation, sample()
C1 training prefix, eval resampling of the prompt audio alone, sample()
C2 as C1 plus remove_silence and rms_norm (the eval prompt pipeline)
D1 eval row as is: other utterance prompt, eval pipeline, predicted duration
D2 as D1 with the reference's real duration
S  stock model, clean prompt (identity_rollout's reference solve)

Reports generated-span mean log-mel minus ground truth, in dB, per band.
"""
import json, os, sys, csv
import torch, torchaudio
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter, adapter_kwargs
from zipvoice.models.modules.bwe import resample_band_limit
from zipvoice.models.modules.perceptual import SpeakerPerceptual
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import add_punctuation, load_prompt_wav, remove_silence, rms_norm

E = "C:/zipvoice_data/exp"
D = "C:/zipvoice_data/data/bwe_eval"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoint-18000.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
G = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SR, CUT, FS, HOP = 24000, 4000.0, 0.1, 256
dev = torch.device("cuda")

tok = EmiliaTokenizer(token_file=f"{E}/adapter_lora/tokens.txt")
cfg = json.load(open(f"{E}/adapter_lora/model.json"))
model = ZipVoiceAdapter(**cfg["model"], **adapter_kwargs(cfg), vocab_size=tok.vocab_size,
                        pad_id=tok.pad_id, feat_scale=FS, sampling_rate=SR)
load_checkpoint(filename=f"{E}/adapter_lora/{ckpt}", model=model, strict=True)
model = model.to(dev).eval().requires_grad_(False)
spk = SpeakerPerceptual(
    speaker_checkpoint="C:/zipvoice_data/download/tts_eval_models/speaker_similarity/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin",
    feat_scale=FS).to(dev).eval()
fx = VocosFbank()
fbank = VocosFbank().fbank.to(dev)

mel_hz = torchaudio.functional.melscale_fbanks(513, 0, SR / 2, 100, SR, norm=None, mel_scale="htk").argmax(0) * SR / 1024
BANDS = {"2-4k": (2e3, 4e3), "4-6k": (4e3, 6e3), "6-8k": (6e3, 8e3), "8-12k": (8e3, 12e3)}
masks = {k: ((mel_hz >= lo) & (mel_hz < hi)).to(dev) for k, (lo, hi) in BANDS.items()}
DB = 20 / torch.log(torch.tensor(10.0)).item()


def gap(gen, gt):
    """Per-bin mean over frames, band average of gen - gt, dB. Both scaled log-mel (T, F)."""
    d = (gen.mean(0) - gt.mean(0)) / FS * DB
    # Energy: per-bin log of the frame-mean power, the statistic a wav-domain
    # band energy measures. Loud frames dominate it; silence does not.
    def energy(x):
        return torch.logsumexp(2 * x / FS, dim=0) - torch.log(torch.tensor(float(x.size(0))))
    de = (energy(gen) - energy(gt)) * 10 / torch.log(torch.tensor(10.0))
    return [float(d[m].mean()) for m in masks.values()] + [float(de[m].mean()) for m in masks.values()]


def feats(wav):
    return fx.extract(wav, sampling_rate=SR).to(dev) * FS  # (T, F)


def ids(text):
    return tok.tokens_to_token_ids(tok.texts_to_tokens([add_punctuation(text)]))[0]


def embed(wav, frames):
    e, v = spk.embed_audio(wav.to(dev), torch.ones(1, frames, dtype=torch.bool, device=dev))
    return torch.cat([e, v], -1)


def run_sample(seed, prompt_tokens, tokens, prompt_feats, emb, gen_frames=None):
    torch.manual_seed(seed)
    out = model.sample(
        tokens=[tokens], prompt_tokens=[prompt_tokens], prompt_features=prompt_feats[None],
        prompt_features_lens=torch.tensor([prompt_feats.size(0)], device=dev),
        features_lens=None if gen_frames is None else torch.tensor([gen_frames], device=dev),
        duration="predict" if gen_frames is None else "real",
        t_shift=0.5, num_step=16, guidance_scale=G, cutoff_hz=CUT, speaker_embedding=emb)
    return out[0][0, : int(out[1][0])]


rows = list(csv.reader(open(f"{D}/test_sr8000.tsv", encoding="utf-8"), delimiter="\t"))[:N]
# With a cuts manifest as the fourth argument, only A and S run, on those
# utterances: the training rollout on the data it was trained on.
CUTS = sys.argv[4] if len(sys.argv) > 4 else None
if CUTS:
    from lhotse import load_manifest_lazy
    cuts = []
    for c in load_manifest_lazy(CUTS):
        if 4.0 <= c.duration <= 10.0:
            cuts.append(c)
        if len(cuts) == N:
            break
acc = {}
with torch.no_grad():
    for i, row in enumerate(cuts if CUTS else (r[:4] for r in rows)):
        if CUTS:
            uid, ptext = row.id, row.supervisions[0].text
            wav = torch.from_numpy(row.load_audio()).float()[:1]
            sr = row.sampling_rate
        else:
            uid, ptext, pwav, text = row
            wav, sr = torchaudio.load(f"{D}/prompts/clean/{uid}.wav")
        wav = torchaudio.functional.resample(wav, sr, SR) if sr != SR else wav
        clean = feats(wav)
        T = clean.size(0)
        P = int(0.3 * T)
        full = ids(ptext)
        k = max(1, min(len(full) - 1, round(len(full) * P / T)))
        seed = 1000 + i

        # Training degradation of the whole utterance, as compute_fbank_loss does.
        masked, degraded = resample_band_limit(
            audio=wav.to(dev), audio_lens=torch.tensor([wav.size(1)], device=dev),
            cutoff_hz=torch.tensor([CUT], device=dev), fbank=fbank, num_frames=T,
            sampling_rate=SR, return_audio=True)
        masked = masked[0].to(clean.dtype) * FS
        if os.environ.get("FLOOR"):
            # The mel-floor degradation training uses without --resample-degrade.
            masked = model._band_limit(clean[None], torch.tensor([CUT], device=dev), randomize=False)[0][0]
        gt_span = clean[P:]

        # A: training path.
        text_cond, pad = model.forward_text_train(tokens=[full], features_lens=torch.tensor([T], device=dev))
        gen_mask = torch.zeros(1, T, dtype=torch.bool, device=dev)
        gen_mask[:, P:] = True
        speech = torch.where(gen_mask[..., None], 0, masked[None])
        _, band_mask = model._band_limit(clean[None], torch.tensor([CUT], device=dev), randomize=False)
        prompt_mask = ~gen_mask & ~pad
        e, v = spk.embed_audio(degraded, prompt_mask)
        emb_train = torch.cat([e, v], -1)
        torch.manual_seed(seed)
        noise = torch.randn(1, T, clean.size(1), device=dev)
        xA, _, xS = model.identity_rollout(
            noise=noise, text_condition=text_cond, speech_condition=speech, padding_mask=pad,
            band_mask=band_mask, cutoff_hz=torch.tensor([CUT], device=dev), prompt_mask=prompt_mask,
            speaker_embedding=emb_train, num_step=16, guidance_scale=G, t_shift=0.5, grad_steps=1,
            clean_condition=torch.where(gen_mask[..., None], 0, clean[None]))
        res = {"A": gap(xA[0, P:], gt_span), "S": gap(xS[0, P:], gt_span)}
        if CUTS:
            for key, val in res.items():
                acc.setdefault(key, []).append(val)
            print(i, uid, " ".join(f"{k}:{v[2]:+.1f}/{v[6]:+.1f}" for k, v in res.items()), flush=True)
            continue

        # B: same prompt mel, same noise, through sample().
        res["B"] = gap(run_sample(seed, full[:k], full[k:], masked[:P], emb_train, T - P), gt_span)

        # C: the prompt audio alone through the eval pipeline.
        pa = wav[:, : P * HOP]
        p8 = torchaudio.functional.resample(pa, SR, 8000)
        p24 = torchaudio.transforms.Resample(8000, SR)(p8)
        pf = feats(p24)
        res["C1"] = gap(run_sample(seed, full[:k], full[k:], pf, embed(p24, pf.size(0)), T - P), gt_span)
        p24b = remove_silence(p24, SR, only_edge=False, trail_sil=200)
        p24b, _ = rms_norm(p24b, 0.1)
        pf = feats(p24b)
        res["C2"] = gap(run_sample(seed, full[:k], full[k:], pf, embed(p24b, pf.size(0)), T - P), gt_span)

        # D: the evaluation row itself.
        ref, rsr = torchaudio.load(f"{D}/references/{uid}.wav")
        ref = torchaudio.functional.resample(ref, rsr, SR) if rsr != SR else ref
        gt_ref = feats(ref)
        pw, _ = load_prompt_wav(pwav, sampling_rate=SR)
        pw = remove_silence(pw, SR, only_edge=False, trail_sil=200)
        pw, _ = rms_norm(pw, 0.1)
        pf = feats(pw)
        emb = embed(pw, pf.size(0))
        res["D1"] = gap(run_sample(seed, ids(ptext), ids(text), pf, emb), gt_ref)
        res["D2"] = gap(run_sample(seed, ids(ptext), ids(text), pf, emb, gt_ref.size(0)), gt_ref)

        for key, val in res.items():
            acc.setdefault(key, []).append(val)
        print(i, uid, " ".join(f"{k}:{v[2]:+.1f}/{v[6]:+.1f}" for k, v in res.items()), flush=True)

print(f"\n{ckpt} guidance {G}, n={N}; gen-span minus ground truth, dB")
print("      " + "".join(f"{b:>8s}" for b in BANDS) + "  | energy" + "".join(f"{b:>8s}" for b in BANDS))
for key, vals in acc.items():
    m = torch.tensor(vals).mean(0)
    print(f"{key:5s} " + "".join(f"{x:8.1f}" for x in m[:4]) + "  |       " + "".join(f"{x:8.1f}" for x in m[4:]))
