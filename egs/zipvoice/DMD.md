# One-pass student by distribution-matching distillation (DMD)

Code: `zipvoice/models/modules/dmd.py`, wired into
`zipvoice/bin/train_zipvoice_adapter.py` behind `--dmd-loss-weight`.
Tests: `python -m unittest tests.test_dmd`.

## Why

The deployable target is the coupled adapter as one pass: one trajectory, one
backbone forward per step, no projection onto the frozen model's low band and
no decoupling. Every way that generator has been trained so far failed for one
of two reasons recorded in `ADAPTER.md` and `document/BWE_RESULTS.md`:

* single-step losses score `x1_hat` from a teacher-forced state and are blind
  to the deficit and the low-band drift the 16-step solve produces;
* regressing the solve needs a high band paired with the *generated* low band,
  and no such pairing exists (sampler inversion and prompt swap both fail).

DMD removes both. It scores the student's solved sample, and it compares
distributions rather than pairs:

```
grad_x  =  (x1_fake - x1_real) / norm       x = student sample, re-noised at t
```

| role | network | prompt | guidance |
| --- | --- | --- | --- |
| student (deployed) | released backbone + band branch + gated decoder LoRA | band-limited | `--dmd-guidance` |
| real score | released backbone (the adapter with a zero band mask, branch not run) | **clean, full band** | `--dmd-real-guidance` |
| fake score | released backbone + a second LoRA, trained on the student's samples | clean, full band | none |

The real score's distribution is what a clean recording would have produced,
so identity and band level are one target. The fake LoRA lives outside the
model (like the discriminators): checkpoints still load strictly as an
ordinary `ZipVoiceAdapter`, so averaging and inference are unchanged. Its
state rides in the checkpoint's extra states under `dmd`, and resumes.

## Schedule

* Every batch: the ordinary FM + reconstruction loss at `--dmd-fm-weight`
  (the anchor to real data), and one fake-score update on a student sample.
* From `--dmd-start-step`, every `--dmd-gen-interval` batches: a student
  rollout (`--dmd-num-step`, graph on the last `--dmd-grad-steps`) and the
  generator loss at `--dmd-loss-weight`. That batch's sample is reused for the
  fake update.
* The fake score starts equal to the teacher. Before it has fitted the
  student, the only generator signal is the guidance direction (guided teacher
  vs unguided fake), hence the fake-only warm-up.

Train at the sampler you deploy: the defaults (16 steps, guidance 1.0,
t_shift 0.5) are `infer_zipvoice.py`'s defaults for `zipvoice_adapter`.

## Run

Start from the LoRA adapter arm's checkpoint if you have one; the released
`model.pt` also works (branch and LoRA then start at their no-op init).

```bash
python3 -m zipvoice.bin.train_zipvoice_adapter \
    --world-size 1 --use-fp16 1 --max-duration 36 --max-len 14 \
    --model-config conf/zipvoice_adapter.json \
    --tokenizer emilia --token-file download/zipvoice/tokens.txt \
    --dataset libritts --manifest-dir ${fbank_dir} \
    --finetune 1 --base-lr 2e-4 --num-iters 6000 \
    --save-every-n 500 --valid-interval 1000 \
    --checkpoint ${init_checkpoint} \
    --adapter-trainable lora --lora-rank 16 \
    --curriculum-rho 0.0 --full-band-prob 0.2 \
    --resample-degrade 1 \
    --dmd-loss-weight 1.0 --dmd-fm-weight 1.0 \
    --dmd-fake-rank 64 --dmd-fake-lr 1e-4 \
    --dmd-start-step 1000 --dmd-gen-interval 5 \
    --dmd-num-step 16 --dmd-guidance 1.0 --dmd-real-guidance 1.0 \
    --dmd-grad-steps 1 --dmd-utts 1 --dmd-max-seconds 8 \
    --dmd-cutoffs 4000,8000 \
    --exp-dir exp/adapter_dmd
```

`--lora-rank` must match the starting checkpoint's if it has LoRA factors.
`--max-duration 36 --max-len 14` follows the drift arms, which also kept two
backbone graphs live; raise it if memory allows. One GPU only.

Then average:

```bash
python3 -m zipvoice.bin.generate_averaged_model \
    --iter 6000 --avg 2 --model-name zipvoice_adapter --exp-dir exp/adapter_dmd
```

and synthesise and score with stages 2 and 3 of `run_adapter_libritts.sh`,
which run plain one-pass inference (no `--project-low-band`, no
`--decouple-residual`):

```bash
stage=2 stop_stage=3 system=adapter_dmd exp_dir=${exp_root}/adapter_dmd num_iters=6000 \
    bash run_adapter_libritts.sh
```

## What to watch

Training log keys:

* `dmdfake_loss`: the fake score's flow matching on student samples. Should
  fall during the warm-up and then track the student.
* `dmd_fake_real_gap`: RMS gap between the two endpoint estimates on the
  generated frames. This is the quantity the generator closes.
* `dmd_grad_rms`, `dmd_norm`: size of the normalised step and its normaliser.
* `residual_rms`: the branch should keep writing; a collapse toward zero is
  the brake failure mode the drift arm showed.

Acceptance, one pass, sr8000: SIM-o >= stock 0.610, UTMOS >= cascade 4.14,
8-12 kHz speech-frame level within ~5 dB of the reference with swing >= 30
(`band_frames.py`). Also check sr4000 and the sr24000 control, which must stay
bit-identical to the released model (the LoRA and branch are gated off there).

Known risks: mode collapse (watch swing and per-utterance dispersion, not
only level); fp16 overflow in the teacher/fake difference at small `1 - t`
(lower `--dmd-t-max` if `dmd_grad_rms` spikes); the teacher's clean-prompt
identity may not be fully recoverable from an 8 kHz prompt, so landing near
the ~0.65 verifier ceiling at 8 kHz is the realistic outcome.
