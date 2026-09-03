# ZipVoice with bandwidth-extended prompts

This is a research fork of [ZipVoice](https://github.com/k2-fsa/ZipVoice) holding
the implementation for our paper. It adds a mel-domain bandwidth extender to the
conditioning path, so that a prompt recorded at 8, 16 or 22.05 kHz yields the
same quality of 24 kHz synthesis as a clean full-band prompt. The extender is
trained jointly with the TTS backbone, which turns ZipVoice's own flow-matching
loss into a task-aware bandwidth-extension objective, and it is provably the
identity when the prompt is already full-band, so nothing is given up on clean
inputs.

What this fork adds:

| Path | What it is |
| --- | --- |
| `zipvoice/models/modules/bwe.py` | NVSR-style ResUNet extender, band-limiting and the cutoff curriculum |
| `zipvoice/models/modules/bwe_discriminators.py` | Spectrogram, cross-band and harmonic discriminators |
| `zipvoice/models/zipvoice_bwe.py` | `ZipVoiceBWE`: a `ZipVoice` subclass with the extender in front of the speech condition |
| `zipvoice/bin/train_zipvoice_bwe.py` | Three-stage training entry point |
| `egs/zipvoice/run_bwe_finetune.sh` | The recipe: data preparation, the three training stages, inference |
| `egs/zipvoice/run_bwe_libritts.sh`, `run_bwe_vctk_eval.sh` | The LibriTTS and VCTK evaluation sweeps |
| `egs/zipvoice/local/` | Evaluation-set construction, MCD, cost benchmark, paired statistics |

Training runs in three stages (`run_bwe_finetune.sh`, stages 5 to 7):
reconstruction pre-training of the extender alone, adaptation against a frozen
TTS backbone, then joint adversarial fine-tuning. Inference is unchanged apart
from `--model-name zipvoice_bwe`; the prompt's original sampling rate is read
from the file and used as the cutoff.

Reproducing the evaluation:

```bash
cd egs/zipvoice
bash run_bwe_finetune.sh      # train
bash run_bwe_libritts.sh      # in-domain sweep over prompt sampling rates
bash run_bwe_vctk_eval.sh     # out-of-domain sweep
```

## Installation

```bash
python3 -m venv zipvoice && source zipvoice/bin/activate  # optional
pip install -r requirements.txt
```

k2 is required for training and speeds up inference; the version has to match
your PyTorch and CUDA build. See
[README_upstream.md](README_upstream.md#4-install-k2-for-training-or-efficient-inference)
for the exact command, and `requirements_eval.txt` for the extra packages the
evaluation needs.

## Upstream documentation

Inference, the pre-trained model variants, the dialogue models, usage guidance
and the Triton and ONNX deployment paths are unchanged from upstream and are
documented in [README_upstream.md](README_upstream.md), a copy of the README of
[k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice).

## Citation

```bibtex
@article{zhu2025zipvoice,
      title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
      author={Zhu, Han and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Li, Zhaoqing and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2506.13053},
      year={2025}
}

@article{zhu2025zipvoicedialog,
      title={ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching},
      author={Zhu, Han and Kang, Wei and Guo, Liyong and Yao, Zengwei and Kuang, Fangjun and Zhuang, Weiji and Li, Zhaoqing and Han, Zhifeng and Zhang, Dong and Zhang, Xin and Song, Xingchen and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2507.09318},
      year={2025}
}
```

A citation for the bandwidth-extension work will be added once the paper has a
reference.
