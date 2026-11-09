# Adapter multistep preservation pilot

Initial weights: `C:\zipvoice_data\exp\adapter\iter-30000-avg-2.pt`.
Isolated source: adapter-arm commit `232420c`, plus local changes here.
Stock parameters remain frozen. Existing FM and high-band L1 losses are retained.
No endpoint regression is applied to generated rollout states.

The auxiliary loss compares two conditional Euler continuations from the same
detached FM training state. One is stock-only (no gradients); one is stock plus
adapter (input gradients through stock remain enabled). Three steps, each at
most 1/16 flow-time, shortened near t=1. Loss averages protected-bin squared
state differences divided by elapsed time squared. Only generated, unpadded
frames and bins where the observation mask is exactly zero are penalized.
One eligible utterance per batch bounds memory. Both continuations run with
dropout disabled; original module modes are restored afterward.

This is a local conditional-window regularizer, not full on-policy training.
The window starts on a clean/noise bridge; it does not replicate the complete
shifted, classifier-free-guided inference solve. No inference code is changed.
It cannot guarantee exact stock waveform preservation, or prevent suppression
of the adapter. Existing high-band quality must be checked against the baseline.

Pilot: 1000 updates, LR 1e-4, preservation weight 100, max duration 24 seconds,
max utterance 12 seconds, FP16, checkpoint every 100, validation every 500.
The 20-step smoke test used max duration 12 and weight 0.1. Its raw preservation
loss was about 2e-5; weight 100 initially gives an auxiliary term around 0.002,
versus approximately 0.2 for the retained objective. This is a heuristic initial
weight, not evidence that the gradient magnitudes are balanced.

Run: `powershell -File train_multistep.ps1` from this worktree.
Resume: pass `-ResumeFrom <checkpoint-path> -NumIters <total-pilot-updates>`.
Training output: `C:\zipvoice_data\exp\adapter_multistep_k3_20260918`.

Tests: `python -m unittest discover -s tests -p test_multistep_preservation.py`.
Tests verify indirect gradient through frozen stock, zero drift at zero adapter
output, and zero penalty when the backbone has no cross-band response.

Before extending the pilot: compare actual 8 kHz-resampled prompts (4 kHz cutoff)
and 16 kHz prompts (8 kHz cutoff), with identical noise, CFG and solver settings.
Check low-band drift, high-band speech structure/listening, and residual strength.
Reduced preservation loss alone is not evidence of a successful adapter.
