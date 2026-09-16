#!/bin/bash

# Train and evaluate the band adapter: a small side branch on a frozen released
# ZipVoice, which writes a velocity residual into the mel bins a band-limited
# prompt does not observe and is multiplied by the band mask, so at full band it
# contributes exactly zero and the model is the released checkpoint.
#
# Data preparation and the band-limited evaluation set are shared with the
# extender study and are not repeated here. Run stages 1-4 and stage 8 of
# run_bwe_libritts.sh first; this script starts from the features and the
# evaluation prompts they produce, and writes into the same result root so the
# existing scoring stage sees the new arm.

# Emilia tokens are IPA. lhotse writes manifests with the locale codec, which on
# a Windows machine is cp1252 and cannot encode them, so force UTF-8 mode.
export PYTHONUTF8=1

set -e
set -u
set -o pipefail

# Windows ships a python3.exe App Execution Alias that is on PATH but only
# prints an install prompt, so probe by running it, not by command -v.
if python3 -c "" >/dev/null 2>&1; then PY=python3; else PY=python; fi

if ${PY} -c "import sys; sys.exit(0 if sys.platform == 'win32' else 1)"; then
      export PYTHONPATH="$(cd ../.. && pwd -W);${PYTHONPATH:-}"
else
      export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"
fi

stage=${stage:-1}
stop_stage=${stop_stage:-3}

# The degraded prompt conditions. 24000 is the undegraded control: the branch is
# masked to zero there, so that arm is the released model and is expected to
# reproduce the stock numbers exactly.
eval_rates=${eval_rates:-"4000 8000 16000"}

# Solver steps at inference. 16 everywhere in the reported table; overridable
# because the branch is fitted on the true interpolation path, one step at a
# time, and then deployed inside a multi-step solve where its own residual has
# been entering ``x_t`` since the first step. Whether that compounding is what
# costs the identity is answerable by shortening the solve, provided the stock
# arm is resynthesised at the same step count -- fewer steps degrades every
# system, so only the gap between them carries the answer.
num_step=${num_step:-16}

wandb=${wandb:-0}
wandb_project=${wandb_project:-zipvoice-adapter}

data_root=${ZIPVOICE_DATA:-/c/zipvoice_data}
download_dir=${data_root}/download
fbank_dir=${data_root}/data/fbank
exp_root=${data_root}/exp
eval_dir=${data_root}/data/bwe_eval
results_dir=${exp_root}/results

tokenizer=emilia
token_file=${download_dir}/zipvoice/tokens.txt

exp_dir=${exp_dir:-${exp_root}/adapter}
num_iters=${num_iters:-30000}

# Where the branch starts. The released checkpoint, for an arm trained from
# scratch; a previous arm's averaged checkpoint, for one that only adds a term
# to an already-trained branch -- which is what isolates that term's effect from
# 30k steps of everything else.
init_checkpoint=${init_checkpoint:-${download_dir}/zipvoice/model.pt}
base_lr=${base_lr:-0.001}
# 0 samples cutoffs uniformly throughout, which is what a finetune off an
# already-trained branch wants: the easy-to-balanced ramp exists to get a fresh
# branch started.
curriculum_rho=${curriculum_rho:-0.7}

# Extra conditioning streams, concatenated onto the frequency encoder's output.
# They are part of what the branch is *given*, not of what it is scored on, so
# they belong in the branch's own training rather than in a finetune: a branch
# trained for 30k steps without them has already learnt to do without them.
#
#   --use-speaker-cond  the prompt's time-pooled per-bin mean and standard
#                       deviation, plus a has-prompt flag, taken from the
#                       band-limited condition mel. Not from the waveform and not
#                       from a pretrained speaker model: training band-limits only
#                       in the mel domain, so the audio in the batch is clean and
#                       full band and anything read off it would be the high band
#                       the branch exists to invent; and SIM-o is computed with a
#                       pretrained speaker model, so conditioning on one would
#                       make the metric uninterpretable.
#   --use-rate-cond     the cutoff in Hz, sinusoidally embedded, broadcast over
#                       time. How far the extrapolation has to reach, as a number.
#
# A pitch stream was tried and dropped: -0.087 to -0.231 SIM-o at every cutoff
# against a 0.008 noise floor, for no UTMOS gain, with validation identical on
# all three loss terms. Pitch is informative but *not identifying*, so the branch
# could satisfy the objective by writing the conditional mean over every speaker
# at that pitch. Hence the rule this arm follows: a stream must carry identity.
#
# --variance-loss-weight guards the same failure directly. It hinges the batch's
# between-utterance dispersion of the written band against the data's, so a
# branch that writes one average high band pays for it. Always computed and
# logged, so a run at 0 measures the term before a run spends gradient on it.
#
# The ablation is one full run per arm, each with its own exp_dir and system:
#   exp_dir=${exp_root}/adapter       system=adapter       spk=0 rate=0 (baseline)
#   exp_dir=${exp_root}/adapter_rate  system=adapter_rate  spk=0 rate=1
#   exp_dir=${exp_root}/adapter_spk   system=adapter_spk   spk=1 rate=1 var=0
#   exp_dir=${exp_root}/adapter_spkv  system=adapter_spkv  spk=1 rate=1 var>0
use_speaker_cond=${use_speaker_cond:-1}
use_rate_cond=${use_rate_cond:-1}
variance_loss_weight=${variance_loss_weight:-0.0}

# The adversarial terms, off unless a start step is given. They exist because
# every other term in the objective is satisfied by a speaker-generic band: flow
# matching scores the frozen backbone's velocity, the L1 is minimised by the
# conditional mean, and the dispersion floor is permutation-invariant -- a branch
# that writes the right set of high bands but attaches them to the wrong speakers
# scores perfectly on it. The cross-band discriminator is the first term that
# asks whether *this* invented band follows *this* observed low band.
#
# Two views, not three: the harmonic discriminator needs F0, and the only F0
# available at training time would come from the band-limited mel the branch is
# already conditioned on.
#
# The weights are not the vocoder-GAN defaults they were first copied from. A
# run at adv=0.1 with the discriminators on 2e-4 diverged: by 2k steps the
# discriminator had separated on the spec view -- its fake hinge ``relu(1 + f)``
# saturated at zero, so it drew gradient from real samples only -- while the
# generator's ``-E[D(fake)]``, which has no lower bound, doubled every 1000 steps
# until it was twice the reconstruction term, and ``residual_rms`` rose 26% as
# the branch shouted to beat it. The cause is capacity: the branch is 1.9M
# parameters writing a masked band, against 1.6M of discriminator, so the
# discriminator wins by default. The learning rate is what corrects that, and
# the weight keeps the term subordinate to reconstruction while it does.
#
# Both endpoints were measured, 1000 steps each, reading the running mean of
# ``d_loss`` (2.0 when the discriminator is at chance). 2e-4 reached 1.60 by step
# 1000 and diverged. 2e-5 went 1.951 -> 1.908, a real descent but at 0.04 per
# 1000 steps, which over the 10k budget lands near 1.55 -- so the critic that
# feature matching matches against would still be close to random for most of
# the run. 6e-5 is between them and reaches 1.62 by step 2000. The weight, not
# the learning rate, is what makes the upper side safe: at 0.02 even a return to
# the 1.75 that diverged the first run contributes 0.035 against a reconstruction
# term of roughly 0.09, where the 0.1 it ran at made that same value twice
# reconstruction.
#
#   exp_dir=${exp_root}/adapter_gan  system=adapter_gan  spk=1 rate=1 var=0 gan=0
adapter_gan_start_step=${adapter_gan_start_step:--1}
adapter_adv_loss_weight=${adapter_adv_loss_weight:-0.02}
adapter_feat_match_loss_weight=${adapter_feat_match_loss_weight:-2.0}
adapter_disc_lr=${adapter_disc_lr:-6e-5}

# Whether the flow state handed to the network keeps the bins above the cutoff.
# It does by default, which is how every arm up to here was trained, and that is
# a leak: ``x_t`` is built from the clean target, so the band the branch is meant
# to invent is already in the state it reads, and copying it scores as well as
# synthesising it. Measured on the 30k checkpoint, removing those bins at
# inference time raises the endpoint error in the written band by 7.5x at
# t=0.1 -- the branch's answer is mostly a function of information that does not
# exist when it runs for real. Stage 5 retrains with them hidden.
mask_xt=${mask_xt:-false}

# How often a checkpoint lands. 2000 was the 30k run's setting and it cost that
# whole run: stopping it at update 1350 left the experiment directory holding
# nothing but args.json. Each file is about 8 MB, so 1000 bounds the loss at
# roughly half an hour of compute for no meaningful disk cost.
save_every=${save_every:-1000}

# The arm name stages 2 and 3 write and score under. Every other arm -- a
# conditioning-stream ablation, or stage 4's finetune -- is its own exp_dir, so
# those two stages are re-run over each with e.g.
# `system=adapter_id exp_dir=${exp_root}/adapter_id num_iters=10000`.
system=${system:-adapter}

# The speaker encoder the branch's pretrained speaker stream reads. Defined here
# rather than beside stage 4, which first trained with it, because stage 2 has to
# hand inference the same encoder: a branch trained with the stream and then run
# without it is given a zero embedding flagged invalid, an input training only
# ever saw for unusably short prompts, and the SIM-o that comes back understates
# the model. Harmless for arms trained without the stream, which ignore it.
speaker_ckpt=${speaker_ckpt:-${download_dir}/tts_eval_models/speaker_similarity/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the adapter against the frozen released model"
      # --adapter-trainable adapter leaves every released tensor out of the
      # optimizer, not merely zero-gradient: ScaledAdam keeps a per-tensor scale,
      # so exclusion is what guarantees the released weights are untouched.
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr ${base_lr} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every} --valid-interval ${save_every} \
            --checkpoint ${init_checkpoint} \
            --adapter-trainable adapter \
            --curriculum-steps ${num_iters} \
            --curriculum-rho ${curriculum_rho} \
            --full-band-prob 0.2 \
            --use-speaker-cond ${use_speaker_cond} \
            --use-rate-cond ${use_rate_cond} \
            --mask-xt ${mask_xt} \
            --variance-loss-weight ${variance_loss_weight} \
            --adapter-gan-start-step ${adapter_gan_start_step} \
            --adapter-adv-loss-weight ${adapter_adv_loss_weight} \
            --adapter-feat-match-loss-weight ${adapter_feat_match_loss_weight} \
            --adapter-disc-lr ${adapter_disc_lr} \
            --exp-dir ${exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Synthesise from band-limited prompts"
      # model.json and tokens.txt are written into the experiment directory by
      # the trainer. model.json in particular is the *resolved* config, carrying
      # whichever conditioning streams that run enabled, so copying the template
      # over it here would build an inference model missing them and the
      # checkpoint would not load.
      for rate in ${eval_rates}; do
            ${PY} -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice_adapter \
                  --model-dir ${exp_dir}/ \
                  --checkpoint-name iter-${num_iters}-avg-2.pt \
                  --tokenizer ${tokenizer} \
                  --test-list ${eval_dir}/test_sr${rate}.tsv \
                  --res-dir ${results_dir}/${system}_sr${rate} \
                  --speaker-encoder-checkpoint ${speaker_ckpt} \
                  --num-step ${num_step}
      done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Score the adapter arm"
      # Scoring lives in run_bwe_libritts.sh; only the arm list differs, so that
      # stage is reused rather than duplicated. The stock and cascade arms it
      # compares against were scored by earlier runs against the same evaluation
      # set and are not resynthesised.
      eval_systems="${eval_systems:-${system}}" \
            stage=10 stop_stage=10 eval_rates="${eval_rates}" \
            bash run_bwe_libritts.sh
fi

# Stage 4 finetunes the trained branch against two further losses, measured on
# audio rather than on the mel. The 30k run above raised UTMOS at a 4 kHz cutoff
# from 2.80 to 3.96 but dropped SIM-o below both baselines at every rate, by more
# than ten times the n=76 noise floor at 4 kHz. Neither of its loss terms carries
# speaker identity: flow matching scores the frozen backbone's velocity, and the
# L1 is minimised by whatever high band is closest on average, which belongs to
# nobody. So the branch is charged for the speaker distance between the vocoded
# endpoint estimate and the vocoded target.
#
# Weights are calibrated, not guessed. Run once with them at 1e-6, which builds
# the losses and logs their magnitudes while leaving the objective effectively
# untouched, then set each so its contribution is roughly 10-30% of the total.
id_exp_dir=${id_exp_dir:-${exp_root}/adapter_id}
id_num_iters=${id_num_iters:-10000}
identity_weight=${identity_weight:-1e-6}
fmatch_weight=${fmatch_weight:-1e-6}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Finetune with identity and feature matching losses"
      # --curriculum-rho 0 samples cutoffs uniformly from the start: the
      # curriculum exists to ease a fresh branch into wide bandwidths, and this
      # branch is already trained. Half the learning rate of stage 1, since this
      # is meant to correct the residual, not relearn it.
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.0005 \
            --num-iters ${id_num_iters} \
            --save-every-n 2000 --valid-interval 2000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --identity-loss-weight ${identity_weight} \
            --fmatch-loss-weight ${fmatch_weight} \
            --speaker-encoder-checkpoint ${speaker_ckpt} \
            --exp-dir ${id_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${id_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${id_exp_dir}
fi

# Stage 5 retrains the branch with the flow state's high band hidden, which is
# the leak the 30k arm was fitted through. It finetunes rather than starting
# over: the frequency encoder and the cross-attention have learned to read the
# band-limited prompt, and only the policy on top of them has to change, from
# preserving what ``x_t`` already carries to writing the band itself. Expect the
# loss to jump at step 0 and settle above the 30k arm's -- the task is now the
# harder one it was always meant to be, so the numbers are not comparable across
# the two arms, only the audio is.
mask_exp_dir=${mask_exp_dir:-${exp_root}/adapter_mask}
mask_num_iters=${mask_num_iters:-10000}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Finetune with the flow state's high band hidden"
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr ${base_lr} \
            --num-iters ${mask_num_iters} \
            --save-every-n 1000 --valid-interval 1000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --mask-xt 1 \
            --exp-dir ${mask_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${mask_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${mask_exp_dir}
fi

# Stage 6 charges the branch for the damage its writes do outside its band.
#
# The branch writes only above the cutoff, but it writes into a *velocity*: the
# solver integrates that into the next state, and the frozen backbone reads the
# whole spectrum, so its prediction moves in the observed bins too. Roughly 94%
# of the speaker similarity lost against the stock model sits in bins the branch
# never touches, and no earlier arm had any gradient into them -- within one
# step the low band of the endpoint estimate comes from the backbone's own
# velocity, which does not depend on the residual at all.
#
# --drift-loss-weight measures that response directly: perturb the state along
# the residual, run the backbone again with the same dropout draw, and charge
# the low-band part of the difference. The gradient reaches the branch through the
# perturbation. It cannot cancel the response -- it still may not write there --
# but it can choose a high band that provokes less of one, and the high bands
# that leave the backbone's low band alone are the ones consistent with the
# speech being generated. So the term should raise similarity without costing
# written energy. If instead it trades them off along the same curve
# --residual-scale already traces at inference, it is only a brake and the
# mechanism is wrong.
#
# Cost: one extra backbone forward per step, no extra parameters, nothing added
# to the checkpoint.
drift_exp_dir=${drift_exp_dir:-${exp_root}/adapter_drift}
drift_num_iters=${drift_num_iters:-10000}
drift_loss_weight=${drift_loss_weight:-1.0}
drift_dt=${drift_dt:-0.5}
# The term as first written charges ``|J.r|^2``, which the branch can lower by
# aiming better or by writing less, and it took the second route. Measured at
# weight 10 against the untermed branch at matched steps: validation drive 2.18x
# lower at step 4000 (0.0378 against 0.0825), and band energy +0.76 dB over the
# frozen model at step 3000, +2.07 dB at 6000 -- gaining 1.31 dB per 3000 steps
# where the untermed branch reaches +11.6 dB, so roughly 21k further steps to
# parity. That is the brake this stage's comment says would falsify the
# mechanism, so the normalised form is the default: ``|J.r|^2 / |r|^2`` is the
# gain along the residual, invariant to its scale, and only its direction is
# charged.
#
# The weight is not carried over, because normalising rescales the term: at step
# 6000 the ratio was 0.68 against a total loss of 0.18, so the 10 used for the
# unnormalised form would contribute 6.8 and swamp the objective. 0.05 puts it
# near 19%, the same 10-30% band the other terms are calibrated to.
drift_normalize=${drift_normalize:-1}
# Lower than the other stages' 72. The drift term's second backbone forward is
# in the graph -- the gradient it exists to create flows back through it to the
# residual -- so a second full set of backbone activations is live at once. At
# 72 the step peaked at 15.8 GiB of this machine's 16.3 GiB.
#
# The batch duration is the weaker of the two knobs, and the measurements say so:
# 44 plateaued at 12.65 GiB and 36 at 12.81 GiB, then 36 climbed to 13.9 GiB once
# a long enough utterance turned up. Attention is quadratic in the longest single
# utterance, not in the batch's total duration, so --max-len is what sets the
# high-water mark the caching allocator then holds; 20 -> 14 is the change that
# caps it. Raise both on a bigger card.
drift_max_duration=${drift_max_duration:-36}
drift_max_len=${drift_max_len:-14}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Train with the low-band drift penalty"
      # --drift-dt is a finite-difference step, not a solver step: the low-band
      # response is linear in the perturbation, so its size costs no accuracy
      # and is set by fp16's resolution instead. See the flag's help.
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 \
            --max-duration ${drift_max_duration} --max-len ${drift_max_len} \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 0 --base-lr ${base_lr} \
            --num-iters ${drift_num_iters} \
            --save-every-n 1000 --valid-interval 1000 \
            --checkpoint ${init_checkpoint} \
            --adapter-trainable adapter \
            --curriculum-rho ${curriculum_rho} \
            --curriculum-steps ${drift_num_iters} \
            --full-band-prob 0.2 \
            --use-pretrained-spk 1 \
            --speaker-encoder-checkpoint ${speaker_ckpt} \
            --mask-xt 0 \
            --resample-degrade 1 \
            --drift-loss-weight ${drift_loss_weight} \
            --drift-dt ${drift_dt} \
            --drift-normalize ${drift_normalize} \
            --exp-dir ${drift_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${drift_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${drift_exp_dir}
fi

# Stage 7 charges the branch for the *level* of the band it writes.
#
# The two-term objective scores the invented band element by element, and most
# of that band is not a function of the low band, so the L1's per-element
# optimum is a shrunken answer. Measured per octave against the evaluation set's
# own references at a 4 kHz cutoff, each system relative to its own 100-2000 Hz
# band:
#
#   band          3-4k    4-6k    6-8k   8-12k
#   references   -19.9   -18.0   -19.9   -22.6
#   cascade      -20.2   -19.7   -23.0   -26.0
#   adapter      -29.0   -27.0   -29.5   -32.3
#
# The shape is right and the level is 9-10 dB down. That is not a failure of
# capacity or of conditioning -- it is what the objective asks for. Note also
# that the single hi/lo ratio band_energy.py reports does not see this: power
# falls steeply with frequency, so that ratio is dominated by the octave just
# above the cutoff and an arm empty from 4-12 kHz can still post +10 dB. Judge
# this stage per octave.
#
# --band-loss-weight pools the endpoint error over tiles of frames and bins
# before taking the absolute value, at three scales, so error that cancels
# inside a tile costs nothing and only the tile's net level offset is charged.
# The unpredictable fine structure is therefore not penalised for being wrong,
# which is the whole difference from the reconstruction term.
#
# If it works, the octave profile rises toward the references while SIM-o holds:
# the branch is being asked for more of the band it already writes, not for a
# different one. If SIM-o falls with the level, then energy and identity trade
# off along one curve here as they did under the drift term, and writing the
# band louder is not the missing piece.
band_exp_dir=${band_exp_dir:-${exp_root}/adapter_band}
band_num_iters=${band_num_iters:-10000}
# Calibrate rather than guess, as with the other added terms: band_loss is
# logged whatever its weight, so run stage 7 once at 0 and read it against
# recon_loss before setting this.
band_loss_weight=${band_loss_weight:-1.0}

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Finetune with the multi-resolution band level loss"
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.0005 \
            --num-iters ${band_num_iters} \
            --save-every-n 1000 --valid-interval 1000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --band-loss-weight ${band_loss_weight} \
            --exp-dir ${band_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${band_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${band_exp_dir}
fi

# Stage 8 trains the branch on the states it will actually be given.
#
# Every arm up to here was fitted one step from clean features, on the flow
# path, where x_t already carries the target's high band scaled by t. The
# branch therefore learns a small correction to a nearly-correct state. At
# inference it applies that correction to a state carrying only its own writes,
# and lands 9-10 dB short after sixteen steps. Measured directly, from the
# trained checkpoint with no weight change, the branch writes 0.121 rms on the
# flow path and 0.068 -- 44% less -- on a state it produced itself twelve steps
# in, while recon_loss doubles and band_loss nearly triples. That error exists
# nowhere on the flow path, which is why no single-step term ever saw it: they
# all score x1_hat = xt + (1-t)*vt, an extrapolation the sixteen-step solve
# never takes.
#
# --rollout-steps runs the solver from noise under no_grad with the branch in
# the loop, hands the resulting state to the loss, and takes one attached step
# from it. The rollout is detached, so this costs k extra forwards of wall
# clock and one activation peak of memory rather than k. It is not a decoupled
# trajectory: the branch writes into every rollout step.
#
# --rollout-prob below 1.0 is scheduled sampling, and it is the difference
# between this and the earlier rollout arm that collapsed the written band to
# +1.9 dB over stock. Trained on self-generated states alone and from scratch,
# a branch that writes nothing sees a state that has nothing, so the target is
# unreachable and the gradient is poor; the flow-path batches keep that fixed
# point from being stable, and the warm start means the branch already writes
# something on the first rollout.
#
# Note what this stage is *not*. It adds no loss term. The endpoint statistics
# on rolled-out states are close to unbiased over tiles -- band_bias is +0.006
# at k=4 falling to +0.002 at k=12 -- so there is no systematic level offset
# for a band term to correct, and stage 7's term should not be revived here.
# What rollout supplies is the input distribution, not a new objective.
#
# If it works, the octave profile rises toward the references at a residual
# scale of 1.0 and validation residual_rms on rolled-out states rises toward
# the flow-path value. If the profile rises and SIM-o falls with it, then
# exposure was not the missing piece either and energy and identity are one
# curve however the branch is trained.
rollout_exp_dir=${rollout_exp_dir:-${exp_root}/adapter_selfforce}
rollout_num_iters=${rollout_num_iters:-10000}
rollout_steps=${rollout_steps:-16}
# Half the batches rolled out, half on the flow path.
rollout_prob=${rollout_prob:-0.5}

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Finetune on rolled-out states"
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.0005 \
            --num-iters ${rollout_num_iters} \
            --save-every-n 1000 --valid-interval 1000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --rollout-steps ${rollout_steps} \
            --rollout-k -1 \
            --rollout-prob ${rollout_prob} \
            --exp-dir ${rollout_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${rollout_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${rollout_exp_dir}
fi

# Stage 9: train the branch on decoupled full solves.
#
# The branch reads the frozen backbone's hidden state at every solver step and
# its writes never re-enter what the backbone reads, so the low band of the
# sample is bit-identical to the frozen model's. Identity is then preserved by
# construction rather than by a penalty, which matters because ~94% of the
# adapter's SIM-o loss sits in bins it never writes -- the damage is the
# feedback, not the written band.
#
# Measured at inference on the stage-4 checkpoint, decoupling holds identity
# almost flat in the dose: SIM-o 0.368 at 1.0 and 0.354 at 2.0, where the
# coupled branch is already at 0.198 by 1.5. What it does not do is fill the
# band -- run decoupled, a coupled-trained branch is 20-35 dB emptier than the
# coupled arm above 3 kHz, because its own writes feeding back is what
# amplified it across the 16 steps. That is a train/test mismatch, not a
# property of the architecture, and this stage is the arm that removes it.
#
# The full solve is affordable here only because of the decoupling. The branch
# cannot influence the trajectory, so all ${rollout_steps} backbone forwards run
# without building a graph and only the adapter forwards carry one; the adapter
# is 256-dim and two layers against a 512-dim 16-layer backbone. Verified: zero
# backbone parameters accumulate gradient, and the adapter's do.
#
# The point of paying for it is the objective. Every loss in this study so far
# has scored x1_hat one step from ground-truth features, where the 9-10 dB level
# deficit does not exist -- the solve manufactures it. Here x1_hat is the sample
# the solver returns. fm_loss is undefined in this mode and reads zero, so
# recon_loss and band_loss are the objective.
decoup_exp_dir=${decoup_exp_dir:-${exp_root}/adapter_decoupsolve}
decoup_num_iters=${decoup_num_iters:-10000}

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
      echo "Stage 9: Finetune on decoupled full solves"
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.0005 \
            --num-iters ${decoup_num_iters} \
            --save-every-n 1000 --valid-interval 1000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --rollout-steps 16 \
            --rollout-decouple 1 \
            --exp-dir ${decoup_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${decoup_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${decoup_exp_dir}
fi

# Stage 10: train the branch as a band filler.
#
# Everything before this has the branch predicting a velocity that is added to
# the backbone's, which puts the two in competition: the branch has to steer a
# trajectory it does not own, and whatever it writes is integrated across 16
# steps, where it shrinks. That shrinkage is what the dose and tilt knobs at
# inference exist to undo, and the training-free sweep showed they buy a cheaper
# way along the energy/identity curve, not a way off it -- every arm that added
# 8-11 dB above 4 kHz cost identity monotonically, down to SIM-o 0.174.
#
# Here the branch predicts no velocity. It reads the frozen backbone's hidden
# state at every solver step, carries its own state across all of them, and
# emits the missing band directly as a mel that is added to the solved sample.
# Nothing it produces is integrated and nothing it writes re-enters what the
# backbone reads, so the low band is the frozen model's exactly and there is no
# shrinkage to correct.
#
# It also closes the exposure gap rather than mitigating it: the hidden states
# the branch reads are the ones the solver really visits, because this is the
# solve. Training and sampling are the same computation.
#
# The failure mode to watch is the opposite of the old one. A velocity field
# carried stochasticity that kept the high band off its conditional average; a
# direct predictor under L1 gives the conditional median, which is flatter than
# any real high band. Symptom: band_loss falling while the per-octave profile
# comes out smooth and quiet. Check with local/band_profile.py against the
# references, not against the loss.
bandfill_exp_dir=${bandfill_exp_dir:-${exp_root}/adapter_bandfill}
bandfill_num_iters=${bandfill_num_iters:-10000}
# The level term is the point of this arm: a direct read-out under L1 alone
# settles on the conditional median, and the median of a high band is quieter
# than any high band. Nothing else in the objective asks for level.
bandfill_band_loss_weight=${bandfill_band_loss_weight:-1.0}

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
      echo "Stage 10: Finetune as a band filler"
      ${PY} -m zipvoice.bin.train_zipvoice_adapter \
            --world-size 1 --use-fp16 1 --max-duration 72 --max-len 20 \
            --model-config conf/zipvoice_adapter.json \
            --tokenizer ${tokenizer} --token-file ${token_file} \
            --dataset libritts --manifest-dir ${fbank_dir} \
            --wandb ${wandb} --wandb-project ${wandb_project} \
            --finetune 1 --base-lr 0.0005 \
            --num-iters ${bandfill_num_iters} \
            --save-every-n 1000 --valid-interval 1000 \
            --checkpoint ${exp_dir}/iter-${num_iters}-avg-2.pt \
            --adapter-trainable adapter \
            --curriculum-rho 0.0 --curriculum-steps 0 \
            --full-band-prob 0.2 \
            --rollout-steps 16 \
            --band-fill 1 \
            --band-loss-weight ${bandfill_band_loss_weight} \
            --exp-dir ${bandfill_exp_dir}

      ${PY} -m zipvoice.bin.generate_averaged_model \
            --iter ${bandfill_num_iters} --avg 2 \
            --model-name zipvoice_adapter \
            --exp-dir ${bandfill_exp_dir}
fi
