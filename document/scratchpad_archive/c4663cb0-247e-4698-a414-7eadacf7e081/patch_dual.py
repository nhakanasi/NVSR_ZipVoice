import sys

ROOT = sys.argv[1]


def patch(path, pairs):
    p = f"{ROOT}/{path}"
    s = open(p, encoding="utf-8", newline="").read()
    crlf = "\r\n" in s
    if crlf:
        s = s.replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "w", encoding="utf-8", newline="").write(s)
    print("ok", path)


# ---------------------------------------------------------------- model
patch("zipvoice/models/zipvoice_dual.py", [
("""    band_mask,
    mel_bin_centers,
    sample_cutoff,
)""", """    band_mask,
    mel_bin_centers,
    resample_band_limit,
    sample_cutoff,
)"""),

("""        self.mel_floor = LOG_MEL_FLOOR * feat_scale
""", """        self.feat_scale = feat_scale
        self.mel_floor = LOG_MEL_FLOOR * feat_scale
"""),

# A helper for the memory, placed before _backbone_step.
("""    def _backbone_step(
        self,""", """    def _memory(
        self,
        t: Tensor,
        y: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        \"\"\"The backbone's hidden state on the branch's own state ``y``.

        The branch reads the backbone's hidden state as its memory, and that
        state has to be computed on ``y``, not on the stock trajectory ``x``.
        At teacher-forced training the two are one state, the flow marginal, so
        the memory carries the band the branch is writing. At inference ``x``
        never receives the branch's writes, so a memory computed on it carries
        nothing above the cutoff: measured on the first dual run (21k, sr8000),
        taking it from ``y`` instead raised 8-12 kHz by 14 dB with SIM-o
        unchanged. The velocity still comes from ``x``, so the low band stays
        the frozen model's.

        Always the conditional pass: the hidden state ``_backbone_step`` hands
        back under guidance is the conditional half's, which is this.
        \"\"\"
        _, hidden = self.forward_fm_decoder(
            t=t,
            xt=y,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            return_hidden=True,
        )
        return hidden

    def _backbone_step(
        self,"""),

("""        rollout_prob: float = 0.0,
        rollout_steps: int = 1,
    ) -> Dict[str, Tensor]:""", """        rollout_prob: float = 0.0,
        rollout_steps: int = 1,
        audio: Optional[Tensor] = None,
        audio_lens: Optional[Tensor] = None,
        fbank: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:"""),

("""            rollout_steps: number of Euler steps from pure noise to ``t``. Must
                be positive when rollout is enabled.
""", """            rollout_steps: number of Euler steps from pure noise to ``t``. Must
                be positive when rollout is enabled.
            audio, audio_lens, fbank: when given, the prompt is band-limited by
                really resampling the waveform and re-extracting the mel, which
                is what the evaluation prompts went through, instead of by the
                mel floor. The mask is then the deterministic one inference
                projects with.
"""),

("""            band_limited, mask = band_limit(
                features=features,
                cutoff_hz=cutoff_hz,
                centers=self.mel_centers,
                floor=self.mel_floor,
                transition_bins=self.band_transition_bins,
                floor_jitter=self.band_floor_jitter,
                noise_std=self.band_noise_std,
            )
""", """            if audio is not None:
                band_limited = (
                    resample_band_limit(
                        audio=audio,
                        audio_lens=audio_lens,
                        cutoff_hz=cutoff_hz,
                        fbank=fbank,
                        num_frames=features.size(1),
                    ).to(features.dtype)
                    * self.feat_scale
                )
                mask = self._band_mask(cutoff_hz)
            else:
                band_limited, mask = band_limit(
                    features=features,
                    cutoff_hz=cutoff_hz,
                    centers=self.mel_centers,
                    floor=self.mel_floor,
                    transition_bins=self.band_transition_bins,
                    floor_jitter=self.band_floor_jitter,
                    noise_std=self.band_noise_std,
                )
"""),

# Rollout loop: velocity from x_roll, memory from y_roll.
("""                    vr, rollout_hidden = self.forward_fm_decoder(
                        t=tr,
                        xt=x_roll,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                        return_hidden=True,
                    )
""", """                    vr = self.forward_fm_decoder(
                        t=tr,
                        xt=x_roll,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                    )
                    rollout_hidden = self._memory(
                        t=tr,
                        y=y_roll,
                        text_condition=text_condition,
                        speech_condition=speech_condition,
                        padding_mask=padding_mask,
                    )
"""),

# Target time: memory from the branch's state on rollout rows.
("""            x1_backbone = backbone_xt + (1.0 - t) * vt
""", """            x1_backbone = backbone_xt + (1.0 - t) * vt
            # On teacher rows the two states are one and the hidden state above
            # is already the branch's. On rollout rows it was computed on the
            # stock trajectory, which inference never hands the branch.
            if rollout_mask.any():
                hidden = self._memory(
                    t=t,
                    y=branch_xt,
                    text_condition=text_condition,
                    speech_condition=speech_condition,
                    padding_mask=padding_mask,
                )
"""),

# Sampler.
("""            v, hidden = self._backbone_step(
                t=t,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
            )
            branch_x1 = self.branch(""", """            v, _ = self._backbone_step(
                t=t,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
            )
            hidden = self._memory(
                t=t,
                y=y,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
            )
            branch_x1 = self.branch("""),
])

# ---------------------------------------------------------------- trainer
patch("zipvoice/bin/train_zipvoice_dual.py", [
("""        fidelity but add one frozen-backbone and one branch forward each.\""",
    )
""", """        fidelity but add two frozen-backbone forwards (velocity on the stock
        state, memory on the branch's) and one branch forward each.\""",
    )

    parser.add_argument(
        "--rollout-start-step",
        type=int,
        default=0,
        help=\"\"\"Rollout is off before this step. A fresh branch's rollouts are
        noise, and training on them teaches nothing.\"\"\",
    )

    parser.add_argument(
        "--rollout-ramp-steps",
        type=int,
        default=0,
        help=\"\"\"After --rollout-start-step, --rollout-prob is reached linearly
        over this many steps. 0 switches it on at once.\"\"\",
    )

    parser.add_argument(
        "--resample-degrade",
        type=str2bool,
        default=False,
        help=\"\"\"Band-limit the prompt by resampling the waveform, as the
        evaluation prompts are, instead of by flooring the mel. The mel floor
        sits 9-13 log-units below a real 24k->8k->24k file above the cutoff,
        so a branch conditioned on it meets a different prompt at evaluation.
        Needs the waveform, so it turns --return-audio on.\"\"\",
    )
"""),

("""    with torch.set_grad_enabled(is_training):

        terms = model(""", """    rollout_prob = 0.0
    if is_training and params.batch_idx_train >= params.rollout_start_step:
        ramp = params.batch_idx_train - params.rollout_start_step + 1
        rollout_prob = params.rollout_prob * min(
            1.0, ramp / max(params.rollout_ramp_steps, 1)
        )

    fbank = None
    if audio is not None:
        fbank = _resample_fbank(device)
        audio = audio.to(device)
        audio_lens = audio_lens.to(device)

    with torch.set_grad_enabled(is_training):

        terms = model("""),

("""            rollout_prob=params.rollout_prob if is_training else 0.0,
            rollout_steps=params.rollout_steps,
        )""", """            rollout_prob=rollout_prob,
            rollout_steps=params.rollout_steps,
            audio=audio,
            audio_lens=audio_lens,
            fbank=fbank,
        )"""),

("""    is_training: bool,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
) -> Tuple[Tensor, MetricsTracker, Optional[dict]]:""", """    is_training: bool,
    discriminators: Optional[Union[nn.Module, DDP]] = None,
    audio: Optional[Tensor] = None,
    audio_lens: Optional[Tensor] = None,
) -> Tuple[Tensor, MetricsTracker, Optional[dict]]:"""),

("""def compute_fbank_loss(""", """_FBANKS = {}


def _resample_fbank(device: torch.device) -> nn.Module:
    \"\"\"The feature extractor's mel module, the one the features were made with.\"\"\"
    if device not in _FBANKS:
        _FBANKS[device] = VocosFbank().fbank.to(device)
    return _FBANKS[device]


def compute_fbank_loss("""),

("""    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
""", """    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    if args.resample_degrade:
        # Forced rather than asserted: the two flags are one decision, and a
        # run that silently fell back to the mel floor would look like a
        # resampling run in the log.
        args.return_audio = True
"""),
])
print("done")
