    def _coupled_step(
        self,
        t: Tensor,
        dt: Tensor,
        x: Tensor,
        y: Tensor,
        batch: Dict[str, Tensor],
        condition: Tensor,
        guidance_scale: Union[float, Tensor] = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """One Euler step of both trajectories, exactly as inference takes it.

        Args:
            t, dt: the current time and the step, scalars or (B, 1, 1).
            x: the backbone's state, which never reads the branch.
            y: the branch's state.
            batch: the conditioning, as :meth:`_condition_batch` returns it.
            condition: the branch's encoded conditioning for this batch.
            guidance_scale: the backbone's classifier-free guidance.

        Returns:
            ``x`` and ``y`` after the step.
        """
        v, hidden = self._backbone_step(
            t=t,
            x=x,
            text_condition=batch["text_condition"],
            speech_condition=batch["speech_condition"],
            padding_mask=batch["padding_mask"],
            guidance_scale=guidance_scale,
        )
        branch_x1 = self.branch(
            t=t,
            xt=y,
            text_condition=batch["text_condition"],
            speech_condition=batch["speech_condition"],
            mask=batch["mask"],
            cutoff_hz=batch["cutoff_hz"],
            prompt_mask=batch["prompt_mask"],
            memory=hidden,
            padding_mask=batch["padding_mask"],
            condition=condition,
        )
        u = endpoint_to_velocity(branch_x1, y, t)

        x = x + v * dt  # the stock trajectory, which never reads the branch
        y = y + u * dt
        if not self.steer_low_band:
            y = self._project(x, y, batch["mask"])
        return x, y

    def _condition_batch(
        self,
        tokens: List[List[int]],
        features: Tensor,
        features_lens: Tensor,
        step: int,
        total_steps: int,
    ) -> Dict[str, Tensor]:
        """Everything a training batch is conditioned on, under a simulated
        band limit. Called under no_grad: none of it depends on the branch.
        """
        text_condition, padding_mask = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        cutoff_hz = sample_cutoff(
            batch_size=features.size(0),
            step=step,
            total_steps=total_steps,
            f_min=self.cutoff_min,
            f_max=self.cutoff_max,
            beta=self.curriculum_beta,
            rho=self.curriculum_rho,
            full_band_prob=self.full_band_prob,
            device=features.device,
        )
        band_limited, mask = band_limit(
            features=features,
            cutoff_hz=cutoff_hz,
            centers=self.mel_centers,
            floor=self.mel_floor,
            transition_bins=self.band_transition_bins,
            floor_jitter=self.band_floor_jitter,
            noise_std=self.band_noise_std,
        )

        # True where the model has to generate, i.e. outside the prompt.
        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1), 0, band_limited
        )

        return {
            "text_condition": text_condition,
            "padding_mask": padding_mask,
            "cutoff_hz": cutoff_hz,
            "mask": mask,
            "speech_condition": speech_condition,
            "prompt_mask": (~speech_condition_mask) & (~padding_mask),
            "frame_mask": speech_condition_mask & (~padding_mask),
        }

    def _encode_condition(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.branch.encode_condition(
            speech_condition=batch["speech_condition"],
            mask=batch["mask"],
            cutoff_hz=batch["cutoff_hz"],
            prompt_mask=batch["prompt_mask"],
        )

    def forward(
        self,
        tokens: List[List[int]],
        features: Tensor,
        features_lens: Tensor,
        noise: Tensor,
        t: Tensor,
        step: int = 0,
        total_steps: int = 0,
        rollout_prob: float = 0.0,
        rollout_steps: int = 1,
        draft_num_step: int = 0,
        draft_k: int = 1,
        draft_guidance_scale: float = 1.0,
        draft_t_shift: float = 0.5,
    ) -> Dict[str, Tensor]:
        """The branch's training terms.

        Two modes. By default the branch is supervised at a single time ``t``:
        with probability ``rollout_prob`` per utterance at a detached state both
        trajectories reached from pure noise in ``rollout_steps`` Euler steps,
        otherwise at the teacher-forced state. With ``draft_num_step > 0`` the
        full coupled solve is run instead, on the inference grid, and the terms
        are taken on its output; see :meth:`_draft_terms`.

        Three terms are returned and the caller weights them:

        ``fm_loss``
            The masked MSE between the predicted and clean endpoints over the
            missing band. The name is retained for checkpoint and metric
            compatibility.
        ``recon_loss``
            An L1 error on the assembled endpoint over the missing band. This
            term is minimised by the per-bin conditional median, which in
            log-mel sits 9-10 dB below the energy of any real sample -- it is
            exactly the hedge the adapter arm measured, and weighting it heavily
            will reproduce that result. It is here because it is also the term
            that stabilises early training by saying plainly where the endpoint
            should land, and because an adversarial stage can then take the
            hedge back out. Weight it accordingly, and 0 is a supported setting.
        ``low_loss``
            The MSE of the branch's endpoint over the observed band, on
            generated frames. Only meaningful with ``steer_low_band``; zero
            and gradient-free without it, since the low band is then the
            backbone's.

        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, (B, T, F), already scaled.
            features_lens: the length of each feature sequence, (B,).
            noise: the initial noise, (B, T, F).
            t: the timestep, (B, 1, 1). Unused in full-solve mode.
            step, total_steps: position in training, for the cutoff curriculum.
            rollout_prob: per-utterance probability of replacing the
                teacher-forced state with a detached self-generated state.
            rollout_steps: number of Euler steps from pure noise to ``t``.
            draft_num_step: if positive, run the full solve with this many
                steps instead of supervising at ``t``.
            draft_k: the number of final steps gradients flow through.
            draft_guidance_scale, draft_t_shift: the backbone's guidance and the
                time grid's shift, which should match inference's.

        Returns:
            ``fm_loss``, ``recon_loss`` and ``low_loss``, and alongside them the
            tensors an adversarial stage needs and would otherwise have to
            recompute: ``x1_hat`` (the endpoint, assembled as inference
            assembles it), ``condition`` (the band-limited prompt),
            ``cutoff_hz``, ``band_mask``, ``frame_mask`` (True where the model
            had to generate), ``t``, and the realized ``rollout_fraction``.
        """
        if not 0.0 <= rollout_prob <= 1.0:
            raise ValueError(f"rollout_prob must be in [0, 1], got {rollout_prob}")
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
        if draft_num_step > 0 and not 1 <= draft_k <= draft_num_step:
            raise ValueError(
                f"draft_k must be in [1, {draft_num_step}], got {draft_k}"
            )

        with torch.no_grad():
            batch = self._condition_batch(
                tokens, features, features_lens, step, total_steps
            )

        if draft_num_step > 0:
            return self._draft_terms(
                batch=batch,
                features=features,
                noise=noise,
                num_step=draft_num_step,
                k=draft_k,
                guidance_scale=draft_guidance_scale,
                t_shift=draft_t_shift,
            )

        with torch.no_grad():
            teacher_xt = features * t + noise * (1 - t)

            rollout_mask = torch.rand(
                features.size(0), 1, 1, device=features.device
            ) < rollout_prob
            if rollout_mask.any():
                # Begin exactly where inference does. Ground-truth high-band
                # content is never inserted into a selected rollout input; it
                # appears only as the endpoint supervision below.
                dt = t / rollout_steps
                x_roll, y_roll = noise, noise
                rollout_condition = self._encode_condition(batch)
                for rollout_step in range(rollout_steps):
                    x_roll, y_roll = self._coupled_step(
                        t=dt * rollout_step,
                        dt=dt,
                        x=x_roll,
                        y=y_roll,
                        batch=batch,
                        condition=rollout_condition,
                    )

                backbone_xt = torch.where(rollout_mask, x_roll, teacher_xt)
                branch_xt = torch.where(rollout_mask, y_roll, teacher_xt)
            else:
                backbone_xt = teacher_xt
                branch_xt = teacher_xt

            # Recompute the frozen backbone at the actual target-time state.
            # Its hidden tensor now has the same state relationship to the
            # branch input that it has inside sample().
            vt, hidden = self.forward_fm_decoder(
                t=t,
                xt=backbone_xt,
                text_condition=batch["text_condition"],
                speech_condition=batch["speech_condition"],
                padding_mask=batch["padding_mask"],
                return_hidden=True,
            )
            # The backbone's own endpoint estimate. It is not part of any loss,
            # but it is the low band of the mel an adversarial stage has to
            # judge: showing a discriminator the ground-truth low band instead
            # would hand it a half-real sample to call fake.
            x1_backbone = backbone_xt + (1.0 - t) * vt

        branch_x1 = self.branch(
            t=t,
            xt=branch_xt,
            text_condition=batch["text_condition"],
            speech_condition=batch["speech_condition"],
            mask=batch["mask"],
            cutoff_hz=batch["cutoff_hz"],
            prompt_mask=batch["prompt_mask"],
            memory=hidden,
            padding_mask=batch["padding_mask"],
        )

        # Assemble the endpoint exactly as inference does: with steering the
        # branch owns every bin, otherwise the backbone keeps the low band.
        if self.steer_low_band:
            x1_hat = branch_x1
        else:
            x1_hat = self._project(x1_backbone, branch_x1, batch["mask"])

        return self._terms(
            batch=batch,
            features=features,
            branch_x1=branch_x1,
            x1_hat=x1_hat,
            t=t,
            rollout_fraction=rollout_mask.float().mean(),
        )

    def _draft_terms(
        self,
        batch: Dict[str, Tensor],
        features: Tensor,
        noise: Tensor,
        num_step: int,
        k: int,
        guidance_scale: float,
        t_shift: float,
    ) -> Dict[str, Tensor]:
        """The training terms on the output of a full coupled solve.

        The solve starts at pure noise and runs on the grid inference uses, so
        every state the branch sees is one it would see at inference, including
        whatever its own steering did to the steps after it. Only the last
        ``k`` steps are differentiated: the first ``num_step - k`` run without
        gradients and their end state is treated as given, which is DRaFT-K
        (Clark et al., 2023). The backbone is frozen and never reads the branch,
        so its half of every step is gradient-free regardless.

        The final Euler step lands exactly on the branch's last endpoint
        estimate wherever the branch owns the bin, so the terms here are that
        estimate's error, with gradients through the steps that led to its
        input.
        """
        grad_enabled = torch.is_grad_enabled()
        timesteps = get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=num_step,
            t_shift=t_shift,
            device=features.device,
        )
        condition = self._encode_condition(batch)

        x, y = noise, noise
        for i in range(num_step):
            with torch.set_grad_enabled(grad_enabled and i >= num_step - k):
                x, y = self._coupled_step(
                    t=timesteps[i],
                    dt=timesteps[i + 1] - timesteps[i],
                    x=x,
                    y=y,
                    batch=batch,
                    condition=condition,
                    guidance_scale=guidance_scale,
                )

        ones = torch.ones(features.size(0), 1, 1, device=features.device)
        return self._terms(
            batch=batch,
            features=features,
            branch_x1=y,
            x1_hat=y,
            t=ones,
            rollout_fraction=ones.mean(),
        )

    def _terms(
        self,
        batch: Dict[str, Tensor],
        features: Tensor,
        branch_x1: Tensor,
        x1_hat: Tensor,
        t: Tensor,
        rollout_fraction: Tensor,
    ) -> Dict[str, Tensor]:
        """The losses and the adversarial stage's context, from the estimates.

        Each bin is weighted by how missing it is: 1.0 above the cutoff, 0.0
        below it, and the ramp in between in proportion. The low-band term uses
        the complement. Frames inside the prompt or the padding count for
        nothing in either.
        """
        mask = batch["mask"]
        frame_weight = batch["frame_mask"].to(features.dtype).unsqueeze(-1)

        weight = mask * frame_weight
        total = weight.sum().clamp(min=1.0)
        fm_loss = (weight * (branch_x1 - features) ** 2).sum() / total
        recon_loss = (weight * (x1_hat - features).abs()).sum() / total

        if self.steer_low_band:
            low_weight = (1.0 - mask) * frame_weight
            low_total = low_weight.sum().clamp(min=1.0)
            low_loss = (low_weight * (branch_x1 - features) ** 2).sum() / low_total
        else:
            low_loss = torch.zeros((), device=features.device)

        return {
            "fm_loss": fm_loss,
            "recon_loss": recon_loss,
            "low_loss": low_loss,
            "x1_hat": x1_hat,
            "condition": batch["speech_condition"],
            "cutoff_hz": batch["cutoff_hz"],
            "band_mask": mask,
            "frame_mask": batch["frame_mask"],
            "t": t,
            "rollout_fraction": rollout_fraction,
        }

