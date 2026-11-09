p = "zipvoice/models/zipvoice_adapter.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)


rep("""        clean_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor]]:""",
"""        clean_condition: Optional[torch.Tensor] = None,
        velocity_low_steps: int = 0,
    ) -> Tuple[
        Optional[torch.Tensor], int, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:""")

rep("""        Returns the sample, zero-padded back to the batch's frame count, the
        index of the utterance it belongs to, and the reference or None;
        ``(None, -1, None)`` when no utterance in the batch was band-limited.
        \"\"\"""",
"""        ``velocity_low_steps`` charges a running cost along the solve, after
        VGG-Flow (arXiv 2512.05116): on that many solver steps, drawn uniformly,
        the squared gap between the adapted velocity and the stock model's
        velocity at the *same* state and with the *same* degraded prompt, over
        the observed bins of the generated span, weighted by the step's width.
        The terminal comparison against the clean-prompt solve cannot see the
        low-band drift that cost SIM-o: its floor is the gap between a clean and
        a degraded prompt, about 0.012 whatever the branch does. This one is zero
        when the branch and LoRA leave the observed band alone, and it
        constrains every step of the trajectory, not the endpoint. It is a loss,
        not a projection: nothing is removed from the velocity. A drawn step
        inside the ungraphed prefix is run with a local graph -- its state stays
        a constant, and the step's velocity is detached before it moves the
        state -- so each costs one guided forward's activations. The stock
        velocity uses the branch-off rule the reference uses (zero band mask),
        under ``no_grad``.

        Returns the sample, zero-padded back to the batch's frame count, the
        index of the utterance it belongs to, the reference or None, and the
        running cost or None; ``(None, -1, None, None)`` when no utterance in
        the batch was band-limited.
        \"\"\"""")

rep("""        if clean_condition is not None and self._released_decoder_trainable():
            raise ValueError(
                "The rollout reference is the frozen model's solve, and the "
                "backbone is being trained"
            )""",
"""        if (
            clean_condition is not None or velocity_low_steps > 0
        ) and self._released_decoder_trainable():
            raise ValueError(
                "The rollout reference is the frozen model's solve, and the "
                "backbone is being trained"
            )
        if not 0 <= velocity_low_steps <= num_step:
            raise ValueError(
                f"velocity_low_steps {velocity_low_steps} outside [0, {num_step}]"
            )""")

rep("""        if not bool(eligible.any()):
            return None, -1, None
""", """        if not bool(eligible.any()):
            return None, -1, None, None
""")

rep("""        self._prev_residual = None
        reference = None
        try:""",
"""        self._prev_residual = None
        reference = None
        charged = set(torch.randperm(num_step)[:velocity_low_steps].tolist())
        stock = dict(conditioning, band_mask=torch.zeros_like(band_mask[one]))
        # Observed bins of the generated span: the prompt frames are discarded
        # at inference, and the missing bins are the branch's to write.
        observed = (
            (~prompt_mask[one, :frames] & ~padding_mask[one, :frames])
            .unsqueeze(-1)
            .to(x.dtype)
            * (1.0 - band_mask[one].to(x.dtype))
        )
        observed_count = observed.sum().clamp(min=1.0)
        running = []

        def charge(step: int, state: torch.Tensor, velocity: torch.Tensor) -> None:
            # The stock call would overwrite the branch's carried residual
            # with its own zero one.
            carried = self._prev_residual
            with torch.no_grad():
                stock_velocity = self.solver.model(t=grid[step], x=state, **stock)
            self._prev_residual = carried
            gap = ((velocity - stock_velocity).pow(2) * observed).sum()
            running.append(gap / observed_count * (grid[step + 1] - grid[step]))

        try:""")

rep("""            first_graphed = num_step - grad_steps
            with torch.no_grad():
                for step in range(first_graphed):
                    velocity = self.solver.model(t=grid[step], x=x, **conditioning)
                    x = x + velocity * (grid[step + 1] - grid[step])
            self._detach_backbone = grad_steps == 1
            for step in range(first_graphed, num_step):
                velocity = self.solver.model(t=grid[step], x=x, **conditioning)
                x = x + velocity * (grid[step + 1] - grid[step])
            x1 = x""",
"""            first_graphed = num_step - grad_steps
            for step in range(first_graphed):
                if step in charged:
                    # A local graph. The LoRA factors live in the backbone, so
                    # it runs with one here.
                    self._detach_backbone = False
                    velocity = self.solver.model(t=grid[step], x=x, **conditioning)
                    self._detach_backbone = True
                    charge(step, x, velocity)
                    velocity = velocity.detach()
                    if self._prev_residual is not None:
                        self._prev_residual = self._prev_residual.detach()
                else:
                    with torch.no_grad():
                        velocity = self.solver.model(
                            t=grid[step], x=x, **conditioning
                        )
                x = x + velocity * (grid[step + 1] - grid[step])
            self._detach_backbone = grad_steps == 1
            for step in range(first_graphed, num_step):
                velocity = self.solver.model(t=grid[step], x=x, **conditioning)
                if step in charged:
                    charge(step, x, velocity)
                x = x + velocity * (grid[step + 1] - grid[step])
            x1 = x""")

rep("""        if frames == noise.size(1):
            return x1, index, reference
""", """        # Normalised by the width the drawn steps cover, so the scale does not
        # depend on how many were drawn.
        velocity_low = (
            torch.stack(running).sum()
            / sum(float(grid[s + 1] - grid[s]) for s in charged)
            if running
            else None
        )
        if frames == noise.size(1):
            return x1, index, reference, velocity_low
""")
rep("""        return torch.cat([x1, pad], dim=1), index, reference

    def multistep_preservation(""",
"""        return torch.cat([x1, pad], dim=1), index, reference, velocity_low

    def multistep_preservation(""")

rep("""        identity_rollout_underfill: bool = False,
        **unused,""", """        identity_rollout_underfill: bool = False,
        identity_rollout_velocity_steps: int = 0,
        **unused,""")
rep("""            sample, index, reference = self.identity_rollout(""",
    """            sample, index, reference, velocity_low = self.identity_rollout(""")
rep("""                clean_condition=clean_condition,
            )
            # ``None`` when""", """                clean_condition=clean_condition,
                velocity_low_steps=identity_rollout_velocity_steps,
            )
            # ``None`` when""")
rep("""                    "rollout_band_bias": rollout_band_bias,
                    "multistep_loss": zero,""",
"""                    "rollout_band_bias": rollout_band_bias,
                    "rollout_velocity_low_loss": (
                        zero if velocity_low is None else velocity_low
                    ),
                    "multistep_loss": zero,""")
open(p, "w", encoding="utf-8").write(s)
print("ok")
