"""Checks for the DMD student (zipvoice/models/modules/dmd.py) on a tiny model.

Run: python -m unittest tests.test_dmd
"""

import unittest

import torch

from zipvoice.models.modules.dmd import (
    FakeScoreLoRA,
    backbone_velocity,
    build_dmd_batch,
    dmd_generator_loss,
    fake_score_loss,
    student_rollout,
)
from zipvoice.models.zipvoice_adapter import ZipVoiceAdapter


def tiny_model(lora_rank: int = 4) -> ZipVoiceAdapter:
    torch.manual_seed(0)
    model = ZipVoiceAdapter(
        fm_decoder_downsampling_factor=[1, 2, 1],
        fm_decoder_num_layers=[1, 1, 1],
        fm_decoder_cnn_module_kernel=[7, 7, 7],
        fm_decoder_feedforward_dim=64,
        fm_decoder_num_heads=2,
        fm_decoder_dim=32,
        text_encoder_num_layers=1,
        text_encoder_feedforward_dim=32,
        text_encoder_cnn_module_kernel=3,
        text_encoder_num_heads=2,
        text_encoder_dim=16,
        query_head_dim=8,
        value_head_dim=4,
        pos_head_dim=4,
        pos_dim=8,
        time_embed_dim=16,
        text_embed_dim=16,
        feat_dim=100,
        vocab_size=20,
        pad_id=0,
        freq_dim=8,
        freq_num_layers=1,
        freq_num_heads=2,
        freq_bin_out=2,
        adapter_dim=16,
        adapter_layers=1,
        adapter_heads=2,
        adapter_kernel=3,
        lora_rank=lora_rank,
    )
    model.set_trainable("lora")
    # A freshly built adapter writes almost nothing; make its residual and the
    # student LoRA large enough that a missing gradient path would show.
    with torch.no_grad():
        for module in model._lora_modules:
            module.lora_B.normal_(std=0.05)
        for param in model.adapter.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    return model


def tiny_batch(model, num_utts=2, cutoffs=(4000.0,)):
    torch.manual_seed(1)
    lens = torch.tensor([40, 32, 36])
    features = torch.randn(3, 40, 100) * 0.5
    tokens = [[1, 2, 3, 4, 5], [3, 4, 5], [6, 7, 8, 9]]
    return build_dmd_batch(
        model,
        tokens=tokens,
        features=features,
        features_lens=lens,
        cutoffs=cutoffs,
        num_utts=num_utts,
    )


def released_grads(model):
    """Released tensors (everything not trained in 'lora' mode) with a grad."""
    return [
        name
        for name, p in model.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]


class DMDTests(unittest.TestCase):
    def test_fake_starts_at_teacher(self):
        model = tiny_model().eval()
        fake = FakeScoreLoRA(model, rank=4)
        batch = tiny_batch(model)
        x = torch.randn_like(batch.features)
        t = torch.tensor(0.6)
        args = dict(
            t=t,
            x=x,
            text_condition=batch.text_condition,
            speech_condition=batch.clean_condition,
            padding_mask=batch.padding_mask,
            guidance_scale=1.0,
        )
        with torch.no_grad():
            real = backbone_velocity(model, **args)
            with fake.enabled():
                same = backbone_velocity(model, **args)
        self.assertTrue(torch.equal(real, same))

    def test_teacher_is_released_model(self):
        # The adapter with no band mask is the released checkpoint: the branch
        # residual and the student LoRA are both gated to zero.
        model = tiny_model().eval()
        batch = tiny_batch(model)
        x = torch.randn_like(batch.features)
        t = torch.tensor(0.3)
        with torch.no_grad():
            teacher = backbone_velocity(
                model,
                t=t,
                x=x,
                text_condition=batch.text_condition,
                speech_condition=batch.clean_condition,
                padding_mask=batch.padding_mask,
            )
            full_band = model.forward_fm_decoder(
                t=t,
                xt=x,
                text_condition=batch.text_condition,
                speech_condition=batch.clean_condition,
                padding_mask=batch.padding_mask,
            )
        torch.testing.assert_close(teacher, full_band, rtol=0, atol=0)

    def test_rollout_is_the_deployed_solve(self):
        model = tiny_model().eval()
        batch = tiny_batch(model)
        noise = torch.randn_like(batch.features)
        cond = batch.student_conditioning()
        with torch.no_grad():
            ours = student_rollout(
                model, noise, cond, num_step=4, guidance_scale=1.0, t_shift=0.5
            )
            model._carry_residual = True
            model._prev_residual = None
            try:
                deployed = model.solver.sample(
                    x=noise,
                    num_step=4,
                    guidance_scale=1.0,
                    t_shift=0.5,
                    **cond,
                )
            finally:
                model._carry_residual = False
                model._prev_residual = None
        torch.testing.assert_close(ours, deployed, rtol=0, atol=0)

    def test_generator_loss_is_zero_until_fake_moves(self):
        model = tiny_model()
        fake = FakeScoreLoRA(model, rank=4)
        batch = tiny_batch(model)
        sample = student_rollout(
            model,
            torch.randn_like(batch.features),
            batch.student_conditioning(),
            num_step=3,
        )
        # Unguided teacher: the fake score is the teacher at step 0, so there is
        # no signal until it has been fitted to the student.
        loss, stats = dmd_generator_loss(model, fake, sample, batch, real_guidance=0.0)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["fake_real_gap"], 0.0)
        # Guided teacher: the gap at step 0 is the guidance direction itself,
        # which is why the trainer runs fake-only updates before this term.
        loss, stats = dmd_generator_loss(model, fake, sample, batch, real_guidance=1.0)
        self.assertGreater(stats["fake_real_gap"], 0.0)

    def test_generator_gradient_reaches_student_only(self):
        model = tiny_model()
        fake = FakeScoreLoRA(model, rank=4)
        with torch.no_grad():
            for param in fake.lora_B.values():
                param.normal_(std=0.05)
        batch = tiny_batch(model)
        sample = student_rollout(
            model,
            torch.randn_like(batch.features),
            batch.student_conditioning(),
            num_step=3,
            grad_steps=2,
        )
        loss, stats = dmd_generator_loss(model, fake, sample, batch)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        lora_grads = [m.lora_B.grad for m in model._lora_modules]
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in lora_grads))
        adapter_grads = [p.grad for p in model.adapter.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in adapter_grads))
        self.assertEqual(released_grads(model), [])
        self.assertTrue(all(p.grad is None for p in fake.parameters()))

    def test_fake_loss_trains_fake_only(self):
        model = tiny_model()
        fake = FakeScoreLoRA(model, rank=4)
        batch = tiny_batch(model)
        with torch.no_grad():
            sample = student_rollout(
                model,
                torch.randn_like(batch.features),
                batch.student_conditioning(),
                num_step=3,
            )
        loss = fake_score_loss(model, fake, sample, batch)
        loss.backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in fake.parameters())
        )
        self.assertTrue(
            all(p.grad is None for p in model.parameters()),
            "the fake score's loss must not reach the student or the backbone",
        )

    def test_fake_hooks_are_inert_outside_enabled(self):
        model = tiny_model().eval()
        batch = tiny_batch(model)
        x = torch.randn_like(batch.features)
        args = dict(
            t=torch.tensor(0.5),
            x=x,
            text_condition=batch.text_condition,
            speech_condition=batch.clean_condition,
            padding_mask=batch.padding_mask,
        )
        with torch.no_grad():
            before = backbone_velocity(model, **args)
            fake = FakeScoreLoRA(model, rank=4)
            for param in fake.lora_B.values():
                param.normal_(std=0.05)
            after = backbone_velocity(model, **args)
            with fake.enabled():
                moved = backbone_velocity(model, **args)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.equal(before, moved))
        self.assertFalse(any("lora_A" in k and "fake" in k for k in model.state_dict()))
        self.assertEqual(
            len(model.state_dict()), len(tiny_model().state_dict()),
            "the fake score must not enter the model's checkpoint",
        )

    def test_batch_masks(self):
        model = tiny_model()
        batch = tiny_batch(model, num_utts=2, cutoffs=(4000.0,))
        self.assertEqual(batch.features.size(0), 2)
        valid = ~batch.padding_mask
        # Prompt is a non-empty prefix; generated span is the rest.
        self.assertTrue(bool((batch.prompt_mask.sum(1) > 0).all()))
        self.assertTrue(torch.equal(batch.prompt_mask | batch.gen_mask, valid))
        self.assertFalse(bool((batch.prompt_mask & batch.gen_mask).any()))
        first_gen = batch.gen_mask.float().argmax(1)
        self.assertTrue(torch.equal(first_gen, batch.prompt_mask.sum(1)))
        # Conditions are zero off the prompt; the student's is band-limited.
        off = ~batch.prompt_mask
        self.assertEqual(float(batch.clean_condition[off].abs().sum()), 0.0)
        self.assertEqual(float(batch.student_condition[off].abs().sum()), 0.0)
        self.assertGreater(float(batch.band_mask.sum()), 0.0)
        self.assertFalse(torch.equal(batch.clean_condition, batch.student_condition))


class ControllerTests(unittest.TestCase):
    def test_training_loop_moves_only_student_and_fake(self):
        from zipvoice.models.modules.dmd import DMDController
        from zipvoice.utils.common import get_parameter_groups_with_lrs
        from zipvoice.utils.feature import VocosFbank
        from zipvoice.utils.optim import ScaledAdam

        model = tiny_model()
        model.train()
        released = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if not p.requires_grad
        }
        student = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        optimizer = ScaledAdam(
            get_parameter_groups_with_lrs(model, lr=1e-3, include_names=True),
            lr=1e-3,
            clipping_scale=2.0,
        )
        dmd = DMDController(
            model, rank=4, lr=1e-3, start_step=1, gen_interval=2, num_step=3,
            grad_steps=2, cutoffs=(4000.0, 8000.0), num_utts=2,
        )
        fake_before = {k: v.detach().clone() for k, v in dmd.fake.state_dict().items()}

        torch.manual_seed(2)
        lens = torch.tensor([40, 32, 36])
        features = torch.randn(3, 40, 100) * 0.5
        audio = torch.randn(3, 40 * 256) * 0.1
        audio_lens = lens * 256
        tokens = [[1, 2, 3, 4, 5], [3, 4, 5], [6, 7, 8, 9]]
        fbank = VocosFbank().fbank
        data = dict(
            tokens=tokens, features=features, features_lens=lens,
            audio=audio, audio_lens=audio_lens, fbank=fbank,
        )
        generator_steps = 0
        for step in range(4):
            loss, stats = dmd.generator_loss(model, step=step, **data)
            if loss is not None:
                generator_steps += 1
                self.assertIn("grad_rms", stats)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            info = dmd.fake_update(model, **data)
            self.assertGreater(info["fake_frames"], 0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(generator_steps, 2)  # steps 1 and 3

        for name, value in released.items():
            self.assertTrue(
                torch.equal(value, dict(model.named_parameters())[name]), name
            )
        moved = [
            name for name, p in model.named_parameters()
            if name in student and not torch.equal(p, student[name])
        ]
        self.assertTrue(any("lora_" in name for name in moved), moved)
        self.assertTrue(any(name.startswith("adapter.") for name in moved), moved)
        self.assertTrue(
            any(not torch.equal(v, fake_before[k]) for k, v in dmd.fake.state_dict().items())
        )

        # The checkpoint is an ordinary adapter: loads strictly into a fresh one.
        tiny_model().load_state_dict(model.state_dict(), strict=True)
        # And the fake score's state round-trips.
        other = DMDController(model, rank=4)
        other.load_state_dict(dmd.state_dict())
        for k, v in dmd.fake.state_dict().items():
            self.assertTrue(torch.equal(v, other.fake.state_dict()[k]))


if __name__ == "__main__":
    unittest.main()
