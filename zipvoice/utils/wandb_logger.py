# Copyright    2026
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mirror the training scripts' TensorBoard metrics to Weights & Biases."""

import logging
from typing import Any, Optional

# Weights & Biases refuses to log at a step below the run's own internal
# counter, which is exactly what happens when a run resumes from a checkpoint
# older than the point it previously reached. Logging the training step as an
# ordinary metric and declaring it the x-axis sidesteps that: the internal
# counter stays monotone while the plotted step follows the checkpoint.
STEP_METRIC = "trainer/global_step"


class FanoutWriter:
    """
    Forwards ``add_scalar`` to a TensorBoard writer and a Weights & Biases run.

    Every metric in the training scripts reaches TensorBoard through
    ``add_scalar``, either directly or via ``MetricsTracker.write_summary``, so
    mirroring that single method sends the same metrics to both backends and
    leaves all the call sites untouched. Either backend may be ``None``.
    """

    def __init__(self, tb_writer: Any = None, wandb_run: Any = None) -> None:
        self.tb_writer = tb_writer
        self.wandb_run = wandb_run

    def add_scalar(self, tag: str, value: Any, global_step: Optional[int] = None):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, global_step)
        if self.wandb_run is not None:
            payload = {tag: value}
            if global_step is not None:
                payload[STEP_METRIC] = global_step
            self.wandb_run.log(payload)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def init_wandb(params: Any, run_id: Optional[str] = None) -> Optional[Any]:
    """
    Start a Weights & Biases run for ``params``, or return ``None`` if disabled.

    Authentication comes from the ``WANDB_API_KEY`` environment variable (or a
    prior ``wandb login``); no key is read from or written to the recipe.

    ``run_id`` reattaches to an existing run. Training checkpoints carry the id
    of the run that wrote them, so resuming a checkpoint continues that run's
    history instead of opening a second one beside it.
    """
    if not params.wandb:
        return None

    import wandb

    # AttributeDict holds tensors, devices and tokenizers alongside the plain
    # hyperparameters. Only the primitives are worth recording, and the rest
    # would not serialize.
    config = {
        k: v
        for k, v in params.items()
        if isinstance(v, (bool, int, float, str, type(None)))
    }
    default_name = (
        str(params.exp_dir).replace("\\", "/").rstrip("/").split("/")[-1]
    )
    run = wandb.init(
        project=params.wandb_project,
        id=run_id,
        name=params.wandb_name or default_name,
        dir=str(params.exp_dir),
        config=config,
        resume="allow",
    )
    run.define_metric(STEP_METRIC)
    run.define_metric("*", step_metric=STEP_METRIC)
    logging.info(f"Logging to Weights & Biases run: {run.url}")
    return run
