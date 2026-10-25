import pytest
import torch

from scheduler import ResfusionFlowMatchScheduler


@pytest.mark.parametrize("nominal_steps,actual_steps", [(6, 1), (10, 2), (20, 4), (42, 10)])
def test_documented_sampling_schedule(nominal_steps, actual_steps):
    scheduler = ResfusionFlowMatchScheduler(num_train_timesteps=1000, shift=3.0, rsr=1.0)
    scheduler.set_timesteps(num_inference_steps=nominal_steps, device="cpu")
    assert len(scheduler.timesteps) == actual_steps
    assert len(scheduler.sigmas) == actual_steps + 1
    assert scheduler.timesteps[0].item() == 500.0
    assert scheduler.sigmas[0].item() == 0.5
    assert scheduler.sigmas[-1].item() == 0.0
    assert torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:])
    assert torch.isfinite(scheduler.timesteps).all()
    assert torch.isfinite(scheduler.sigmas).all()
