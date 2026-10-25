from types import SimpleNamespace

import pytest
import torch

import resfusion_flux2 as flux2


@pytest.fixture
def fake_components(monkeypatch, tmp_path):
    events = []
    loaded = {}

    class Generator(torch.nn.Module):
        def __init__(self, args, ckpt_file=None):
            super().__init__()
            self.noise_scheduler = object()
            self.device = "cpu"
            self.dtype = None
            loaded["checkpoint"] = ckpt_file

        def to(self, device, dtype=None):
            self.device = str(device)
            self.dtype = dtype or self.dtype
            events.append(("generator", self.device))
            return self

        def resfusion_inference_loop(self, **kwargs):
            assert self.device == "cuda"
            assert kwargs["prompt_embeds"].dtype == self.dtype
            assert kwargs["txt_ids"].dtype == self.dtype
            events.append(("restore", self.device))
            return "restored"

    class Pipeline:
        def __init__(self):
            self.text_encoder = SimpleNamespace(eval=lambda: None)
            self.device = "cpu"
            self.fail_encode = False
            self.fail_transfer = False

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            loaded["model_id"] = model_id
            loaded["pipeline_kwargs"] = kwargs
            return cls()

        def to(self, device, dtype=None):
            self.device = str(device)
            events.append(("text", self.device))
            if self.fail_transfer and self.device == "cuda":
                raise RuntimeError("text transfer failed")
            return self

        def encode_prompt(self, prompt, device):
            assert self.device == str(device) == "cuda"
            events.append(("encode", prompt))
            if self.fail_encode:
                raise RuntimeError("text encoding failed")
            return torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)

    class LQ:
        def to(self, device, dtype):
            assert str(device) == "cuda"
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux2, "Resfusion_flux2_gen", Generator)
    monkeypatch.setattr(flux2, "Flux2KleinPipeline", Pipeline)
    monkeypatch.setattr(flux2, "Accelerator", lambda **kwargs: SimpleNamespace(**kwargs))
    checkpoint = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"fixture")
    args = SimpleNamespace(
        pretrained_model_name_or_path="example/flux2",
        resfusion_path=str(checkpoint),
        mixed_precision="fp16",
    )
    return args, events, loaded, LQ()


def test_default_loading_remains_on_cuda_and_omits_unused_components(fake_components):
    args, events, loaded, lq = fake_components
    model = flux2.Resfusion_flux2_test(args)
    assert not model.cpu_offload
    assert events == [("generator", "cuda"), ("text", "cuda")]
    assert loaded["pipeline_kwargs"] == {
        "transformer": None,
        "vae": None,
        "scheduler": None,
        "torch_dtype": torch.float16,
    }
    assert loaded["checkpoint"] == args.resfusion_path
    events.clear()
    assert model(lq, prompt="building") == "restored"
    assert events == [("encode", "building"), ("restore", "cuda")]


@pytest.mark.parametrize("precision,dtype", [("fp16", torch.float16), ("bf16", torch.bfloat16)])
def test_offload_swaps_components_before_each_image(fake_components, precision, dtype):
    args, events, _, lq = fake_components
    args.cpu_offload = True
    args.mixed_precision = precision
    model = flux2.Resfusion_flux2_test(args)
    assert events == [("generator", "cpu"), ("text", "cpu")]
    events.clear()
    for prompt in ("tree", "window"):
        assert model(lq, prompt=prompt) == "restored"
    assert events == [
        ("generator", "cpu"), ("text", "cuda"), ("encode", "tree"),
        ("text", "cpu"), ("generator", "cuda"), ("restore", "cuda"),
        ("generator", "cpu"), ("text", "cuda"), ("encode", "window"),
        ("text", "cpu"), ("generator", "cuda"), ("restore", "cuda"),
    ]
    assert model.model.dtype == dtype
    assert model.pipe.device == "cpu"


@pytest.mark.parametrize("failure", ["fail_encode", "fail_transfer"])
def test_offload_returns_text_to_cpu_when_encoding_fails(fake_components, failure):
    args, events, _, lq = fake_components
    args.cpu_offload = True
    model = flux2.Resfusion_flux2_test(args)
    setattr(model.pipe, failure, True)
    events.clear()
    with pytest.raises(RuntimeError):
        model(lq, prompt="tree")
    assert model.pipe.device == "cpu"
    assert model.model.device == "cpu"
    assert events[-1] == ("text", "cpu")
    assert not any(event[0] == "restore" for event in events)


def test_missing_checkpoint_does_not_construct_a_random_adapter(fake_components):
    args, _, loaded, _ = fake_components
    args.resfusion_path = ""
    with pytest.raises(FileNotFoundError, match="--resfusion_path"):
        flux2.Resfusion_flux2_test(args)
    assert "checkpoint" not in loaded


def test_cuda_is_required(fake_components, monkeypatch):
    args, _, loaded, _ = fake_components
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EnvironmentError, match="requires CUDA"):
        flux2.Resfusion_flux2_test(args)
    assert "checkpoint" not in loaded
