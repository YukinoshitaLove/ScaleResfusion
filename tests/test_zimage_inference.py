"""Small Z-Image inference regressions using mock and tiny models."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scratch_dir(tmp_path):
    return tmp_path


@pytest.fixture
def checkpoint_path(scratch_dir):
    # Only the file-path contract is tested here; mocked loaders never read it.
    path = scratch_dir / "gen_hq_transformer_lora.safetensors"
    path.write_bytes(b"mock loader sentinel, not real weights")
    return path


@pytest.fixture
def cli():
    # Execute the actual lightweight entry point without importing another backbone.
    path = ROOT / "test_resfusion_zimage.py"
    module = ModuleType("_zimage_cli_under_test")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _cli_args(checkpoint_path):
    return ["-i", "inputs", "-o", "outputs", "--resfusion_path", str(checkpoint_path)]


def test_cli_defaults_and_boolean_prompt_switch(cli, checkpoint_path):
    args = cli.parse_args(_cli_args(checkpoint_path))
    assert args.pretrained_model_name_or_path == "Tongyi-MAI/Z-Image"
    assert args.mixed_precision == "bf16"
    assert (args.seed, args.flowT, args.rsr, args.upscale, args.process_size) == (42, 20, 1.0, 4, 512)
    assert args.align_method == "adain"
    assert args.save_prompts is True
    assert args.prompt is None
    args = cli.parse_args(_cli_args(checkpoint_path) + ["--no-save_prompts", "--prompt", "a photo"])
    assert args.save_prompts is False
    assert args.prompt == "a photo"
    assert cli.parse_args(_cli_args(checkpoint_path) + ["--save_prompts"]).save_prompts is True


@pytest.mark.parametrize("missing", ["-i", "-o", "--resfusion_path"])
def test_cli_requires_io_and_checkpoint(cli, checkpoint_path, missing):
    argv = _cli_args(checkpoint_path)
    index = argv.index(missing)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        cli.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_cli_rejects_non_file_checkpoint(cli, scratch_dir, kind):
    path = scratch_dir if kind == "directory" else scratch_dir / "missing.safetensors"
    with pytest.raises(SystemExit) as error:
        cli.parse_args(_cli_args(path))
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", [
    ("--flowT", "0"), ("--upscale", "-1"), ("--process_size", "0"),
    ("--rsr", "0"), ("--rsr", "nan"), ("--rsr", "inf"),
    ("--mixed_precision", "float16"),
])
def test_cli_rejects_invalid_sampling_values(cli, checkpoint_path, flag, value):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(_cli_args(checkpoint_path) + [flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp32"])
def test_cli_accepts_precision_choices(cli, checkpoint_path, precision):
    args = cli.parse_args(_cli_args(checkpoint_path) + ["--mixed_precision", precision])
    assert args.mixed_precision == precision


def test_cli_has_no_eager_model_imports():
    tree = ast.parse((ROOT / "test_resfusion_zimage.py").read_text(encoding="utf-8"))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert set(imports) <= {"argparse", "math", "os"}


@pytest.fixture
def runtime():
    # Intentionally not importorskip: missing server dependencies must fail visibly.
    import torch
    import resfusion_zimage as inference
    import ReferenceNet_Modules.zimage_referencenet as reference
    return SimpleNamespace(torch=torch, inference=inference, reference=reference)


def test_loader_requires_checkpoint_before_loading_base(runtime, monkeypatch, scratch_dir):
    calls = []
    monkeypatch.setattr(runtime.inference.ZImageReferenceTransformer2DModel, "from_pretrained",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    for path in (None, "", scratch_dir, scratch_dir / "missing.safetensors"):
        with pytest.raises(FileNotFoundError, match="trained ScaleResfusion Z-Image LoRA checkpoint"):
            runtime.inference.initialize_transformer("base", path)
    assert calls == []


def test_loader_uses_original_adapter_api_and_propagates_failure(runtime, monkeypatch, checkpoint_path):
    calls = []

    class Transformer:
        def requires_grad_(self, enabled):
            calls.append(("requires_grad", enabled))

        def load_lora_adapter(self, path, *, weight_name, adapter_name):
            calls.append(("load_lora_adapter", path, weight_name, adapter_name))

    transformer = Transformer()  # Deliberately has no add_adapter fallback.

    def load_base(*args, **kwargs):
        calls.append(("from_pretrained", args, kwargs))
        return transformer

    monkeypatch.setattr(runtime.inference.ZImageReferenceTransformer2DModel, "from_pretrained", load_base)
    assert runtime.inference.initialize_transformer("base", str(checkpoint_path)) is transformer
    assert calls == [
        ("from_pretrained", ("base",), {"subfolder": "transformer"}),
        ("requires_grad", False),
        ("load_lora_adapter", str(checkpoint_path), checkpoint_path.name, "default_transformer"),
    ]

    def fail_load(*args, **kwargs):
        raise RuntimeError("invalid adapter weights")

    transformer.load_lora_adapter = fail_load
    with pytest.raises(RuntimeError, match="invalid adapter weights"):
        runtime.inference.initialize_transformer("base", str(checkpoint_path))


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
def test_wrapper_loads_text_only_pipeline(runtime, monkeypatch, checkpoint_path, precision):
    torch, inference = runtime.torch, runtime.inference
    pipeline_calls = []
    accelerator_calls = []

    class Generator(torch.nn.Module):
        def __init__(self, args, ckpt_file):
            super().__init__()
            assert ckpt_file == str(checkpoint_path)

        def to(self, *args, **kwargs):
            return self  # Do not initialise CUDA for this small mocked test.

    text_encoder = torch.nn.Linear(2, 2)
    pipe = SimpleNamespace(text_encoder=text_encoder, to=lambda *args, **kwargs: None)

    def load_pipeline(*args, **kwargs):
        pipeline_calls.append((args, kwargs))
        return pipe

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference, "Resfusion_zimage_gen", Generator)
    monkeypatch.setattr(inference.ZImagePipeline, "from_pretrained", load_pipeline)
    monkeypatch.setattr(inference, "Accelerator", lambda **kwargs: accelerator_calls.append(kwargs))
    args = SimpleNamespace(resfusion_path=str(checkpoint_path), pretrained_model_name_or_path="base",
                           mixed_precision=precision)
    warning = pytest.warns(RuntimeWarning, match="not guaranteed") if precision == "fp16" else nullcontext()
    with warning:
        wrapper = inference.Resfusion_zimage_test(args)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    assert wrapper.weight_dtype == dtype
    assert pipeline_calls == [(("base",), {"transformer": None, "vae": None, "torch_dtype": dtype})]
    assert accelerator_calls == [{"mixed_precision": "no" if precision == "fp32" else precision}]
    assert text_encoder.training is False
    assert all(not parameter.requires_grad for parameter in text_encoder.parameters())


@pytest.mark.parametrize("container", ["list", "tuple", "tensor"])
def test_encode_prompt_preserves_list_or_tensor_dtype_handling(runtime, container):
    torch, inference = runtime.torch, runtime.inference
    calls = []
    raw = torch.ones(1, 3, 6)
    if container == "list":
        raw = [raw[0]]
    elif container == "tuple":
        raw = (raw[0],)

    def encode(**kwargs):
        calls.append(kwargs)
        return raw, None

    wrapper = inference.Resfusion_zimage_test.__new__(inference.Resfusion_zimage_test)
    torch.nn.Module.__init__(wrapper)
    wrapper.device = torch.device("cpu")
    wrapper.weight_dtype = torch.bfloat16
    wrapper.pipe = SimpleNamespace(encode_prompt=encode)
    result = wrapper.encode_prompt("a photo")
    assert calls == [{"prompt": ["a photo"], "device": torch.device("cpu"),
                      "do_classifier_free_guidance": False}]
    if container == "tensor":
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bfloat16
    else:
        assert isinstance(result, list)
        assert all(item.dtype == torch.bfloat16 for item in result)


def test_sampling_keeps_two_calls_masks_timestep_sign_and_vae_path(runtime, monkeypatch):
    torch, inference = runtime.torch, runtime.inference
    calls, samples, decoded = [], [], []
    block_k, block_v, block_masks = (object(),), (object(),), (object(),)
    prompt_embeds = [torch.zeros(3, 6)]

    class VAE(torch.nn.Module):
        config = SimpleNamespace(shift_factor=0.25, scaling_factor=2.0)

        def encode(self, image):
            def sample():
                samples.append(True)
                return torch.full_like(image, 0.6)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=sample))

        def decode(self, latent):
            decoded.append(latent.clone())
            return SimpleNamespace(sample=latent)

    class Transformer(torch.nn.Module):
        def forward(self, **kwargs):
            calls.append(kwargs)
            if kwargs["joint_attention_kwargs"]["reference_mode"] == "write":
                return block_k, block_v, block_masks
            # float64 makes omission of out.float() observable in the final dtype.
            return ([torch.full_like(kwargs["x"][0], 0.25, dtype=torch.float64)],)

    generator = inference.Resfusion_zimage_gen.__new__(inference.Resfusion_zimage_gen)
    torch.nn.Module.__init__(generator)
    generator.vae = VAE()
    generator.hq_transformer = Transformer()
    scheduler = SimpleNamespace(rsr=1.0, timesteps=torch.tensor([500.0, 200.0]),
                                sigmas=torch.tensor([0.5, 0.2, 0.0]))
    generator.noise_scheduler = scheduler
    accelerator = SimpleNamespace(autocast=lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.full_like(x, 0.2))
    result = generator.resfusion_inference_loop(torch.zeros(1, 2, 2, 2), scheduler,
                                                prompt_embeds, accelerator)
    assert generator.num_sampling_steps == 2
    assert samples == [True]
    assert len(calls) == 4
    for index, expected_t in enumerate((0.5, 0.8)):
        write, read = calls[2 * index:2 * index + 2]
        assert write["joint_attention_kwargs"] == {"reference_mode": "write", "scale": 0.0}
        read_kwargs = read["joint_attention_kwargs"]
        assert set(read_kwargs) == {"reference_mode", "block_k_samples", "block_v_samples",
                                    "block_reference_attention_masks"}
        assert read_kwargs["reference_mode"] == "read"
        assert read_kwargs["block_k_samples"] is block_k
        assert read_kwargs["block_v_samples"] is block_v
        assert read_kwargs["block_reference_attention_masks"] is block_masks
        for call in (write, read):
            assert call["cap_feats"] is prompt_embeds
            assert call["return_dict"] is False
            assert call["x"].shape == (1, 2, 1, 2, 2)
            torch.testing.assert_close(call["t"], torch.tensor([expected_t]))
        torch.testing.assert_close(write["x"], torch.full_like(write["x"], 0.7))
    # z_LQ=.7, start=.45, velocity=-.25, total delta_sigma=-.5 => z_final=.575.
    # VAE decode input is .575 / 2 + .25 = .5375.
    assert result.dtype == torch.float32
    assert result.shape == (1, 2, 2, 2)
    assert len(decoded) == 1
    torch.testing.assert_close(result, torch.full_like(result, 0.5375))


def _identity_attention(torch):
    return SimpleNamespace(heads=2, norm_q=None, norm_k=None,
                           to_q=torch.nn.Identity(), to_k=torch.nn.Identity(), to_v=torch.nn.Identity(),
                           to_out=[torch.nn.Identity()])


def test_processor_kv_layout_and_concatenated_masks(runtime, monkeypatch):
    torch, reference = runtime.torch, runtime.reference
    dispatches = []

    def dispatch(query, key, value, **kwargs):
        dispatches.append((query, key, value, kwargs))
        return query

    monkeypatch.setattr(reference, "dispatch_attention_fn", dispatch)
    processor = reference.ZSingleStreamReferenceAttnProcessor()
    attn = _identity_attention(torch)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    mask = torch.tensor([[True, True, False]])
    k, v, reference_mask, output = processor(attn, hidden, attention_mask=mask, return_kv=True)
    assert k.shape == v.shape == (1, 3, 2, 4)  # batch, sequence, heads, head_dim
    assert reference_mask is mask
    torch.testing.assert_close(k, hidden.unflatten(-1, (2, 4)))
    torch.testing.assert_close(output, hidden)
    assert dispatches[0][3]["attn_mask"].shape == (1, 1, 1, 3)

    current = torch.ones(1, 2, 8)
    current_mask = torch.tensor([[True, False]])
    result = processor(attn, current, attention_mask=current_mask, k_hidden_states=k,
                       v_hidden_states=v, reference_attention_mask=reference_mask)
    query, key, value, kwargs = dispatches[1]
    assert query.shape == (1, 2, 2, 4)
    assert key.shape == value.shape == (1, 5, 2, 4)
    torch.testing.assert_close(key[:, :2], current.unflatten(-1, (2, 4)))
    torch.testing.assert_close(key[:, 2:], k)
    torch.testing.assert_close(value[:, 2:], v)
    assert kwargs["attn_mask"].shape == (1, 1, 1, 5)
    assert kwargs["attn_mask"].flatten().tolist() == [True, False, True, True, False]
    assert kwargs["backend"] is None and kwargs["parallel_config"] is None
    assert kwargs["dropout_p"] == 0.0 and kwargs["is_causal"] is False
    assert result.shape == current.shape


@pytest.mark.parametrize("kind", ["only_key", "only_value", "different_lengths"])
def test_processor_rejects_invalid_reference_pairs(runtime, kind):
    torch, reference = runtime.torch, runtime.reference
    key = torch.zeros(1, 3, 2, 4)
    value = torch.zeros(1, 4 if kind == "different_lengths" else 3, 2, 4)
    if kind == "only_key":
        value = None
    elif kind == "only_value":
        key = None
    with pytest.raises(ValueError, match="Both k_hidden_states|same sequence length"):
        reference.ZSingleStreamReferenceAttnProcessor()(
            _identity_attention(torch), torch.zeros(1, 2, 8), attention_mask=torch.ones(1, 2, dtype=torch.bool),
            k_hidden_states=key, v_hidden_states=value,
            reference_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        )


@pytest.fixture
def tiny_reference(runtime):
    from peft import LoraConfig
    torch, reference = runtime.torch, runtime.reference
    torch.manual_seed(42)
    model = reference.ZImageReferenceTransformer2DModel(
        in_channels=2, dim=16, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
        cap_feat_dim=6, axes_dims=[2, 2, 4], axes_lens=[128, 8, 8],
    )
    # Pretrained pad tokens are learned; initialise the empty tiny-model tokens.
    with torch.no_grad():
        model.x_pad_token.zero_()
        model.cap_pad_token.zero_()
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"], init_lora_weights=False),
                      adapter_name="default_transformer")
    model.eval()
    lora_layers = [module for module in model.modules()
                   if hasattr(module, "lora_A") and "default_transformer" in module.lora_A]
    assert len(lora_layers) == 4  # noise/context refiners and both main blocks
    inputs = dict(x=[torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)],
                  t=torch.tensor([0.5, 0.8]), cap_feats=[torch.randn(3, 6), torch.randn(35, 6)],
                  return_dict=False)
    return model, lora_layers, inputs


def test_reference_write_read_restores_lora_scale_and_batch_masks(runtime, tiny_reference):
    torch = runtime.torch
    model, lora_layers, inputs = tiny_reference
    scales_before = [dict(layer.scaling) for layer in lora_layers]
    assert all(scale["default_transformer"] == 1.0 for scale in scales_before)
    observed = []
    hooks = [layer.register_forward_pre_hook(
        lambda module, args: observed.append(module.scaling["default_transformer"]))
        for layer in lora_layers]
    try:
        with torch.no_grad():
            block_k, block_v, masks = model(**inputs, joint_attention_kwargs={"reference_mode": "write", "scale": 0.0})
        assert observed == [0.0] * 4
        assert [dict(layer.scaling) for layer in lora_layers] == scales_before
        assert len(block_k) == len(block_v) == len(masks) == 2
        for key, value, mask in zip(block_k, block_v, masks):
            assert key.shape == value.shape == (2, 96, 2, 8)
            assert mask.shape == (2, 96) and mask.dtype == torch.bool
            assert mask[0].sum().item() == 64 and mask[1].sum().item() == 96
            assert mask[0, :64].all() and not mask[0, 64:].any()
        observed.clear()
        with torch.no_grad():
            output = model(**inputs, joint_attention_kwargs={
                "reference_mode": "read", "block_k_samples": block_k, "block_v_samples": block_v,
                "block_reference_attention_masks": masks,
            })[0]
        assert observed == [1.0] * 4
        assert [dict(layer.scaling) for layer in lora_layers] == scales_before
        assert len(output) == 2
        assert all(item.shape == (2, 1, 4, 4) and torch.isfinite(item).all() for item in output)
    finally:
        for hook in hooks:
            hook.remove()


def test_reference_scale_restored_when_write_raises(runtime, tiny_reference):
    torch = runtime.torch
    model, lora_layers, inputs = tiny_reference
    scales_before = [dict(layer.scaling) for layer in lora_layers]

    def fail_attention(module, args):
        raise RuntimeError("intentional write failure")

    hook = model.layers[0].attention.to_q.register_forward_pre_hook(fail_attention)
    try:
        with torch.no_grad(), pytest.raises(RuntimeError, match="intentional write failure"):
            model(**inputs, joint_attention_kwargs={"reference_mode": "write", "scale": 0.0})
        assert [dict(layer.scaling) for layer in lora_layers] == scales_before
    finally:
        hook.remove()
