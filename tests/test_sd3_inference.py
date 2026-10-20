"""Small SD3 interface and sampling tests using mock and tiny models."""

import argparse
import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.attention_processor import Attention
from safetensors.torch import save_file

import resfusion_sd3 as sd3
from ReferenceNet_Modules.sd3_referencenet import (
    SD3ReferenceNetReadModel,
    SD3ReferenceNetWriteModel,
)
from Reference_Atten.sd3_reference_attn_processor import (
    SD3ReferenceReadJointAttnProcessor2_0,
    SD3ReferenceWriteJointAttnProcessor2_0,
    set_sd3_reference_read_attn,
    set_sd3_reference_write_attn,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_kwargs(num_layers):
    return dict(
        sample_size=4, patch_size=2, in_channels=4, out_channels=4,
        num_layers=num_layers, attention_head_dim=4, num_attention_heads=2,
        joint_attention_dim=6, caption_projection_dim=8,
        pooled_projection_dim=5, pos_embed_max_size=4,
    )


def test_cli_defaults_and_boolean_optional_action(monkeypatch):
    # Execute only the parser definition; no CLI-level model/image imports.
    tree = ast.parse((ROOT / "test_resfusion_sd3.py").read_text(encoding="utf-8"))
    parser_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[parser_fn], type_ignores=[]), "sd3_cli_parser", "exec"), namespace)
    monkeypatch.setattr("sys.argv", ["sd3", "-i", "images", "-o", "out", "--resfusion_path", "gen.safetensors"])
    args = namespace["parse_args"]()
    assert (args.seed, args.flowT, args.rsr, args.upscale, args.process_size, args.lq_layers) == (42, 20, 1.0, 4, 512, 23)
    assert args.mixed_precision == "fp16"
    assert args.save_prompts is True
    assert args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-3-medium-diffusers"
    assert not hasattr(args, "lora_rank")
    monkeypatch.setattr("sys.argv", [*sys.argv, "--no-save_prompts", "--mixed_precision", "bf16"])
    args = namespace["parse_args"]()
    assert args.save_prompts is False
    assert args.mixed_precision == "bf16"


def test_missing_checkpoint_is_rejected_before_loading(monkeypatch, tmp_path):
    def unexpected_load(*args, **kwargs):
        pytest.fail("A missing checkpoint must fail before any base-model loading")

    monkeypatch.setattr(sd3.SD3ReferenceNetReadModel, "from_pretrained", unexpected_load)
    for path in (None, "", str(tmp_path), str(tmp_path / "missing.safetensors")):
        with pytest.raises(FileNotFoundError, match="resfusion_path"):
            sd3.initialize_read_transformer("unused-base", path)


def test_cuda_is_required_before_model_loading(monkeypatch, tmp_path):
    checkpoint = tmp_path / "gen.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(resfusion_path=str(checkpoint))
    with pytest.raises(EnvironmentError, match="requires CUDA"):
        sd3.Resfusion_sd3_test(args)
    with pytest.raises(EnvironmentError, match="requires CUDA"):
        sd3.Resfusion_sd3_gen(args, str(checkpoint))


@pytest.mark.parametrize("precision,dtype", [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)])
def test_text_only_pipeline_retains_three_encoders(monkeypatch, tmp_path, precision, dtype):
    checkpoint = tmp_path / "gen.safetensors"
    checkpoint.touch()
    calls = {}

    class Encoder:
        def eval(self):
            self.is_eval = True

    class TextPipeline:
        def __init__(self):
            self.text_encoder = Encoder()
            self.text_encoder_2 = Encoder()
            self.text_encoder_3 = Encoder()

        def to(self, device, dtype):
            calls["pipe_to"] = (device.type, dtype)
            return self

        def encode_prompt(self, prompt, **kwargs):
            calls["prompt"] = (prompt, kwargs)
            return torch.zeros(1, 3, 6), None, torch.zeros(1, 5), None

    class Generator(torch.nn.Module):
        def __init__(self, args, ckpt_file):
            super().__init__()
            calls["generator_checkpoint"] = ckpt_file

        def to(self, device, dtype):
            calls["generator_to"] = (device.type, dtype)
            return self

    def text_from_pretrained(base, **kwargs):
        calls["pipeline_load"] = (base, kwargs)
        return TextPipeline()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sd3, "Resfusion_sd3_gen", Generator)
    monkeypatch.setattr(sd3.StableDiffusion3Pipeline, "from_pretrained", text_from_pretrained)
    monkeypatch.setattr(sd3, "Accelerator", lambda **kwargs: calls.setdefault("accelerator", kwargs))
    args = SimpleNamespace(
        resfusion_path=str(checkpoint), mixed_precision=precision,
        pretrained_model_name_or_path="sd3-base",
    )
    wrapper = sd3.Resfusion_sd3_test(args)
    assert calls["pipeline_load"] == ("sd3-base", dict(transformer=None, vae=None, scheduler=None, torch_dtype=dtype))
    assert calls["generator_to"] == calls["pipe_to"] == ("cuda", dtype)
    assert calls["accelerator"] == {"mixed_precision": "no" if precision == "fp32" else precision}
    assert wrapper.pipe.text_encoder.is_eval
    assert wrapper.pipe.text_encoder_2.is_eval
    assert wrapper.pipe.text_encoder_3.is_eval
    prompt_embeds, pooled_embeds = wrapper.encode_prompt("caption")
    assert prompt_embeds.dtype == pooled_embeds.dtype == dtype
    prompt, kwargs = calls["prompt"]
    assert prompt == kwargs["prompt_2"] == kwargs["prompt_3"] == "caption"
    assert kwargs["device"].type == "cuda"
    assert kwargs["do_classifier_free_guidance"] is False


def test_generator_uses_independent_23_layer_writer_and_requested_dtype(monkeypatch, tmp_path):
    checkpoint = tmp_path / "gen.safetensors"
    checkpoint.touch()
    calls = {}

    class Sigmas:
        def cuda(self):
            return self

    class Scheduler:
        def __init__(self, **kwargs):
            calls["schedule"] = kwargs
            self.sigmas = Sigmas()
            self.timesteps = [500, 350, 200, 50]

        def set_timesteps(self, **kwargs):
            calls["set_timesteps"] = kwargs

    def load_vae(base, **kwargs):
        calls["vae"] = (base, kwargs)
        return torch.nn.Identity()

    def load_writer(base, **kwargs):
        calls["writer"] = (base, kwargs)
        return torch.nn.Identity()

    def load_reader(base, ckpt_file, **kwargs):
        calls["reader"] = (base, ckpt_file, kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sd3, "ResfusionFlowMatchScheduler", Scheduler)
    monkeypatch.setattr(sd3.AutoencoderKL, "from_pretrained", load_vae)
    monkeypatch.setattr(sd3.SD3ReferenceNetWriteModel, "from_pretrained", load_writer)
    monkeypatch.setattr(sd3, "set_sd3_reference_write_attn", lambda model: model)
    monkeypatch.setattr(sd3, "initialize_read_transformer", load_reader)
    # The generator also defaults to 23 when used without the CLI namespace.
    args = SimpleNamespace(mixed_precision="bf16", pretrained_model_name_or_path="sd3-base", rsr=1.0, flowT=20)
    generator = sd3.Resfusion_sd3_gen(args, str(checkpoint))
    assert generator.lq_transformer is not generator.hq_transformer
    assert calls["writer"] == ("sd3-base", dict(subfolder="transformer", num_layers=23))
    assert calls["vae"] == ("sd3-base", dict(subfolder="vae"))
    assert calls["reader"] == ("sd3-base", str(checkpoint), {})
    assert calls["schedule"] == dict(num_train_timesteps=1000, shift=3.0, rsr=1.0)
    assert calls["set_timesteps"] == dict(num_inference_steps=20, device="cuda")
    assert generator.num_sampling_steps == 4


def test_original_sample_shift_per_step_kv_timestep_and_euler(monkeypatch):
    sampled = torch.linspace(-0.2, 0.2, 16).reshape(1, 4, 2, 2)
    noise = torch.full_like(sampled, 0.1)
    shift, scaling = 0.07, 0.23
    writer_calls, reader_calls, decoded = [], [], []

    class Posterior:
        def sample(self):
            return sampled.clone()

        def mode(self):
            pytest.fail("SD3 must sample the VAE posterior")

    class VAE:
        config = SimpleNamespace(shift_factor=shift, scaling_factor=scaling)

        def encode(self, image):
            return SimpleNamespace(latent_dist=Posterior())

        def decode(self, latent):
            decoded.append(latent.clone())
            return SimpleNamespace(sample=latent)

    def writer(**kwargs):
        kv = ((kwargs["timestep"].clone(),), (kwargs["timestep"].clone() + 1,))
        writer_calls.append((kwargs, kv))
        return kv

    def reader(**kwargs):
        reader_calls.append(kwargs)
        return (torch.full_like(kwargs["hidden_states"], 0.03 * len(reader_calls)),)

    model = sd3.Resfusion_sd3_gen.__new__(sd3.Resfusion_sd3_gen)
    torch.nn.Module.__init__(model)
    model.vae, model.lq_transformer, model.hq_transformer = VAE(), writer, reader
    scheduler = SimpleNamespace(rsr=1.0, timesteps=torch.tensor([500.0, 350.0, 200.0]), sigmas=torch.tensor([0.5, 0.35, 0.2, 0.0]))
    model.noise_scheduler = scheduler
    monkeypatch.setattr(torch, "randn_like", lambda latent: noise.clone())
    output = model.resfusion_inference_loop(
        torch.zeros_like(sampled), scheduler, torch.zeros(1, 3, 6), torch.zeros(1, 5),
        SimpleNamespace(autocast=lambda **kwargs: nullcontext()),
    )
    lq_latent = (sampled - shift) * scaling
    expected = 0.5 * lq_latent + 0.5 * noise
    assert len(writer_calls) == len(reader_calls) == model.num_sampling_steps == 3
    for i, ((kwargs, kv), reader_kwargs) in enumerate(zip(writer_calls, reader_calls)):
        torch.testing.assert_close(kwargs["referencenet_cond"], lq_latent)
        torch.testing.assert_close(kwargs["timestep"], scheduler.timesteps[i:i + 1])
        torch.testing.assert_close(reader_kwargs["timestep"], scheduler.timesteps[i:i + 1])
        assert reader_kwargs["block_k_samples"] is kv[0]
        assert reader_kwargs["block_v_samples"] is kv[1]
        torch.testing.assert_close(reader_kwargs["hidden_states"], expected)
        expected = expected + (scheduler.sigmas[i + 1] - scheduler.sigmas[i]) * (0.03 * (i + 1))
    expected = expected / scaling + shift
    torch.testing.assert_close(decoded[0], expected)
    torch.testing.assert_close(output, expected.clamp(-1, 1))


def test_tiny_reference_models_keep_image_only_kv_and_output_shape():
    torch.manual_seed(0)
    writer = set_sd3_reference_write_attn(SD3ReferenceNetWriteModel(**_tiny_kwargs(1))).eval()
    reader = set_sd3_reference_read_attn(SD3ReferenceNetReadModel(**_tiny_kwargs(2))).eval()
    latent, text = torch.randn(1, 4, 4, 4), torch.randn(1, 3, 6)
    conditions = dict(timestep=torch.tensor([500.0]), pooled_projections=torch.randn(1, 5), encoder_hidden_states=text)
    with torch.no_grad():
        keys, values = writer(referencenet_cond=latent, **conditions)
        output = reader(hidden_states=latent, block_k_samples=keys, block_v_samples=values, return_dict=False, **conditions)[0]
    assert len(keys) == len(values) == 1
    # Four image tokens only; the three text tokens are not written to the bank.
    assert keys[0].shape == values[0].shape == (1, 2, 4, 4)
    assert output.shape == latent.shape
    assert torch.isfinite(output).all()


def test_attention_writer_and_reader_preserve_original_concatenation_order():
    torch.manual_seed(1)
    attn = Attention(query_dim=8, added_kv_proj_dim=8, dim_head=4, heads=2, out_dim=8, context_pre_only=False, bias=True)
    image, text = torch.randn(1, 4, 8), torch.randn(1, 3, 8)

    def split_heads(tensor):
        return tensor.view(1, -1, 2, 4).transpose(1, 2)

    image_q, image_k, image_v = (split_heads(proj(image)) for proj in (attn.to_q, attn.to_k, attn.to_v))
    text_q, text_k, text_v = (split_heads(proj(text)) for proj in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
    queries = torch.cat([image_q, text_q], dim=2)

    def expected_output(keys, values):
        scores = queries @ keys.transpose(-1, -2) / (queries.shape[-1] ** 0.5)
        output = (scores.softmax(dim=-1) @ values).transpose(1, 2).reshape(1, 7, 8)
        return attn.to_out[1](attn.to_out[0](output[:, :4])), attn.to_add_out(output[:, 4:])

    with torch.no_grad():
        k_bank, v_bank, written_image, written_text = SD3ReferenceWriteJointAttnProcessor2_0()(attn, image, text)
        torch.testing.assert_close(k_bank, image_k)
        torch.testing.assert_close(v_bank, image_v)
        expected_image, expected_text = expected_output(torch.cat([image_k, text_k], dim=2), torch.cat([image_v, text_v], dim=2))
        torch.testing.assert_close(written_image, expected_image)
        torch.testing.assert_close(written_text, expected_text)
        reference_k, reference_v = image_k + 0.3, image_v - 0.2
        read_image, read_text = SD3ReferenceReadJointAttnProcessor2_0()(attn, image, text, k_hidden_states=reference_k, v_hidden_states=reference_v)
        expected_image, expected_text = expected_output(
            torch.cat([image_k, text_k, reference_k], dim=2),
            torch.cat([image_v, text_v, reference_v], dim=2),
        )
        torch.testing.assert_close(read_image, expected_image)
        torch.testing.assert_close(read_text, expected_text)


def test_checkpoint_loader_infers_rank_and_keeps_conv2d_lora(monkeypatch, tmp_path):
    # Both target kinds occur in the released SD3 adapter; no base weights are downloaded.
    model = SD3ReferenceNetReadModel(**_tiny_kwargs(2))
    rank = 2
    tensors = {
        "transformer.pos_embed.proj.lora_A.weight": torch.randn(rank, 4, 2, 2),
        "transformer.pos_embed.proj.lora_B.weight": torch.randn(8, rank, 1, 1),
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(rank, 8),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(8, rank),
    }
    checkpoint = tmp_path / "gen_hq_transformer_lora.safetensors"
    save_file(tensors, str(checkpoint))
    monkeypatch.setattr(sd3.SD3ReferenceNetReadModel, "from_pretrained", lambda *args, **kwargs: model)
    loaded = sd3.initialize_read_transformer("unused-base", str(checkpoint))
    adapter = "default_transformer"
    assert loaded.peft_config[adapter].r == rank
    assert isinstance(loaded.pos_embed.proj.get_base_layer(), torch.nn.Conv2d)
    torch.testing.assert_close(loaded.pos_embed.proj.lora_A[adapter].weight, tensors["transformer.pos_embed.proj.lora_A.weight"])
    torch.testing.assert_close(loaded.pos_embed.proj.lora_B[adapter].weight, tensors["transformer.pos_embed.proj.lora_B.weight"])
    torch.testing.assert_close(loaded.transformer_blocks[0].attn.to_q.lora_A[adapter].weight, tensors["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"])
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
