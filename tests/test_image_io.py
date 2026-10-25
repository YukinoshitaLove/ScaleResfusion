import numpy as np
import pytest
import torch
from torchvision.transforms import ToPILImage

from my_utils.image_io import restored_tensor_to_pil


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_supported_dtypes_keep_existing_quantisation(dtype):
    value = torch.linspace(-1, 1, 3 * 11 * 13, dtype=dtype).reshape(3, 11, 13)
    expected = ToPILImage()(value.cpu() * 0.5 + 0.5)
    assert np.array_equal(np.array(restored_tensor_to_pil(value)), np.array(expected))


def test_bfloat16_retains_affine_rounding_then_converts_for_pillow():
    value = torch.linspace(-1, 1, 3 * 11 * 13, dtype=torch.bfloat16).reshape(3, 11, 13)
    expected = ToPILImage()((value.cpu() * 0.5 + 0.5).float())
    actual = restored_tensor_to_pil(value)
    assert actual.mode == "RGB"
    assert actual.size == (13, 11)
    assert np.array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_invalid_model_output_is_not_silently_saved_as_black(invalid):
    value = torch.zeros(3, 4, 4)
    value[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="non-finite"):
        restored_tensor_to_pil(value)
