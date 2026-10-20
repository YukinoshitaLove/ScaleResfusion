"""Lightweight shared inference I/O tests; no model or GPU imports.

Run with ``python -m pytest tests/test_inference_io.py --basetemp <sandbox>``.
All filesystem fixtures are created beneath pytest's ``tmp_path``.
"""

import os

import pytest
from PIL import Image

from my_utils.captions import get_caption_path, load_caption
from my_utils.preprocess import list_images, prepare_input_image


@pytest.mark.parametrize(
    ("image_name", "caption_name"),
    [
        ("photo.png", "photo.txt"),
        ("photo.edit.v2.JPG", "photo.edit.v2.txt"),
        ("夜景.café.v1.webp", "夜景.café.v1.txt"),
    ],
)
def test_get_caption_path_uses_input_directory_and_complete_stem(
    tmp_path, image_name, caption_name
):
    input_dir = tmp_path / "inputs"
    image_path = tmp_path / "elsewhere" / image_name

    assert get_caption_path(str(input_dir), str(image_path)) == str(
        input_dir / "txt" / caption_name
    )


@pytest.mark.parametrize("single_image", [False, True], ids=["directory", "single-image"])
def test_caption_lookup_for_directory_and_single_image_inputs(tmp_path, single_image):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_path = input_dir / "scene.edit.v2.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)
    caption_dir = input_dir / "txt"
    caption_dir.mkdir()
    (caption_dir / "scene.edit.v2.txt").write_text("夜景, café\n", encoding="utf-8")
    source = str(image_path if single_image else input_dir)

    image_paths = list_images(source)
    # This is the same input-root selection used by the inference CLIs.
    caption_root = source if os.path.isdir(source) else os.path.dirname(source)

    assert image_paths == [str(image_path)]
    assert load_caption(caption_root, image_paths[0]) == "夜景, café"


@pytest.mark.parametrize(
    ("stored_caption", "expected"),
    [
        (" \n  a detailed scene  \t\n", "a detailed scene"),
        (" \t夜景, café, façades\nsecond line\n ", "夜景, café, façades\nsecond line"),
        ("", ""),
        (" \n\t ", ""),
    ],
    ids=["trim-boundaries", "utf8-and-multiline", "empty-file", "whitespace-only"],
)
def test_existing_caption_is_utf8_trimmed_and_takes_precedence(
    tmp_path, stored_caption, expected
):
    caption_dir = tmp_path / "txt"
    caption_dir.mkdir()
    image_path = tmp_path / "scene.v2.png"
    (caption_dir / "scene.v2.txt").write_text(stored_caption, encoding="utf-8")

    assert load_caption(str(tmp_path), str(image_path), "fallback") == expected


@pytest.mark.parametrize("fallback", ["a fixed caption", "", " \tunchanged\n "])
def test_missing_caption_returns_fallback_verbatim(tmp_path, fallback):
    image_path = tmp_path / "scene.v2.png"

    assert load_caption(
        str(tmp_path), str(image_path), fallback_prompt=fallback
    ) == fallback


@pytest.mark.parametrize("explicit_none", [False, True], ids=["default", "explicit-none"])
def test_missing_caption_without_fallback_reports_expected_path(tmp_path, explicit_none):
    image_path = tmp_path / "scene.v2.png"
    expected_caption = tmp_path / "txt" / "scene.v2.txt"
    kwargs = {"fallback_prompt": None} if explicit_none else {}

    with pytest.raises(FileNotFoundError) as error:
        load_caption(str(tmp_path), str(image_path), **kwargs)

    assert str(expected_caption) in str(error.value)


def test_list_images_sorts_supported_extensions_and_ignores_nested_files(tmp_path):
    image_names = [
        "z.png",
        "A.JPG",
        "m.jpeg",
        "c.BMP",
        "b.webp",
        "a.PNG",
        "n.jpg",
        "e.JPEG",
        "d.bmp",
        "y.WEBP",
    ]
    for name in image_names:
        (tmp_path / name).touch()
    for name in ("readme.txt", "notes.md", "animation.gif", "other.tif"):
        (tmp_path / name).touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.png").touch()
    caption_dir = tmp_path / "txt"
    caption_dir.mkdir()
    (caption_dir / "z.txt").write_text("caption", encoding="utf-8")

    assert list_images(str(tmp_path)) == sorted(
        str(tmp_path / name) for name in image_names
    )


def test_list_images_accepts_single_image_without_scanning_its_siblings(tmp_path):
    image_path = tmp_path / "scene.edit.v2.PNG"
    image_path.touch()
    (tmp_path / "another.png").touch()

    assert list_images(str(image_path)) == [str(image_path)]


def test_list_images_preserves_a_relative_single_image_path(tmp_path, monkeypatch):
    (tmp_path / "scene.png").touch()
    monkeypatch.chdir(tmp_path)

    assert list_images("scene.png") == ["scene.png"]


def test_list_images_returns_empty_list_for_empty_directory(tmp_path):
    assert list_images(str(tmp_path)) == []


def test_list_images_missing_input_reports_path(tmp_path):
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError) as error:
        list_images(str(missing))

    assert str(missing) in str(error.value)


def _patterned_rgb(size):
    width, height = size
    image = Image.new("RGB", size)
    image.putdata(
        [
            ((x * 47 + y * 19) % 256, (x * 13 + y * 71) % 256, (x * 89 + y * 31) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return image


@pytest.mark.parametrize(
    ("original_size", "prepared_size", "expected_resize_flag"),
    [
        ((20, 16), (80, 64), False),
        ((19, 21), (64, 80), False),
        ((21, 19), (80, 64), False),
        ((16, 16), (64, 64), False),
        ((5, 7), (64, 80), True),
        ((7, 5), (80, 64), True),
        ((15, 16), (64, 64), True),
    ],
    ids=[
        "exact-four-times",
        "floor-portrait",
        "floor-landscape",
        "threshold-is-not-enlarged",
        "tiny-portrait",
        "tiny-landscape",
        "below-threshold",
    ],
)
def test_prepare_input_image_dimensions_and_original_size(
    original_size, prepared_size, expected_resize_flag
):
    image = _patterned_rgb(original_size)
    original_pixels = image.tobytes()

    prepared, returned_original_size, resize_flag = prepare_input_image(
        image, upscale=4, process_size=64
    )

    assert returned_original_size == original_size
    assert resize_flag is expected_resize_flag
    assert prepared.mode == "RGB"
    assert prepared.size == prepared_size
    assert prepared.width % 16 == prepared.height % 16 == 0
    # The original dimensions must survive preprocessing for final x4 restoration.
    assert image.size == original_size
    assert image.tobytes() == original_pixels


def test_prepare_input_image_defaults_to_four_times_upscaling():
    image = _patterned_rgb((8, 12))

    prepared, original_size, resize_flag = prepare_input_image(image, process_size=16)

    assert prepared.size == (32, 48)
    assert original_size == (8, 12)
    assert resize_flag is False


@pytest.mark.parametrize(
    ("original_size", "enlarged_size", "upscaled_size", "aligned_size"),
    [
        ((19, 21), None, (76, 84), (64, 80)),
        ((5, 7), (16, 22), (64, 88), (64, 80)),
    ],
    ids=["no-enlargement", "tiny-image-enlargement"],
)
def test_prepare_input_image_resizes_rather_than_crops_for_alignment(
    original_size, enlarged_size, upscaled_size, aligned_size
):
    image = _patterned_rgb(original_size)
    expected = image if enlarged_size is None else image.resize(enlarged_size)
    expected = expected.resize(upscaled_size)
    expected = expected.resize(aligned_size, Image.Resampling.LANCZOS)

    prepared, _, _ = prepare_input_image(image, upscale=4, process_size=64)

    assert prepared.size == expected.size
    assert prepared.tobytes() == expected.tobytes()
