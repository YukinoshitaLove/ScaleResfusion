import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ("inference_sd3_2b.sh", "test_resfusion_sd3.py"),
    ("inference_zimage_6b.sh", "test_resfusion_zimage.py"),
    ("inference_flux2_9b.sh", "test_resfusion_flux2.py"),
)


@pytest.mark.parametrize("script,entry", SCRIPTS)
@pytest.mark.parametrize("caption_state", ["absent", "empty", "partial", "complete"])
def test_wrappers_generate_missing_captions(tmp_path, script, entry, caption_state):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required for the release shell wrappers")
    lr = tmp_path / "LR"
    hr = tmp_path / "HR"
    lr.mkdir()
    hr.mkdir()
    for name in ("first.part.png", "second.JPG"):
        (lr / name).touch()
    if caption_state != "absent":
        (lr / "txt").mkdir()
    if caption_state in ("partial", "complete"):
        (lr / "txt/first.part.txt").write_text("building, ,", encoding="utf-8")
    if caption_state == "complete":
        (lr / "txt/second.txt").write_text("tree, ,", encoding="utf-8")
    weights = []
    for name in ("lora.safetensors", "ram.pth", "dape.pth"):
        path = tmp_path / name
        path.touch()
        weights.append(str(path))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$1" >> "$STAGE_LOG"\n', encoding="utf-8")
    python_stub.chmod(0o755)
    log = tmp_path / "stages.txt"
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""), "STAGE_LOG": str(log)}
    completed = subprocess.run(
        [bash, str(ROOT / "scripts" / script), str(lr), str(hr), str(tmp_path / "out"), *weights],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    expected = [entry, "test_metrics.py"]
    if caption_state != "complete":
        expected.insert(0, "generate_captions.py")
    assert log.read_text(encoding="utf-8").splitlines() == expected
