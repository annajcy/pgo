from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_make_cloth_grid() -> None:
    import importlib.util

    script = (
        Path(__file__).resolve().parents[2]
        / "examples/python/mass_spring_cloth.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mass_spring_cloth", str(script)
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    vertices, triangles, pinned = mod.make_cloth_grid(3)
    assert vertices.shape == (9, 3)
    assert triangles.shape == (8, 3)
    assert pinned.shape == (3,)


def test_python_cloth_example_smoke(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "examples/python/mass_spring_cloth.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frames",
            "2",
            "--resolution",
            "3",
            "--output",
            str(tmp_path / "frames"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert any((tmp_path / "frames").glob("*.obj"))
