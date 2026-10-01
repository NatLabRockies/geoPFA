"""Tests for self-preparing public Newberry tutorial inputs."""

from __future__ import annotations

from pathlib import Path

from geopfa import datasets


def test_newberry_preparation_downloads_then_builds_missing_processed_data(
    tmp_path: Path, monkeypatch
) -> None:
    """A fresh example directory receives release and derived inputs once."""
    calls: list[tuple[str, object]] = []

    def fake_download(data_dir: Path) -> None:
        calls.append(("download", data_dir))
        data_dir.mkdir(parents=True, exist_ok=True)

    def fake_prepare(project_dir: Path, dimensions: str) -> None:
        calls.append(("prepare", dimensions))
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "data" / "heat_processed.csv").write_text("x\n")
        (project_dir / "config").mkdir(exist_ok=True)
        (
            project_dir / "config" / "newberry_superhot_processed_config.json"
        ).write_text("{}")

    monkeypatch.setattr(
        datasets, "setup_newberry_tutorial_data", fake_download
    )
    monkeypatch.setattr(
        datasets, "_prepare_newberry_processed_data", fake_prepare
    )

    config_path, data_dir = datasets.ensure_newberry_tutorial_processed_data(
        tmp_path, dimensions="2d"
    )

    assert calls == [("download", tmp_path / "data"), ("prepare", "2d")]
    assert config_path.is_file()
    assert data_dir == tmp_path / "data"


def test_newberry_preparation_reuses_complete_local_processed_data(
    tmp_path: Path, monkeypatch
) -> None:
    """Rerunning an example never recomputes an intact local preparation."""
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    config_dir.mkdir()
    (data_dir / "heat_processed.csv").write_text("x\n")
    (config_dir / "newberry_superhot_processed_config.json").write_text("{}")
    calls: list[Path] = []

    def record_download(target: Path) -> None:
        calls.append(target)

    monkeypatch.setattr(
        datasets, "setup_newberry_tutorial_data", record_download
    )
    monkeypatch.setattr(
        datasets,
        "_prepare_newberry_processed_data",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not prepare")),
    )

    config_path, data_dir = datasets.ensure_newberry_tutorial_processed_data(
        tmp_path, dimensions="3d"
    )

    assert calls == [tmp_path / "data"]
    assert config_path.is_file()
    assert data_dir == tmp_path / "data"
