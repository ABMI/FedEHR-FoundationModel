"""Config loading, output paths and JSON writing shared by all stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tomllib
from pathlib import Path

RAW_ROOT = Path(os.environ.get("RAW_ROOT", "data/omop_parquet"))


def load_config(config_path: str) -> dict:
    """Load a TOML config and resolve raw_root and output_root to absolute paths."""
    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    raw_root = Path(config["paths"]["raw_root"]).expanduser().resolve()
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw folder not found: {raw_root}")
    if output_root == raw_root or output_root.is_relative_to(raw_root):
        raise ValueError("output_root must be outside raw_root.")

    output_root.mkdir(parents=True, exist_ok=True)
    config["paths"]["raw_root"] = raw_root
    config["paths"]["output_root"] = output_root
    config["config_path"] = str(Path(config_path).resolve())
    return config


def config_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", required=True, help="TOML config with raw and output paths")
    return parent


def source_parts(raw_root: Path, table: str) -> str:
    """Glob pattern for an OMOP table's Parquet parts: raw_root/dbo/TABLE/batch-*/*.parquet."""
    folder = raw_root / "dbo" / table
    if not folder.is_dir() or not any(folder.glob("batch-*/*.parquet")):
        raise FileNotFoundError(f"{table} Parquet files not found: {folder}")
    return str(folder / "batch-*" / "*.parquet")


def reference_root(config: dict, key: str = "root") -> Path | None:
    """Reference-site output folder from [reference].<key>, or None; a new site reuses its rules.

    root: base preprocessing (units, range checks, deciles, vocab); variant_root: combined-token vocab.
    """
    value = config.get("reference", {}).get(key)
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Reference output folder not found: {path}")
    return path


def copy_reference_file(reference: Path, config: dict, *parts: str) -> Path:
    """Copy one reference-site output file to the same relative path."""
    source = reference.joinpath(*parts)
    if not source.is_file():
        raise FileNotFoundError(f"Reference file not found: {source}")
    destination = output_file(config, *parts)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def output_file(config: dict, *parts: str) -> Path:
    """Path under output_root; parent folders are created."""
    path = config["paths"]["output_root"].joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def result_folder_parts(kind: str, initialization: str, run_name: str | None = None) -> tuple[str, ...]:
    if run_name is not None:
        # Restricted so a run name cannot overwrite older results or escape the folder.
        if initialization != "continued" or re.fullmatch(r"epoch[1-9][0-9]*", run_name) is None:
            raise ValueError("run_name is only for continued models, e.g. epoch20.")
    if initialization == "pretrained":
        return (kind,)
    if run_name is None:
        return (kind, initialization)
    return (kind, initialization, run_name)


def source_fingerprint(raw_root: Path) -> str:
    """SHA-256 of _COMPLETE.json; Parquet contents are not hashed."""
    marker = raw_root / "_COMPLETE.json"
    return hashlib.sha256(marker.read_bytes()).hexdigest()


def export_finish_date(raw_root: Path) -> str:
    """Export finish date (YYYY-MM-DD) from _COMPLETE.json."""
    marker = json.loads((raw_root / "_COMPLETE.json").read_text(encoding="utf-8"))
    return marker["finished_at"][:10]


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via a .tmp file."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
