"""Versioned configuration loading with deterministic hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        try:
            value = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"Cannot parse configuration {path}: {exc}") from exc
        if value is None:
            raise ConfigError(f"Configuration {path} is empty") from json_error
    if not isinstance(value, dict):
        raise ConfigError("Configuration root must be a mapping")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    path: Path
    raw: dict[str, Any]
    input: dict[str, Any]
    output: dict[str, Any]
    sha256: str

    def canonical_json(self) -> str:
        return json.dumps(self.raw, sort_keys=True, separators=(",", ":"))


def load_config(path: str | Path) -> ResolvedConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _load_mapping(config_path)
    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must be exactly 1")

    input_config = _require_mapping(raw.get("input"), "input")
    output_config = _require_mapping(raw.get("output", {}), "output")
    source_kind = input_config.get("source")
    if source_kind not in {"video", "npy_manifest"}:
        raise ConfigError("input.source must be 'video' or 'npy_manifest'")
    source_path_value = input_config.get("path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ConfigError("input.path must be a non-empty string")

    source_path = Path(source_path_value).expanduser()
    if not source_path.is_absolute():
        source_path = (config_path.parent / source_path).resolve()
    input_config["path"] = str(source_path)

    timestamp_csv = input_config.get("timestamp_csv")
    if timestamp_csv is not None:
        if not isinstance(timestamp_csv, str) or not timestamp_csv:
            raise ConfigError("input.timestamp_csv must be a non-empty string")
        timestamp_path = Path(timestamp_csv).expanduser()
        if not timestamp_path.is_absolute():
            timestamp_path = (config_path.parent / timestamp_path).resolve()
        input_config["timestamp_csv"] = str(timestamp_path)

    max_frames = input_config.get("max_frames")
    if max_frames is not None and (
        not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0
    ):
        raise ConfigError("input.max_frames must be a positive integer or null")

    resolved = dict(raw)
    resolved["input"] = input_config
    resolved["output"] = output_config
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return ResolvedConfig(
        path=config_path,
        raw=resolved,
        input=input_config,
        output=output_config,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
