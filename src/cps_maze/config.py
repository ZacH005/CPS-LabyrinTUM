from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def camera(self) -> dict[str, Any]:
        return self.raw["camera"]

    @property
    def serial(self) -> dict[str, Any]:
        return self.raw["serial"]

    @property
    def servo(self) -> dict[str, Any]:
        return self.raw["servo"]

    @property
    def vision(self) -> dict[str, Any]:
        return self.raw["vision"]

    @property
    def control(self) -> dict[str, Any]:
        return self.raw["control"]

    @property
    def maze(self) -> dict[str, Any]:
        return self.raw["maze"]

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.path.parent.parent / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw, path=config_path)
