# Copyright 2026 Youngseok Lee
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    config_path: Path
    root: Path

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    @property
    def outputs_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["outputs_dir"])

    @property
    def train_data(self) -> Path:
        return self.resolve(self.raw["paths"]["train_data"])

    @property
    def test_data(self) -> Path:
        return self.resolve(self.raw["paths"]["test_data"])


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = config_path.parent.parent.resolve()
    return ExperimentConfig(raw=raw, config_path=config_path, root=root)

