from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvRunLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CsvRunLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

