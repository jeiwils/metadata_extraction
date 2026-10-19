"""
Helpers shared by a_extract_pages.py and b_extract_metadata.py:
reading the settings file, and choosing which items to process.
"""

import random
import re
import tomllib
from pathlib import Path


class Config:
    """
    The settings file (TOML). Every setting is required: if one is missing, the
    run stops with a message naming it, instead of quietly using a default.
    """

    def __init__(self, file: str | Path):
        self.file = Path(file).resolve()
        with open(self.file, "rb") as f:
            self._settings = tomllib.load(f)

    def value(self, section: str, key: str):
        """The setting `key` from `[section]`."""
        try:
            return self._settings[section][key]
        except KeyError:
            raise SystemExit(f"{self.file.name} is missing '{key}' under [{section}]")

    def path(self, section: str, key: str) -> Path:
        """A file or folder setting. Relative paths count from the settings file's own folder."""
        return (self.file.parent / self.value(section, key)).resolve()


def items_to_process(items: list[Path], limit: int, random_sample: bool) -> list[Path]:
    """
    Which items to work on: all of them when limit is 0; otherwise the first
    `limit`, or a random `limit` if random_sample is true. The random choice
    uses a fixed seed, so every run picks the same items.
    """
    if not limit or limit >= len(items):
        return items
    if not random_sample:
        return items[:limit]
    chosen = set(random.Random(42).sample(sorted(items), limit))
    return [item for item in items if item in chosen]


def natural_sort_key(path: Path) -> list:
    """Sort key that reads numbers as numbers, so 'IMG_2.jpg' comes before 'IMG_10.jpg'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]