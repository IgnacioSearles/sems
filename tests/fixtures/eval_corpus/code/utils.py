"""Assorted helpers shared by the billing and content services."""

import math
import re
import unicodedata
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def slugify(title: str) -> str:
    """Lowercase ASCII words joined by hyphens, suitable for a URL path segment."""
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", normalized.lower())
    return "-".join(words)


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yields lists of at most `size` items, in order."""
    if size < 1:
        raise ValueError("size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


_DURATION_PART = re.compile(r"(\d+)([hms])")
_SECONDS_PER_UNIT = {"h": 3600, "m": 60, "s": 1}


def parse_duration(text: str) -> int:
    """'1h30m' -> 5400. Units must appear at most once, largest first."""
    parts = _DURATION_PART.findall(text.strip().lower())
    if not parts or "".join(number + unit for number, unit in parts) != text.strip().lower():
        raise ValueError(f"not a duration: {text!r}")
    return sum(int(number) * _SECONDS_PER_UNIT[unit] for number, unit in parts)


def human_bytes(size: int) -> str:
    """1536 -> '1.5 KB'. Uses powers of 1024, like most file managers."""
    if size < 1024:
        return f"{size} B"
    exponent = min(int(math.log(size, 1024)), 4)
    value = size / 1024**exponent
    return f"{value:.1f} {['B', 'KB', 'MB', 'GB', 'TB'][exponent]}"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges nested dicts; values from `override` win."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def levenshtein(left: str, right: str) -> int:
    """Minimum number of single-character insertions, deletions or substitutions."""
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + cost))
        previous = current
    return previous[-1]


def luhn_valid(number: str) -> bool:
    """Checksum used by payment card numbers."""
    digits = [int(char) for char in number if char.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def read_env_file(path: str) -> dict[str, str]:
    """Parses KEY=VALUE lines, ignoring blanks and # comments; strips matching quotes."""
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation between closest ranks, like numpy's default."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
