"""Shared plumbing for the four experiments.

Every experiment writes a JSON file into docs/experiments and prints a table. Nothing in a
document is typed by hand, so these two outputs are the only place a published number can
come from.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "experiments"


def write(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    line = "".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(headers, widths, strict=True)))
    out = [line, "-" * sum(widths)]
    for row in rows:
        out.append(
            "".join(
                c.ljust(w) if i == 0 else c.rjust(w)
                for i, (c, w) in enumerate(zip(row, widths, strict=True))
            )
        )
    return "\n".join(out)
