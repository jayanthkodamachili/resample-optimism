"""Render the committed architecture SVG.

Hand written rather than produced by a diagramming library, for two reasons given in
ADR-004: the diagram has eleven boxes and will not grow, and a committed SVG with intrinsic
dimensions and an explicit light background renders identically on GitHub in both themes,
which a Mermaid fence does not.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"

W, H = 940, 470
INK = "#1b1f23"
MUTED = "#57606a"
LEAK = "#b42318"
GOOD = "#116329"
LINE = "#8c959f"


def box(x, y, w, h, title, subtitle="", stroke=MUTED, fill="#ffffff", bold=False):
    weight = "600" if bold else "400"
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + w / 2}" y="{y + (20 if subtitle else h / 2 + 4)}" '
        f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" '
        f'font-weight="{weight}" fill="{INK}">{title}</text>',
    ]
    if subtitle:
        for i, line in enumerate(subtitle.split("|")):
            out.append(
                f'<text x="{x + w / 2}" y="{y + 38 + i * 14}" text-anchor="middle" '
                f'font-family="Helvetica,Arial,sans-serif" font-size="11" '
                f'fill="{MUTED}">{line}</text>'
            )
    return "".join(out)


def arrow(x1, y1, x2, y2, stroke=LINE, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="1.5" marker-end="url(#a)"{dash}/>'
    )


def label(x, y, text, fill=MUTED, size=11, anchor="middle", weight="400"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{text}</text>'
    )


def render() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="Architecture of resample-optimism">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/>'
        "</marker></defs>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        label(
            W / 2, 26, "One scenario, three protocols, one untouched reference", INK, 14,
            weight="600",
        ),
    ]

    # Left column: generation and the two disjoint partitions.
    parts.append(box(24, 52, 170, 50, "seeded generator", "fixed seed, no network"))
    parts.append(box(24, 126, 170, 50, "stratified split", "disjoint by construction"))
    parts.append(arrow(109, 102, 109, 126))

    parts.append(box(24, 200, 170, 46, "modelling pool", stroke=MUTED))
    parts.append(
        box(24, 286, 170, 58, "honest holdout", "no resampler|ever sees it",
            stroke=GOOD, fill="#f2fbf4", bold=True)
    )
    parts.append(arrow(80, 176, 80, 200))
    parts.append(arrow(140, 176, 140, 286))

    # Middle column: the three protocols, all fed from the same pool.
    rows = [
        (72, "leaked: resample, then cross-validate", "synthetic rows land in both folds",
         LEAK, "#fff5f4"),
        (150, "correct: resample inside each fold", "validation portion stays raw",
         GOOD, "#f2fbf4"),
        (228, "control: no resampling at all", "raw imbalanced pool", MUTED, "#ffffff"),
    ]
    for y, title, sub, stroke, fill in rows:
        parts.append(box(250, y, 300, 58, title, sub, stroke=stroke, fill=fill))
        parts.append(arrow(194, 223, 250, y + 29))

    # Right column: one estimate chip per protocol, then the shared honest score.
    for y, *_ in rows:
        parts.append(box(600, y + 10, 150, 38, "cv estimate", stroke=MUTED))
        parts.append(arrow(550, y + 29, 600, y + 29))

    parts.append(
        box(600, 300, 150, 44, "holdout score", stroke=GOOD, fill="#f2fbf4")
    )
    parts.append(arrow(194, 315, 600, 322, dashed=True))
    # Dashed lines leave each protocol at its right edge so they pass outside the boxes
    # below rather than cutting through them.
    for y, *_ in rows:
        parts.append(arrow(550, y + 50, 598, 306, dashed=True))

    # Bottom: the subtraction the whole repository is about.
    parts.append(
        box(250, 376, 620, 72, "optimism = cv estimate - honest holdout score",
            "leaked and correct ship a byte-identical model, so the holdout column cannot "
            "tell them apart|the fix moves the estimate, never the model", bold=True)
    )
    parts.append(arrow(640, 344, 620, 376))
    parts.append(arrow(760, 120, 836, 162))
    parts.append(arrow(836, 162, 836, 376))
    parts.append(label(846, 250, "all three", MUTED, 10, anchor="start"))
    parts.append(label(846, 262, "estimates", MUTED, 10, anchor="start"))

    parts.append(label(222, 207, "same", MUTED, 10))
    parts.append(label(222, 219, "pool", MUTED, 10))
    parts.append(label(400, 368, "models scored on the untouched holdout", MUTED, 10))

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
