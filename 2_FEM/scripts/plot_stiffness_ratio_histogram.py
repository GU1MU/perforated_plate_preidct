from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns


FONT_FAMILY = ["DejaVu Sans"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "orange": {
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_ROOT = SCRIPT_DIR.parent
INPUT_DIR = FEM_ROOT / "results" / "stiffness"
OUTPUT_DIR = FEM_ROOT / "figures"
SOLID_STEM = "solid_1_plate"
FIGURE_STEM = "stiffness_ratio_by_specimen"
TABLE_NAME = "stiffness_ratio_table.csv"
FORMATS = ("png", "svg")
PLATE_PATTERN = re.compile(r"^(?P<group_id>\d+)_(?P<instance_index>\d+)_plate$")


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _to_int(value: str | float | int | None) -> int:
    if value is None or str(value).strip() == "":
        return -1
    return int(float(value))


def load_stiffness_rows(input_dir: Path = INPUT_DIR) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("*_stiffness.csv")):
        stem = path.name[: -len("_stiffness.csv")]
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                stiffness = _to_float(row.get("equivalent_stiffness"))
                if stiffness is None:
                    continue
                rows.append({
                    "plate": stem,
                    "path": path,
                    "status": row.get("status", ""),
                    "frame_index": _to_int(row.get("frame_index")),
                    "equivalent_stiffness": stiffness,
                })
    return rows


def _last_ok_by_plate(rows: list[dict]) -> dict[str, dict]:
    selected = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        plate = row["plate"]
        if plate not in selected or row["frame_index"] >= selected[plate]["frame_index"]:
            selected[plate] = row
    return selected


def parse_perforated_plate_stem(stem: str) -> tuple[int, int] | None:
    match = PLATE_PATTERN.match(stem)
    if match is None:
        return None
    return int(match.group("group_id")), int(match.group("instance_index"))


def compute_ratio_records(rows: list[dict], solid_stem: str = SOLID_STEM) -> list[dict]:
    selected = _last_ok_by_plate(rows)
    if solid_stem not in selected:
        raise ValueError(f"solid baseline not found in stiffness rows: {solid_stem}")
    solid_stiffness = float(selected[solid_stem]["equivalent_stiffness"])
    if solid_stiffness == 0.0:
        raise ValueError("solid baseline stiffness is zero")

    ratios = []
    for plate in sorted(selected):
        if plate == solid_stem:
            continue
        parsed = parse_perforated_plate_stem(plate)
        if parsed is None:
            continue
        group_id, instance_index = parsed
        row = selected[plate]
        equivalent_stiffness = float(row["equivalent_stiffness"])
        ratios.append({
            "plate": plate,
            "label": f"Group{group_id}+{instance_index}",
            "group_id": group_id,
            "instance_index": instance_index,
            "frame_index": int(row["frame_index"]),
            "equivalent_stiffness": equivalent_stiffness,
            "solid_stiffness": solid_stiffness,
            "stiffness_ratio": equivalent_stiffness / solid_stiffness,
        })
    if not ratios:
        raise ValueError("no perforated-plate stiffness records found")
    return sorted(ratios, key=lambda row: (row["group_id"], row["instance_index"]))


def _apply_theme() -> None:
    sns.set_theme(
        style="white",
        font=FONT_FAMILY[0],
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "font.family": FONT_FAMILY,
        },
    )


def format_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(False)
    ax.minorticks_on()
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(TOKENS["ink"])
        ax.spines[side].set_linewidth(2.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(
        which="major",
        direction="inout",
        bottom=True,
        left=True,
        colors=TOKENS["ink"],
        labelsize=14,
        width=2.2,
        length=9,
    )
    ax.tick_params(
        which="minor",
        direction="inout",
        bottom=True,
        left=True,
        colors=TOKENS["ink"],
        width=1.6,
        length=5,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO_FONT_FAMILY)
        label.set_fontweight("bold")


def render_bar_chart(ratios: list[dict], output_dir: Path, formats: Iterable[str] = FORMATS) -> list[Path]:
    _apply_theme()
    width = max(7.2, 1.0 + 1.05 * len(ratios))
    fig, ax = plt.subplots(figsize=(width, 5.2), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.subplots_adjust(top=0.94, right=0.98, left=0.14, bottom=0.16)

    labels = [row["label"] for row in ratios]
    values = [row["stiffness_ratio"] for row in ratios]
    positions = list(range(len(ratios)))
    ax.bar(
        positions,
        values,
        width=0.36,
        color=COLOR_FAMILIES["blue"]["base"],
        edgecolor=COLOR_FAMILIES["blue"]["dark"],
        linewidth=2.2,
        alpha=0.82,
    )
    for position, value in zip(positions, values):
        ax.text(
            position,
            value + 0.018,
            f"{value:.3f}",
            color=TOKENS["ink"],
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="bottom",
            fontfamily=MONO_FONT_FAMILY[0],
        )
    format_axes(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("Specimen", color=TOKENS["ink"], fontsize=16, fontweight="bold")
    ax.set_ylabel(r"Equivalent stiffness ratio, $K/K_{\mathrm{solid}}$", color=TOKENS["ink"], fontsize=16, fontweight="bold")
    upper = max(values) * 1.12 if values else 1.0
    ax.set_ylim(0.0, max(1.0, upper))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for output_format in formats:
        suffix = output_format.lstrip(".").lower()
        output_path = output_dir / f"{FIGURE_STEM}.{suffix}"
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def write_ratio_table(ratios: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / TABLE_NAME
    fieldnames = [
        "plate",
        "label",
        "group_id",
        "instance_index",
        "frame_index",
        "equivalent_stiffness",
        "solid_stiffness",
        "stiffness_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ratios:
            writer.writerow(row)
    return output_path


def render_outputs(
    ratios: list[dict],
    output_dir: Path = OUTPUT_DIR,
    formats: Iterable[str] = FORMATS,
) -> list[Path]:
    outputs = render_bar_chart(ratios, output_dir, formats=formats)
    outputs.append(write_ratio_table(ratios, output_dir))
    return outputs


def main() -> int:
    rows = load_stiffness_rows(INPUT_DIR)
    ratios = compute_ratio_records(rows, solid_stem=SOLID_STEM)
    outputs = render_outputs(ratios, OUTPUT_DIR, formats=FORMATS)
    for output_path in outputs:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
