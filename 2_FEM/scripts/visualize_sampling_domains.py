from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


FONT_FAMILY = ["DejaVu Sans"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "neutral_light": "#E2E5EA",
}

COLOR_FAMILIES = {
    "blue": {
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "orange": {
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
    "pink": {
        "xlight": "#FCDAD6",
        "light": "#F5BACC",
        "base": "#F390CA",
        "mid": "#BD569B",
        "dark": "#8A3A6F",
    },
}

DEFAULT_GROUP_IDS = (1, 5, 9)
SCRIPT_DIR = Path(__file__).resolve().parent
FEM_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST_PATH = FEM_ROOT / "temp" / "inp" / "group_manifest.json"
DEFAULT_GENERATOR_PATH = SCRIPT_DIR / "generate_perforated_plate_inp.py"
DEFAULT_OUTPUT_DIR = FEM_ROOT / "figures"


@dataclass(frozen=True)
class SamplingDomain:
    group_id: int
    cluster: str
    direction: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    @property
    def x_min(self) -> float:
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        return self.x_range[1]

    @property
    def y_min(self) -> float:
        return self.y_range[0]

    @property
    def y_max(self) -> float:
        return self.y_range[1]

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_generator_manifest(generator_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("generate_perforated_plate_inp", generator_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load generator script: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_manifest"):
        raise ValueError(f"generator script has no build_manifest function: {generator_path}")
    return module.build_manifest([])


def _coerce_range(entry: dict, field_name: str, group_id: int) -> tuple[float, float]:
    values = entry.get(field_name)
    if not isinstance(values, list) and not isinstance(values, tuple):
        raise ValueError(f"group {group_id} is missing {field_name}")
    if len(values) != 2:
        raise ValueError(f"group {group_id} {field_name} must have two bounds")
    lower = float(values[0])
    upper = float(values[1])
    if lower >= upper:
        raise ValueError(f"group {group_id} {field_name} lower bound must be below upper bound")
    return lower, upper


def _entry_group_id(entry: dict) -> int:
    group_id = entry.get("group_id", entry.get("id"))
    if group_id is None:
        raise ValueError("sampling domain entry is missing group_id")
    return int(group_id)


def _domain_from_entry(entry: dict) -> SamplingDomain:
    group_id = _entry_group_id(entry)
    return SamplingDomain(
        group_id=group_id,
        cluster=str(entry.get("cluster", "")),
        direction=str(entry.get("direction", "")),
        x_range=_coerce_range(entry, "x_range", group_id),
        y_range=_coerce_range(entry, "y_range", group_id),
    )


def load_sampling_domains(manifest: dict, group_ids: Sequence[int]) -> list[SamplingDomain]:
    raw_domains = manifest.get("sampling_domains") or manifest.get("groups")
    if not raw_domains:
        raise ValueError("manifest has no sampling_domains or groups entries")

    domains_by_id = {}
    for entry in raw_domains:
        domain = _domain_from_entry(entry)
        domains_by_id[domain.group_id] = domain

    missing = [group_id for group_id in group_ids if group_id not in domains_by_id]
    if missing:
        raise ValueError(f"manifest has no sampling domain for group(s): {missing}")
    return [domains_by_id[group_id] for group_id in group_ids]


def _plate_size(manifest: dict) -> tuple[float, float]:
    plate = manifest.get("plate", {})
    return float(plate.get("x", 150.0)), float(plate.get("y", 200.0))


def _min_center_to_edge(manifest: dict) -> float | None:
    constraints = manifest.get("constraints", {})
    value = constraints.get("min_center_to_edge")
    if value is None:
        return None
    return float(value)


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
    ax.set_xlabel("X coordinate (mm)", color=TOKENS["ink"], fontsize=16, fontweight="bold")
    ax.set_ylabel("Y coordinate (mm)", color=TOKENS["ink"], fontsize=16, fontweight="bold")


def _palette_for_group(group_id: int) -> dict[str, str]:
    ordered_palettes = (
        COLOR_FAMILIES["blue"],
        COLOR_FAMILIES["orange"],
        COLOR_FAMILIES["olive"],
        COLOR_FAMILIES["pink"],
    )
    explicit = {
        1: COLOR_FAMILIES["blue"],
        5: COLOR_FAMILIES["orange"],
        9: COLOR_FAMILIES["olive"],
    }
    return explicit.get(group_id, ordered_palettes[(group_id - 1) % len(ordered_palettes)])


def _draw_plate(ax: plt.Axes, plate_x: float, plate_y: float) -> None:
    plate = Rectangle(
        (0.0, 0.0),
        plate_x,
        plate_y,
        facecolor="#F7F8FA",
        edgecolor=TOKENS["ink"],
        linewidth=2.8,
        zorder=1,
    )
    ax.add_patch(plate)


def _draw_edge_limit(ax: plt.Axes, plate_x: float, plate_y: float, margin: float | None) -> None:
    if margin is None or margin <= 0:
        return
    limit = Rectangle(
        (margin, margin),
        plate_x - 2.0 * margin,
        plate_y - 2.0 * margin,
        facecolor="none",
        edgecolor=TOKENS["muted"],
        linewidth=1.8,
        linestyle=(0, (5, 5)),
        zorder=2,
    )
    ax.add_patch(limit)


def _label_anchor(domain: SamplingDomain) -> tuple[float, float, str, str]:
    if domain.group_id == 1:
        return domain.x_min + 5.0, domain.y_max - 7.0, "left", "top"
    if domain.group_id == 5:
        return domain.x_max - 5.0, domain.y_min + 7.0, "right", "bottom"
    if domain.group_id == 9:
        return (domain.x_min + domain.x_max) / 2.0, domain.y_min + 7.0, "center", "bottom"
    return (domain.x_min + domain.x_max) / 2.0, (domain.y_min + domain.y_max) / 2.0, "center", "center"


def _draw_domain(ax: plt.Axes, domain: SamplingDomain, *, label: bool = True) -> None:
    palette = _palette_for_group(domain.group_id)
    patch = Rectangle(
        (domain.x_min, domain.y_min),
        domain.width,
        domain.height,
        facecolor=palette["base"],
        edgecolor=palette["dark"],
        linewidth=3.0,
        alpha=0.42,
        zorder=3,
    )
    ax.add_patch(patch)
    if not label:
        return

    label_x, label_y, horizontal_alignment, vertical_alignment = _label_anchor(domain)
    ax.text(
        label_x,
        label_y,
        f"Group {domain.group_id}",
        color=palette["dark"],
        fontsize=14,
        fontweight="bold",
        ha=horizontal_alignment,
        va=vertical_alignment,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": TOKENS["surface"],
            "edgecolor": palette["light"],
            "linewidth": 1.2,
            "alpha": 0.88,
        },
        zorder=4,
    )


def _configure_plate_axes(ax: plt.Axes, plate_x: float, plate_y: float) -> None:
    format_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    pad = max(plate_x, plate_y) * 0.045
    ax.set_xlim(-pad, plate_x + pad)
    ax.set_ylim(-pad, plate_y + pad)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_yticks([0, 25, 50, 75, 100, 125, 150, 175, 200])


def _figure() -> tuple[plt.Figure, plt.Axes]:
    _apply_theme()
    fig, ax = plt.subplots(figsize=(6.6, 8.0), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.subplots_adjust(top=0.96, right=0.96, left=0.18, bottom=0.12)
    return fig, ax


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for output_format in formats:
        suffix = output_format.lstrip(".").lower()
        output_path = output_dir / f"{stem}.{suffix}"
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def render_combined_domains(
    manifest: dict,
    domains: Sequence[SamplingDomain],
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    plate_x, plate_y = _plate_size(manifest)
    fig, ax = _figure()
    _draw_plate(ax, plate_x, plate_y)
    _draw_edge_limit(ax, plate_x, plate_y, _min_center_to_edge(manifest))
    for domain in domains:
        _draw_domain(ax, domain)
    _configure_plate_axes(ax, plate_x, plate_y)
    stem = "sampling_domains_groups_" + "_".join(str(domain.group_id) for domain in domains)
    return _save_figure(fig, output_dir, stem, formats)


def render_individual_domain(
    manifest: dict,
    domain: SamplingDomain,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    plate_x, plate_y = _plate_size(manifest)
    fig, ax = _figure()
    _draw_plate(ax, plate_x, plate_y)
    _draw_edge_limit(ax, plate_x, plate_y, _min_center_to_edge(manifest))
    _draw_domain(ax, domain)
    _configure_plate_axes(ax, plate_x, plate_y)
    return _save_figure(fig, output_dir, f"sampling_domain_group_{domain.group_id}", formats)


def render_sampling_domain_figures(
    manifest: dict,
    domains: Sequence[SamplingDomain],
    output_dir: Path,
    formats: Iterable[str] = ("png", "svg"),
) -> list[Path]:
    outputs = []
    outputs.extend(render_combined_domains(manifest, domains, output_dir, formats))
    for domain in domains:
        outputs.extend(render_individual_domain(manifest, domain, output_dir, formats))
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize selected FEM perforated-plate sampling domains.",
    )
    parser.add_argument(
        "--source",
        choices=("generator", "manifest"),
        default="generator",
        help="Read sampling domains from the generator script or from group_manifest.json.",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=DEFAULT_GENERATOR_PATH,
        help="Path to generate_perforated_plate_inp.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to group_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUP_IDS),
        help="Group ids to visualize.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png",],
        help="Output formats, for example: png svg.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source == "generator":
        manifest = load_generator_manifest(args.generator)
    else:
        manifest = load_manifest(args.manifest)
    domains = load_sampling_domains(manifest, args.groups)
    outputs = render_sampling_domain_figures(manifest, domains, args.output_dir, args.formats)
    for output_path in outputs:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
