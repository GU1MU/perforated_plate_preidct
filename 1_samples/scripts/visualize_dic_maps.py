from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree


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

ABAQUS_COLORS = [
    "#0000A8",
    "#0039FF",
    "#008DFF",
    "#00D7FF",
    "#00FFB0",
    "#00FF3B",
    "#66FF00",
    "#C7FF00",
    "#FFF000",
    "#FFB000",
    "#FF7200",
    "#FF3000",
    "#C80000",
]

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

FIELD_ORDER = {
    "dx_displacement": 0,
    "dy_displacement": 1,
    "exx_strain": 2,
    "eyy_strain": 3,
    "exy_strain": 4,
    "max_principal_strain": 5,
}


@dataclass(frozen=True)
class FieldInfo:
    slug: str
    label: str
    unit: str
    palette_root: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug or "field"


def field_info(field_name: str, path: Path) -> FieldInfo:
    key = f"{field_name} {path.name}".lower()
    if "dx" in key:
        return FieldInfo("dx_displacement", "DX displacement", "mm", "blue")
    if "dy" in key:
        return FieldInfo("dy_displacement", "DY displacement", "mm", "olive")
    if "exx" in key:
        return FieldInfo("exx_strain", "Lagrangian Exx strain", "strain", "orange")
    if "eyy" in key:
        return FieldInfo("eyy_strain", "Lagrangian Eyy strain", "strain", "olive")
    if "exy" in key:
        return FieldInfo("exy_strain", "Lagrangian Exy strain", "strain", "pink")
    return FieldInfo(slugify(field_name), field_name, "value", "blue")


def field_sort_key(path: Path) -> tuple[int, str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8")
    info = field_info(header.columns[-1], path)
    return FIELD_ORDER.get(info.slug, 99), path.name.lower()


def is_dic_cloud_csv(path: Path) -> bool:
    if "Analysis" not in path.parts:
        return False
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8")
    except Exception:
        return False
    if len(header.columns) < 5:
        return False
    field_name = str(header.columns[-1])
    identity = f"{path.name} {field_name}"
    return "3D" in path.name or any(token in identity for token in ("DX", "DY", "Exx"))


def find_dic_cloud_csvs(sample_dir: Path) -> list[Path]:
    analysis_dir = sample_dir / "Analysis"
    if not analysis_dir.exists():
        return []
    files = [path for path in analysis_dir.rglob("*.csv") if is_dic_cloud_csv(path)]
    return sorted(files, key=field_sort_key)


def load_cloud_csv(path: Path) -> tuple[pd.DataFrame, FieldInfo]:
    raw = pd.read_csv(path, encoding="utf-8")
    if len(raw.columns) < 5:
        raise ValueError(f"{path} does not look like a DIC cloud-map CSV.")

    x_col, y_col, value_col = raw.columns[1], raw.columns[2], raw.columns[-1]
    data = pd.DataFrame(
        {
            "index": pd.to_numeric(raw[raw.columns[0]], errors="coerce"),
            "x": pd.to_numeric(raw[x_col], errors="coerce"),
            "y": pd.to_numeric(raw[y_col], errors="coerce"),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError(f"{path} has no finite DIC values to plot.")
    return data, field_info(str(value_col), path)


def clean_boundary_outliers(
    data: pd.DataFrame,
    *,
    boundary_fraction: float = 0.08,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"x", "y", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for boundary outlier cleaning: {sorted(missing)}")

    x_span = float(data["x"].max() - data["x"].min())
    y_span = float(data["y"].max() - data["y"].min())
    x_band = x_span * boundary_fraction
    y_band = y_span * boundary_fraction
    boundary = (
        (data["x"] >= data["x"].max() - x_band)
        | (data["y"] >= data["y"].max() - y_band)
        | (data["y"] <= data["y"].min() + y_band)
    )

    reference = data.loc[~boundary, "value"]
    if len(reference) < 4:
        reference = data["value"]
    q1 = float(reference.quantile(0.25))
    q3 = float(reference.quantile(0.75))
    iqr = q3 - q1
    if math.isclose(iqr, 0.0):
        center = float(reference.median())
        spread = float((reference - center).abs().median())
        iqr = spread * 1.349 if not math.isclose(spread, 0.0) else 1e-12

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    remove = boundary & ((data["value"] < lower) | (data["value"] > upper))
    return data.loc[~remove].copy(), data.loc[remove].copy()


def impute_boundary_outliers(
    data: pd.DataFrame,
    *,
    boundary_fraction: float = 0.08,
    iqr_multiplier: float = 1.5,
    neighbor_count: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned, outliers = clean_boundary_outliers(
        data,
        boundary_fraction=boundary_fraction,
        iqr_multiplier=iqr_multiplier,
    )
    imputed = data.copy()
    if outliers.empty:
        replacements = outliers.copy()
        replacements["original_value"] = pd.Series(dtype=float)
        replacements["imputed_value"] = pd.Series(dtype=float)
        return imputed, replacements

    valid = cleaned[["x", "y", "value"]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise ValueError("Cannot impute boundary outliers without finite reference values.")

    valid_points = valid[["x", "y"]].to_numpy(dtype=float)
    valid_values = valid["value"].to_numpy(dtype=float)
    outlier_points = outliers[["x", "y"]].to_numpy(dtype=float)
    k = min(max(int(neighbor_count), 1), len(valid))
    tree = cKDTree(valid_points)
    _, neighbor_indices = tree.query(outlier_points, k=k)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    imputed_values = valid_values[neighbor_indices].mean(axis=1)

    replacements = outliers.copy()
    replacements["original_value"] = replacements["value"].to_numpy(dtype=float)
    replacements["imputed_value"] = imputed_values
    for row_index, replacement_value in zip(replacements.index, imputed_values):
        imputed.loc[row_index, "value"] = replacement_value
    return imputed, replacements


def impute_strain_component_boundary_outliers(
    strain: pd.DataFrame,
    component: str,
    *,
    boundary_fraction: float = 0.08,
    iqr_multiplier: float = 6.0,
    neighbor_count: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"index", "x", "y", component}
    missing = required - set(strain.columns)
    if missing:
        raise ValueError(f"Missing columns for strain outlier imputation: {sorted(missing)}")

    component_field = strain[["index", "x", "y", component]].rename(columns={component: "value"})
    imputed_field, replacements = impute_boundary_outliers(
        component_field,
        boundary_fraction=boundary_fraction,
        iqr_multiplier=iqr_multiplier,
        neighbor_count=neighbor_count,
    )
    imputed = strain.copy()
    imputed.loc[imputed_field.index, component] = imputed_field["value"]
    replacements = replacements.rename(columns={"value": component})
    replacements.insert(0, "component", component)
    return imputed, replacements


def point_size(row_count: int) -> float:
    if row_count < 1_000:
        return 20.0
    if row_count < 5_000:
        return 10.0
    if row_count < 20_000:
        return 5.0
    return 3.0


def build_colormap(values: pd.Series, root: str) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    finite = values[np.isfinite(values)]
    vmin = float(finite.min())
    vmax = float(finite.max())
    if math.isclose(vmin, vmax):
        pad = 1.0 if math.isclose(vmin, 0.0) else abs(vmin) * 0.05
        vmin -= pad
        vmax += pad

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "abaqus_saturated_continuous",
        ABAQUS_COLORS,
        N=256,
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap, norm


def draw_continuous_cloud_field(
    ax: plt.Axes,
    data: pd.DataFrame,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
):
    required = {"x", "y", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for cloud-map rendering: {sorted(missing)}")

    field = data[["x", "y", "value"]].replace([np.inf, -np.inf], np.nan).dropna()
    field = field.groupby(["x", "y"], as_index=False, sort=False)["value"].mean()
    if len(field) < 3:
        scatter = ax.scatter(
            field["x"],
            field["y"],
            c=field["value"],
            cmap=cmap,
            norm=norm,
            s=point_size(len(field)),
            marker="s",
            linewidths=0.0,
            edgecolors="none",
        )
        scatter.set_rasterized(True)
        return scatter

    points = field[["x", "y"]].to_numpy(dtype=float)
    values = field["value"].to_numpy(dtype=float)
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        scatter = ax.scatter(
            field["x"],
            field["y"],
            c=field["value"],
            cmap=cmap,
            norm=norm,
            s=point_size(len(field)),
            marker="s",
            linewidths=0.0,
            edgecolors="none",
        )
        scatter.set_rasterized(True)
        return scatter

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    spacing = median_neighbor_spacing(points)
    triangles = triangulation.triangles
    if len(triangles) and np.isfinite(spacing) and spacing > 0:
        tri_points = points[triangles]
        edge_lengths = np.column_stack(
            [
                np.linalg.norm(tri_points[:, 0] - tri_points[:, 1], axis=1),
                np.linalg.norm(tri_points[:, 1] - tri_points[:, 2], axis=1),
                np.linalg.norm(tri_points[:, 2] - tri_points[:, 0], axis=1),
            ]
        )
        triangulation.set_mask(edge_lengths.max(axis=1) > spacing * 4.0)

    mesh = ax.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        linewidths=0.0,
        edgecolors="none",
        antialiased=False,
    )
    mesh.set_rasterized(True)
    return mesh


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


def render_field_map(
    data: pd.DataFrame,
    info: FieldInfo,
    sample_name: str,
    output_dir: Path,
    *,
    subtitle_extra: str | None = None,
) -> list[Path]:
    cmap, norm = build_colormap(data["value"], info.palette_root)

    sns.set_theme(
        style="white",
        font=FONT_FAMILY[0],
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "font.family": FONT_FAMILY,
        },
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.subplots_adjust(top=0.96, right=0.84, left=0.14, bottom=0.15)

    draw_continuous_cloud_field(ax, data, cmap, norm)

    format_axes(ax)
    ax.set_aspect("equal", adjustable="box")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar_ticks = np.linspace(float(norm.vmin), float(norm.vmax), 7)
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.07,
        pad=0.035,
        ticks=colorbar_ticks,
    )
    colorbar.ax.minorticks_on()
    colorbar.ax.tick_params(
        which="major",
        labelsize=14,
        colors=TOKENS["ink"],
        width=2.0,
        length=8,
        direction="inout",
    )
    colorbar.ax.tick_params(
        which="minor",
        colors=TOKENS["ink"],
        width=1.4,
        length=4,
        direction="inout",
    )
    colorbar.outline.set_edgecolor(TOKENS["ink"])
    colorbar.outline.set_linewidth(2.0)
    colorbar.set_label(
        f"{info.label} ({info.unit})",
        color=TOKENS["ink"],
        fontsize=16,
        fontweight="bold",
        labelpad=14,
    )
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(MONO_FONT_FAMILY)
        label.set_fontweight("bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{info.slug}.png",
        output_dir / f"{info.slug}.svg",
    ]
    for output_path in outputs:
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outputs


def render_cloud_map(csv_path: Path, sample_name: str, output_dir: Path) -> list[Path]:
    data, info = load_cloud_csv(csv_path)
    return render_field_map(data, info, sample_name, output_dir)


def load_displacement_fields(sample_dir: Path) -> pd.DataFrame:
    csvs = find_dic_cloud_csvs(sample_dir)
    dx_path = next((path for path in csvs if field_sort_key(path)[0] == FIELD_ORDER["dx_displacement"]), None)
    dy_path = next((path for path in csvs if field_sort_key(path)[0] == FIELD_ORDER["dy_displacement"]), None)
    if dx_path is None or dy_path is None:
        raise FileNotFoundError(f"{sample_dir} needs both DX and DY DIC cloud-map CSV files.")

    dx_data, _ = load_cloud_csv(dx_path)
    dy_data, _ = load_cloud_csv(dy_path)
    dx_data = dx_data.rename(columns={"value": "u"})
    dy_data = dy_data.rename(columns={"value": "v"})
    merged = dx_data.merge(dy_data[["index", "x", "y", "v"]], on=["index", "x", "y"], how="inner")
    return merged.dropna(subset=["x", "y", "u", "v"]).reset_index(drop=True)


def median_neighbor_spacing(points: np.ndarray) -> float:
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        return 1.0
    return float(np.median(nearest))


def compute_small_strain_fields(
    displacement: pd.DataFrame,
    *,
    neighbor_count: int = 24,
    min_neighbors: int = 10,
    max_radius: float | None = None,
) -> pd.DataFrame:
    required = {"x", "y", "u", "v"}
    missing = required - set(displacement.columns)
    if missing:
        raise ValueError(f"Missing columns for strain calculation: {sorted(missing)}")
    if len(displacement) < min_neighbors:
        raise ValueError("Not enough displacement points for local strain calculation.")

    data = displacement.reset_index(drop=True).copy()
    points = data[["x", "y"]].to_numpy(dtype=float)
    values_u = data["u"].to_numpy(dtype=float)
    values_v = data["v"].to_numpy(dtype=float)
    tree = cKDTree(points)
    k = min(max(neighbor_count, min_neighbors), len(data))
    distances, indices = tree.query(points, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]
        distances = distances[:, None]

    gradients = np.full((len(data), 4), np.nan, dtype=float)
    for row_idx, (neighbor_idx, neighbor_dist) in enumerate(zip(indices, distances)):
        finite = np.isfinite(neighbor_dist)
        if max_radius is not None:
            finite &= neighbor_dist <= max_radius
        neighbor_idx = neighbor_idx[finite]
        if len(neighbor_idx) < min_neighbors:
            continue

        centered = points[neighbor_idx] - points[row_idx]
        design = np.column_stack([centered[:, 0], centered[:, 1], np.ones(len(neighbor_idx))])
        du_dx, du_dy, _ = np.linalg.lstsq(design, values_u[neighbor_idx], rcond=None)[0]
        dv_dx, dv_dy, _ = np.linalg.lstsq(design, values_v[neighbor_idx], rcond=None)[0]
        gradients[row_idx] = [du_dx, du_dy, dv_dx, dv_dy]

    data[["du_dx", "du_dy", "dv_dx", "dv_dy"]] = gradients
    data["exx_from_displacement"] = data["du_dx"]
    data["eyy"] = data["dv_dy"]
    data["exy"] = 0.5 * (data["du_dy"] + data["dv_dx"])
    return data.dropna(subset=["eyy", "exy"]).reset_index(drop=True)


def compute_max_principal_strain(strain: pd.DataFrame) -> pd.Series:
    required = {"exx", "eyy", "exy"}
    missing = required - set(strain.columns)
    if missing:
        raise ValueError(f"Missing columns for principal strain calculation: {sorted(missing)}")

    exx = pd.to_numeric(strain["exx"], errors="coerce")
    eyy = pd.to_numeric(strain["eyy"], errors="coerce")
    exy = pd.to_numeric(strain["exy"], errors="coerce")
    center = 0.5 * (exx + eyy)
    radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    return center + radius


def add_max_principal_strain(strain: pd.DataFrame, exx_field: pd.DataFrame) -> pd.DataFrame:
    required = {"index", "x", "y", "value"}
    missing = required - set(exx_field.columns)
    if missing:
        raise ValueError(f"Missing columns for Exx field merge: {sorted(missing)}")

    exx = exx_field[["index", "x", "y", "value"]].rename(columns={"value": "exx"})
    combined = strain.merge(exx, on=["index", "x", "y"], how="inner")
    combined["max_principal_strain"] = compute_max_principal_strain(combined)
    return combined.dropna(subset=["max_principal_strain"]).reset_index(drop=True)


def write_field_csv(data: pd.DataFrame, output_path: Path, field_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")


def render_derived_strain_maps(strain: pd.DataFrame, sample_name: str, output_dir: Path) -> list[Path]:
    rendered: list[Path] = []
    fields = [
        ("eyy", FieldInfo("eyy_strain", "Lagrangian Eyy strain", "strain", "olive")),
        ("exy", FieldInfo("exy_strain", "Lagrangian Exy strain", "strain", "pink")),
    ]
    if "max_principal_strain" in strain.columns:
        fields.append(
            (
                "max_principal_strain",
                FieldInfo("max_principal_strain", "Lagrangian max principal strain", "strain", "orange"),
            )
        )

    for column, info in fields:
        field_data = strain[["index", "x", "y", column]].rename(columns={column: "value"})
        subtitle = "Computed from DX and DY by small-strain displacement gradients."
        if column == "max_principal_strain":
            subtitle = "Computed from Exx, Eyy, and Exy by the in-plane principal strain equation."
        rendered.extend(
            render_field_map(
                field_data,
                info,
                sample_name,
                output_dir,
                subtitle_extra=subtitle,
            )
        )
    return rendered


def render_sample(sample_dir: Path, output_root: Path, processed_root: Path | None = None) -> list[Path]:
    rendered: list[Path] = []
    sample_output = output_root / sample_dir.name
    csv_paths = find_dic_cloud_csvs(sample_dir)
    exx_for_principal: pd.DataFrame | None = None
    for csv_path in csv_paths:
        data, info = load_cloud_csv(csv_path)
        if sample_dir.name == "longitudinal" and info.slug == "exx_strain":
            data, replacements = impute_boundary_outliers(data)
            exx_for_principal = data.copy()
            if processed_root is not None:
                write_field_csv(data, processed_root / "longitudinal_cleaned_exx.csv", info.label)
                write_field_csv(
                    replacements,
                    processed_root / "longitudinal_replaced_exx_outliers.csv",
                    info.label,
                )
            rendered.extend(
                render_field_map(
                    data,
                    info,
                    sample_dir.name,
                    sample_output,
                    subtitle_extra=f"Replaced {len(replacements):,} boundary outlier values with local neighbor means.",
                )
            )
        else:
            if info.slug == "exx_strain":
                exx_for_principal = data.copy()
            rendered.extend(render_field_map(data, info, sample_dir.name, sample_output))

    if any(field_sort_key(path)[0] == FIELD_ORDER["dx_displacement"] for path in csv_paths) and any(
        field_sort_key(path)[0] == FIELD_ORDER["dy_displacement"] for path in csv_paths
    ):
        displacement = load_displacement_fields(sample_dir)
        spacing = median_neighbor_spacing(displacement[["x", "y"]].to_numpy(dtype=float))
        strain = compute_small_strain_fields(
            displacement,
            neighbor_count=24,
            min_neighbors=10,
            max_radius=spacing * 4.0,
        )
        eyy_replacements = pd.DataFrame()
        if sample_dir.name == "longitudinal":
            strain, eyy_replacements = impute_strain_component_boundary_outliers(
                strain,
                "eyy",
                iqr_multiplier=6.0,
                neighbor_count=12,
            )
        if exx_for_principal is not None:
            strain = add_max_principal_strain(strain, exx_for_principal)
        if processed_root is not None:
            write_field_csv(strain, processed_root / f"{sample_dir.name}_derived_strains.csv", "derived strain")
            if sample_dir.name == "longitudinal":
                write_field_csv(
                    eyy_replacements,
                    processed_root / "longitudinal_replaced_eyy_outliers.csv",
                    "Lagrangian Eyy strain",
                )
        rendered.extend(render_derived_strain_maps(strain, sample_dir.name, sample_output))
    rendered.extend(render_sample_kf_curves(sample_dir, output_root))
    return rendered


def iter_sample_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir() and (path / "Analysis").exists())


def render_all(
    data_root: Path,
    output_root: Path,
    processed_root: Path | None = None,
) -> tuple[dict[str, list[Path]], list[str]]:
    rendered_by_sample: dict[str, list[Path]] = {}
    missing: list[str] = []
    for sample_dir in iter_sample_dirs(data_root):
        rendered = render_sample(sample_dir, output_root, processed_root)
        if rendered:
            rendered_by_sample[sample_dir.name] = rendered
        else:
            missing.append(sample_dir.name)
    return rendered_by_sample, missing


def prepare_kf_curve(raw: pd.DataFrame) -> pd.DataFrame:
    force = (
        pd.to_numeric(raw["垂向Y1力(kN)"], errors="coerce")
        + pd.to_numeric(raw["垂向Y2力(kN)"], errors="coerce")
    ) / 2.0
    displacement = pd.to_numeric(raw["垂向Y2位移(mm)"], errors="coerce")
    curve = pd.DataFrame(
        {
            "index": pd.to_numeric(raw["序号()"], errors="coerce"),
            "time_s": pd.to_numeric(raw["时间(s)"], errors="coerce"),
            "displacement_mm": displacement,
            "force_kN": force,
        }
    ).dropna(subset=["displacement_mm", "force_kN"])
    return curve.reset_index(drop=True)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if math.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r_squared)


def fit_linear_elastic_segment(
    curve: pd.DataFrame,
    *,
    area_mm2: float,
    gauge_length_mm: float,
    min_points: int = 60,
) -> dict[str, float | int]:
    data = curve.dropna(subset=["displacement_mm", "force_kN"]).reset_index(drop=True)
    if len(data) < min_points:
        raise ValueError("Not enough load-displacement points for linear segment fitting.")

    x = data["displacement_mm"].to_numpy(dtype=float)
    y = data["force_kN"].to_numpy(dtype=float)
    x_span_total = float(np.nanmax(x) - np.nanmin(x))
    if math.isclose(x_span_total, 0.0):
        raise ValueError("Displacement range is zero; cannot fit elastic modulus.")

    window_sizes = sorted(
        {
            min_points,
            max(min_points, int(len(data) * 0.15)),
            max(min_points, int(len(data) * 0.20)),
            max(min_points, int(len(data) * 0.25)),
            max(min_points, int(len(data) * 0.30)),
        }
    )
    candidates: list[dict[str, float | int]] = []
    for window_size in window_sizes:
        if window_size >= len(data):
            continue
        step = max(1, window_size // 12)
        for start in range(0, len(data) - window_size + 1, step):
            end = start + window_size
            xw = x[start:end]
            yw = y[start:end]
            x_range = float(np.max(xw) - np.min(xw))
            if x_range < x_span_total * 0.08:
                continue
            slope, intercept, r_squared = linear_fit(xw, yw)
            if slope <= 0:
                continue
            candidates.append(
                {
                    "start_row": start,
                    "end_row": end - 1,
                    "points": window_size,
                    "displacement_start_mm": float(xw[0]),
                    "displacement_end_mm": float(xw[-1]),
                    "force_start_kN": float(yw[0]),
                    "force_end_kN": float(yw[-1]),
                    "stiffness_kN_per_mm": slope,
                    "intercept_kN": intercept,
                    "r_squared": r_squared,
                    "x_range_mm": x_range,
                }
            )
    if not candidates:
        raise ValueError("No positive-slope linear segment candidate found.")

    max_r2 = max(float(candidate["r_squared"]) for candidate in candidates)
    r2_tolerance = 1e-5 if max_r2 > 0.9995 else 0.002
    eligible = [candidate for candidate in candidates if float(candidate["r_squared"]) >= max_r2 - r2_tolerance]
    best = max(
        eligible,
        key=lambda candidate: (
            float(candidate["stiffness_kN_per_mm"]),
            float(candidate["x_range_mm"]),
            int(candidate["points"]),
        ),
    )
    best["elastic_modulus_MPa"] = (
        float(best["stiffness_kN_per_mm"]) * 1000.0 * gauge_length_mm / area_mm2
    )
    best["area_mm2"] = area_mm2
    best["gauge_length_mm"] = gauge_length_mm
    return best


def render_kf_curve_fit(
    curve: pd.DataFrame,
    fit: dict[str, float | int],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    sns.set_theme(
        style="white",
        font=FONT_FAMILY[0],
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "font.family": FONT_FAMILY,
        },
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.subplots_adjust(top=0.96, right=0.96, left=0.14, bottom=0.16)

    ax.plot(
        curve["displacement_mm"],
        curve["force_kN"],
        color="#0039FF",
        linewidth=3.2,
        label="Measured curve",
    )
    start = int(fit["start_row"])
    end = int(fit["end_row"]) + 1
    segment = curve.iloc[start:end]
    x_fit = segment["displacement_mm"].to_numpy(dtype=float)
    y_fit = float(fit["stiffness_kN_per_mm"]) * x_fit + float(fit["intercept_kN"])
    ax.plot(
        x_fit,
        y_fit,
        color="#C80000",
        linewidth=4.2,
        label=f"Linear segment, E={float(fit['elastic_modulus_MPa']):.1f} MPa",
    )
    ax.legend(
        loc="upper left",
        frameon=False,
        prop={"family": FONT_FAMILY[0], "weight": "bold", "size": 14},
    )
    ax.set_xlabel("Displacement (mm)", color=TOKENS["ink"], fontsize=16, fontweight="bold")
    ax.set_ylabel("Load (kN)", color=TOKENS["ink"], fontsize=16, fontweight="bold")
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
        colors=TOKENS["ink"],
        labelsize=14,
        width=2.2,
        length=9,
    )
    ax.tick_params(
        which="minor",
        direction="inout",
        colors=TOKENS["ink"],
        width=1.6,
        length=5,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO_FONT_FAMILY)
        label.set_fontweight("bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}_linear_fit.png", output_dir / f"{stem}_linear_fit.svg"]
    for output_path in outputs:
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outputs


def render_sample_kf_curves(
    sample_dir: Path,
    output_root: Path,
    *,
    area_mm2: float = 300.0,
    gauge_length_mm: float = 200.0,
) -> list[Path]:
    rendered: list[Path] = []
    sample_output = output_root / sample_dir.name
    for path in sorted(sample_dir.glob("KF_curve*.csv")):
        raw = pd.read_csv(path, encoding="utf-8")
        curve = prepare_kf_curve(raw)
        fit = fit_linear_elastic_segment(
            curve,
            area_mm2=area_mm2,
            gauge_length_mm=gauge_length_mm,
            min_points=max(50, int(len(curve) * 0.15)),
        )
        rendered.extend(render_kf_curve_fit(curve, fit, sample_output, path.stem))
    return rendered


def analyze_kf_curves(
    data_root: Path,
    output_root: Path,
    *,
    area_mm2: float = 300.0,
    gauge_length_mm: float = 200.0,
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, float | int | str]] = []
    rendered: list[Path] = []
    kf_output = output_root / "kf_curves"
    for path in sorted(data_root.glob("KF_curve_*.csv")):
        raw = pd.read_csv(path, encoding="utf-8")
        curve = prepare_kf_curve(raw)
        fit = fit_linear_elastic_segment(
            curve,
            area_mm2=area_mm2,
            gauge_length_mm=gauge_length_mm,
            min_points=max(50, int(len(curve) * 0.15)),
        )
        row = {"source_file": path.name, **fit}
        rows.append(row)
        rendered.extend(render_kf_curve_fit(curve, fit, kf_output, path.stem))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        aggregate = {
            "source_file": "cross_validation_mean",
            "elastic_modulus_MPa": float(summary["elastic_modulus_MPa"].mean()),
            "elastic_modulus_std_MPa": float(summary["elastic_modulus_MPa"].std(ddof=1)),
            "elastic_modulus_cv_percent": float(
                summary["elastic_modulus_MPa"].std(ddof=1)
                / summary["elastic_modulus_MPa"].mean()
                * 100.0
            ),
            "area_mm2": area_mm2,
            "gauge_length_mm": gauge_length_mm,
        }
        summary = pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)
    return summary, rendered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize exported DIC cloud-map CSV files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("1_samples") / "data",
        help="Directory containing sample folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("1_samples") / "figures",
        help="Directory where sample figure folders will be created.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("1_samples") / "data" / "processed",
        help="Directory where cleaned and derived DIC data CSV files will be written.",
    )
    parser.add_argument(
        "--modulus-output",
        type=Path,
        default=Path("1_samples") / "data" / "elastic_modulus_summary.csv",
        help="CSV path for the elastic modulus summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rendered_by_sample, missing = render_all(args.data_root, args.output_root, args.processed_root)
    modulus_summary, kf_rendered = analyze_kf_curves(args.data_root, args.output_root)
    if not modulus_summary.empty:
        args.modulus_output.parent.mkdir(parents=True, exist_ok=True)
        modulus_summary.to_csv(args.modulus_output, index=False, encoding="utf-8-sig")
    total_files = sum(len(paths) for paths in rendered_by_sample.values())
    for sample, outputs in rendered_by_sample.items():
        print(f"{sample}: rendered {len(outputs)} files")
    for sample in missing:
        print(f"{sample}: no exported DIC cloud-map CSV files found")
    print(f"kf_curves: rendered {len(kf_rendered)} files")
    print(f"elastic modulus summary: {args.modulus_output}")
    print(f"total: rendered {total_files + len(kf_rendered)} files under {args.output_root}")
    return 0 if total_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
