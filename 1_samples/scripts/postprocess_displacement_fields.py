from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AffineResult:
    center: np.ndarray
    translation: np.ndarray
    H: np.ndarray
    inlier_mask: np.ndarray
    residual_norm: np.ndarray
    iterations: int
    method: str


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _design_matrix(x: np.ndarray, y: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x - center[0], y - center[1]])


def _fit_once(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    center: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(inlier_mask.sum()) < 3:
        raise ValueError("At least three finite displacement points are required for affine fitting.")

    design = _design_matrix(x, y, center)
    coeff_ux = np.linalg.lstsq(design[inlier_mask], ux[inlier_mask], rcond=None)[0]
    coeff_uy = np.linalg.lstsq(design[inlier_mask], uy[inlier_mask], rcond=None)[0]
    translation = np.array([coeff_ux[0], coeff_uy[0]], dtype=float)
    H = np.array(
        [
            [coeff_ux[1], coeff_ux[2]],
            [coeff_uy[1], coeff_uy[2]],
        ],
        dtype=float,
    )
    predicted = design @ np.column_stack([coeff_ux, coeff_uy])
    residual = np.column_stack([ux, uy]) - predicted
    residual_norm = np.sqrt(np.sum(residual**2, axis=1))
    return translation, H, predicted[:, 0], predicted[:, 1], residual_norm


def fit_affine_displacement(
    x,
    y,
    ux,
    uy,
    mask=None,
    *,
    center=None,
    robust: bool = True,
    mad_multiplier: float = 6.0,
    max_iterations: int = 4,
    min_inlier_fraction: float = 0.5,
) -> AffineResult:
    """Fit the global affine displacement field.

    The fitted field is:
    u_x = a_x + H_xx * (x - x_c) + H_xy * (y - y_c)
    u_y = a_y + H_yx * (x - x_c) + H_yy * (y - y_c)
    """

    x = _as_float_array(x)
    y = _as_float_array(y)
    ux = _as_float_array(ux)
    uy = _as_float_array(uy)
    if not (len(x) == len(y) == len(ux) == len(uy)):
        raise ValueError("x, y, ux, and uy must have the same length.")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ux) & np.isfinite(uy)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if int(valid.sum()) < 3:
        raise ValueError("At least three finite displacement points are required for affine fitting.")

    fit_center = (
        np.array([float(np.mean(x[valid])), float(np.mean(y[valid]))], dtype=float)
        if center is None
        else np.asarray(center, dtype=float)
    )
    if fit_center.shape != (2,):
        raise ValueError("center must contain exactly two values.")

    inliers = valid.copy()
    iterations = 0
    translation: np.ndarray | None = None
    H: np.ndarray | None = None
    residual_norm = np.full(len(x), np.nan, dtype=float)
    method = "least_squares"

    for iteration in range(max(1, int(max_iterations))):
        iterations = iteration + 1
        translation, H, _, _, residual_norm = _fit_once(x, y, ux, uy, fit_center, inliers)
        if not robust:
            break

        reference = residual_norm[inliers]
        median = float(np.median(reference))
        mad = float(np.median(np.abs(reference - median)))
        if math.isclose(mad, 0.0):
            spread = float(np.std(reference))
            scale = spread if not math.isclose(spread, 0.0) else 1e-12
        else:
            scale = 1.4826 * mad
        threshold = median + mad_multiplier * scale
        next_inliers = valid & (residual_norm <= threshold)
        min_inliers = max(3, int(math.ceil(valid.sum() * min_inlier_fraction)))
        if int(next_inliers.sum()) < min_inliers:
            break
        if np.array_equal(next_inliers, inliers):
            method = "robust_mad"
            break
        inliers = next_inliers
        method = "robust_mad"

    if translation is None or H is None:
        raise RuntimeError("Affine displacement fit did not run.")

    return AffineResult(
        center=fit_center,
        translation=translation,
        H=H,
        inlier_mask=inliers,
        residual_norm=residual_norm,
        iterations=iterations,
        method=method,
    )


def predict_affine_displacement(
    x,
    y,
    affine_result: AffineResult,
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    centered = np.column_stack([x - affine_result.center[0], y - affine_result.center[1]])
    predicted = affine_result.translation + centered @ affine_result.H.T
    return predicted[:, 0], predicted[:, 1]


def compute_average_strain(H) -> dict[str, Any]:
    H = np.asarray(H, dtype=float)
    if H.shape != (2, 2):
        raise ValueError("H must be a 2 by 2 matrix.")

    epsilon = 0.5 * (H + H.T)
    W = 0.5 * (H - H.T)
    gamma_xy = float(H[0, 1] + H[1, 0])
    omega_z = float(0.5 * (H[1, 0] - H[0, 1]))
    principal_values, principal_vectors = np.linalg.eigh(epsilon)
    max_index = int(np.argmax(principal_values))
    max_vector = principal_vectors[:, max_index]
    angle_from_x = math.atan2(float(max_vector[1]), float(max_vector[0]))
    axis_angle_from_x = _fold_axis_angle(angle_from_x)
    angle_to_y = _axis_angle_difference(angle_from_x, math.pi / 2.0)

    return {
        "epsilon": epsilon,
        "W": W,
        "epsilon_xx": float(epsilon[0, 0]),
        "epsilon_yy": float(epsilon[1, 1]),
        "epsilon_xy": float(epsilon[0, 1]),
        "gamma_xy": gamma_xy,
        "omega_z": omega_z,
        "principal_strains": [float(value) for value in principal_values],
        "max_principal_strain": float(principal_values[max_index]),
        "max_principal_angle_from_x_deg": float(math.degrees(axis_angle_from_x)),
        "principal_angle_to_y_deg": float(math.degrees(angle_to_y)),
    }


def compute_local_small_strain_fields(
    displacement: pd.DataFrame,
    ux_column: str,
    uy_column: str,
    *,
    prefix: str,
    neighbor_count: int = 24,
    min_neighbors: int = 10,
    max_radius: float | None = None,
) -> pd.DataFrame:
    required = {"index", "x", "y", ux_column, uy_column}
    missing = required - set(displacement.columns)
    if missing:
        raise ValueError(f"Missing columns for local strain calculation: {sorted(missing)}")
    if len(displacement) < min_neighbors:
        raise ValueError("Not enough displacement points for local strain calculation.")

    data = displacement.reset_index(drop=True).copy()
    points = data[["x", "y"]].to_numpy(dtype=float)
    ux = data[ux_column].to_numpy(dtype=float)
    uy = data[uy_column].to_numpy(dtype=float)
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(ux) & np.isfinite(uy)
    tree = cKDTree(points[finite])
    finite_positions = np.flatnonzero(finite)
    k = min(max(int(neighbor_count), int(min_neighbors)), int(finite.sum()))
    distances, neighbor_positions = tree.query(points, k=k)
    if neighbor_positions.ndim == 1:
        neighbor_positions = neighbor_positions[:, None]
        distances = distances[:, None]

    gradients = np.full((len(data), 4), np.nan, dtype=float)
    neighbor_counts = np.zeros(len(data), dtype=int)
    for row_index, (row_distances, row_neighbor_positions) in enumerate(zip(distances, neighbor_positions)):
        valid_neighbor = np.isfinite(row_distances)
        if max_radius is not None:
            valid_neighbor &= row_distances <= max_radius
        neighbor_indices = finite_positions[row_neighbor_positions[valid_neighbor]]
        if len(neighbor_indices) < min_neighbors:
            continue

        centered = points[neighbor_indices] - points[row_index]
        if np.linalg.matrix_rank(centered) < 2:
            continue
        design = np.column_stack([centered[:, 0], centered[:, 1], np.ones(len(neighbor_indices))])
        du_dx, du_dy, _ = np.linalg.lstsq(design, ux[neighbor_indices], rcond=None)[0]
        dv_dx, dv_dy, _ = np.linalg.lstsq(design, uy[neighbor_indices], rcond=None)[0]
        gradients[row_index] = [du_dx, du_dy, dv_dx, dv_dy]
        neighbor_counts[row_index] = len(neighbor_indices)

    strain = data[["index", "x", "y"]].copy()
    if "z" in data.columns:
        strain["z"] = data["z"]
    strain[f"{prefix}_strain_neighbor_count"] = neighbor_counts
    strain[f"{prefix}_du_dx"] = gradients[:, 0]
    strain[f"{prefix}_du_dy"] = gradients[:, 1]
    strain[f"{prefix}_dv_dx"] = gradients[:, 2]
    strain[f"{prefix}_dv_dy"] = gradients[:, 3]
    strain[f"{prefix}_exx"] = gradients[:, 0]
    strain[f"{prefix}_eyy"] = gradients[:, 3]
    strain[f"{prefix}_exy"] = 0.5 * (gradients[:, 1] + gradients[:, 2])
    strain[f"{prefix}_gamma_xy"] = gradients[:, 1] + gradients[:, 2]

    principal = _principal_strain_from_components(
        strain[f"{prefix}_exx"].to_numpy(dtype=float),
        strain[f"{prefix}_eyy"].to_numpy(dtype=float),
        strain[f"{prefix}_exy"].to_numpy(dtype=float),
    )
    strain[f"{prefix}_min_principal_strain"] = principal["min_principal_strain"]
    strain[f"{prefix}_max_principal_strain"] = principal["max_principal_strain"]
    strain[f"{prefix}_principal_angle_from_x_deg"] = principal["principal_angle_from_x_deg"]
    strain[f"{prefix}_principal_angle_to_y_deg"] = principal["principal_angle_to_y_deg"]
    return strain


def _principal_strain_from_components(
    exx: np.ndarray,
    eyy: np.ndarray,
    exy: np.ndarray,
) -> dict[str, np.ndarray]:
    center = 0.5 * (exx + eyy)
    radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    max_principal = center + radius
    min_principal = center - radius
    angle = 0.5 * np.arctan2(2.0 * exy, exx - eyy)

    use_y_axis = np.abs(exy) < 1e-14
    use_y_axis &= eyy >= exx
    angle = np.where(use_y_axis, math.pi / 2.0, angle)
    angle_from_x = ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0
    angle_to_y = ((angle - math.pi / 2.0 + math.pi / 2.0) % math.pi) - math.pi / 2.0
    invalid = ~(np.isfinite(exx) & np.isfinite(eyy) & np.isfinite(exy))
    angle_from_x = np.where(invalid, np.nan, angle_from_x)
    angle_to_y = np.where(invalid, np.nan, angle_to_y)
    return {
        "min_principal_strain": min_principal,
        "max_principal_strain": max_principal,
        "principal_angle_from_x_deg": np.degrees(angle_from_x),
        "principal_angle_to_y_deg": np.degrees(angle_to_y),
    }


def build_strain_fields(
    processed: pd.DataFrame,
    *,
    neighbor_count: int = 24,
    min_neighbors: int = 10,
    max_radius: float | None = None,
) -> pd.DataFrame:
    if max_radius is None:
        spacing = median_neighbor_spacing(processed[["x", "y"]].to_numpy(dtype=float))
        max_radius = spacing * 4.0

    variants = [
        ("raw", "ux_raw", "uy_raw"),
        ("rigid_removed", "ux_rigid_removed", "uy_rigid_removed"),
        ("shear_removed", "ux_shear_removed", "uy_shear_removed"),
    ]
    strain_parts: list[pd.DataFrame] = []
    key_columns = ["index", "x", "y"] + (["z"] if "z" in processed.columns else [])
    for prefix, ux_column, uy_column in variants:
        part = compute_local_small_strain_fields(
            processed,
            ux_column,
            uy_column,
            prefix=prefix,
            neighbor_count=neighbor_count,
            min_neighbors=min_neighbors,
            max_radius=max_radius,
        )
        value_columns = [column for column in part.columns if column not in key_columns]
        strain_parts.append(part[key_columns + value_columns])

    strain = strain_parts[0]
    for part in strain_parts[1:]:
        strain = strain.merge(part, on=key_columns, how="inner")
    return strain


def median_neighbor_spacing(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 1.0
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        return 1.0
    return float(np.median(nearest))


def _fold_axis_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi / 2.0) % math.pi - math.pi / 2.0


def _axis_angle_difference(angle_a: float, angle_b: float) -> float:
    return (angle_a - angle_b + math.pi / 2.0) % math.pi - math.pi / 2.0


def remove_rigid_motion(
    x,
    y,
    ux,
    uy,
    affine_result: AffineResult,
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    ux = _as_float_array(ux)
    uy = _as_float_array(uy)
    metrics = compute_average_strain(affine_result.H)
    W = metrics["W"]
    centered = np.column_stack([x - affine_result.center[0], y - affine_result.center[1]])
    rigid = affine_result.translation + centered @ W.T
    corrected = np.column_stack([ux, uy]) - rigid
    return corrected[:, 0], corrected[:, 1]


def remove_average_shear(
    x,
    y,
    ux,
    uy,
    gamma_xy: float,
    *,
    center=None,
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    ux = _as_float_array(ux)
    uy = _as_float_array(uy)
    fit_center = (
        np.array([float(np.mean(x)), float(np.mean(y))], dtype=float)
        if center is None
        else np.asarray(center, dtype=float)
    )
    E_shear = np.array([[0.0, gamma_xy / 2.0], [gamma_xy / 2.0, 0.0]], dtype=float)
    centered = np.column_stack([x - fit_center[0], y - fit_center[1]])
    shear = centered @ E_shear.T
    corrected = np.column_stack([ux, uy]) - shear
    return corrected[:, 0], corrected[:, 1]


def rotate_to_principal_frame(
    x,
    y,
    ux,
    uy,
    theta: float,
    *,
    center=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    ux = _as_float_array(ux)
    uy = _as_float_array(uy)
    fit_center = (
        np.array([float(np.mean(x)), float(np.mean(y))], dtype=float)
        if center is None
        else np.asarray(center, dtype=float)
    )
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    rotated_xy = np.column_stack([x - fit_center[0], y - fit_center[1]]) @ R
    rotated_uv = np.column_stack([ux, uy]) @ R
    return rotated_xy[:, 0], rotated_xy[:, 1], rotated_uv[:, 0], rotated_uv[:, 1]


def is_displacement_cloud_csv(path: Path, component: str) -> bool:
    token = f"D{component.upper()}"
    if "Analysis" not in path.parts or path.suffix.lower() != ".csv":
        return False
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8")
    except Exception:
        return False
    if len(header.columns) < 5:
        return False
    identity = f"{path.name} {header.columns[-1]}".upper()
    return token in identity and "3D".upper() in identity


def find_displacement_csv(sample_dir: Path, component: str) -> Path:
    analysis_dir = sample_dir / "Analysis"
    if not analysis_dir.exists():
        raise FileNotFoundError(f"{sample_dir} has no Analysis directory.")
    matches = sorted(path for path in analysis_dir.rglob("*.csv") if is_displacement_cloud_csv(path, component))
    if not matches:
        raise FileNotFoundError(f"{sample_dir} has no D{component.upper()} displacement cloud CSV.")
    return matches[0]


def _load_component(path: Path, value_column: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")
    if len(raw.columns) < 5:
        raise ValueError(f"{path} does not look like a DIC displacement cloud CSV.")
    data = pd.DataFrame(
        {
            "index": pd.to_numeric(raw.iloc[:, 0], errors="coerce"),
            "x": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "y": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "z": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            value_column: pd.to_numeric(raw.iloc[:, -1], errors="coerce"),
        }
    )
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def load_displacement_cloud(sample_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    dx_path = find_displacement_csv(sample_dir, "x")
    dy_path = find_displacement_csv(sample_dir, "y")
    dx = _load_component(dx_path, "ux_raw")
    dy = _load_component(dy_path, "uy_raw")
    merged = dx.merge(dy, on="index", suffixes=("", "_dy"), how="inner")
    if merged.empty:
        raise ValueError(f"{sample_dir} has no matching DX/DY displacement points.")

    for column in ("x", "y", "z"):
        delta = (merged[column] - merged[f"{column}_dy"]).abs().max()
        if pd.notna(delta) and float(delta) > 1e-7:
            raise ValueError(f"{sample_dir} DX/DY coordinates do not match on column {column}.")
    merged = merged[["index", "x", "y", "z", "ux_raw", "uy_raw"]]
    merged = merged.dropna(subset=["x", "y", "ux_raw", "uy_raw"]).reset_index(drop=True)
    return merged, {"dx": str(dx_path), "dy": str(dy_path)}


def build_postprocessed_displacement(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"index", "x", "y", "z", "ux_raw", "uy_raw"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for displacement postprocessing: {sorted(missing)}")

    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    ux = data["ux_raw"].to_numpy(dtype=float)
    uy = data["uy_raw"].to_numpy(dtype=float)
    raw_fit = fit_affine_displacement(x, y, ux, uy)
    raw_metrics = compute_average_strain(raw_fit.H)
    ux_affine, uy_affine = predict_affine_displacement(x, y, raw_fit)
    ux_rigid, uy_rigid = remove_rigid_motion(x, y, ux, uy, raw_fit)
    rigid_fit = fit_affine_displacement(
        x,
        y,
        ux_rigid,
        uy_rigid,
        mask=raw_fit.inlier_mask,
        center=raw_fit.center,
        robust=False,
    )
    ux_shear, uy_shear = remove_average_shear(
        x,
        y,
        ux_rigid,
        uy_rigid,
        raw_metrics["gamma_xy"],
        center=raw_fit.center,
    )
    shear_fit = fit_affine_displacement(
        x,
        y,
        ux_shear,
        uy_shear,
        mask=raw_fit.inlier_mask,
        center=raw_fit.center,
        robust=False,
    )
    ux_shear_affine, uy_shear_affine = predict_affine_displacement(x, y, shear_fit)

    output = data.copy()
    output["fit_inlier"] = raw_fit.inlier_mask
    output["ux_affine_raw"] = ux_affine
    output["uy_affine_raw"] = uy_affine
    output["ux_raw_affine_residual"] = ux - ux_affine
    output["uy_raw_affine_residual"] = uy - uy_affine
    output["raw_affine_residual_norm"] = raw_fit.residual_norm
    output["ux_rigid_removed"] = ux_rigid
    output["uy_rigid_removed"] = uy_rigid
    output["ux_shear_removed"] = ux_shear
    output["uy_shear_removed"] = uy_shear
    output["ux_shear_removed_residual"] = ux_shear - ux_shear_affine
    output["uy_shear_removed_residual"] = uy_shear - uy_shear_affine

    summary = {
        "point_count": int(len(output)),
        "fit_inlier_count": int(raw_fit.inlier_mask.sum()),
        "fit_outlier_count": int((~raw_fit.inlier_mask).sum()),
        "coordinate_bounds": {
            "x_min": float(output["x"].min()),
            "x_max": float(output["x"].max()),
            "y_min": float(output["y"].min()),
            "y_max": float(output["y"].max()),
        },
        "raw_fit": _fit_summary(raw_fit, raw_metrics),
        "rigid_removed_fit": _fit_summary(rigid_fit, compute_average_strain(rigid_fit.H)),
        "shear_removed_fit": _fit_summary(shear_fit, compute_average_strain(shear_fit.H)),
        "residuals": {
            "raw_affine_rms_mm": _rms(output["raw_affine_residual_norm"]),
            "ux_raw_affine_rms_mm": _rms(output["ux_raw_affine_residual"]),
            "uy_raw_affine_rms_mm": _rms(output["uy_raw_affine_residual"]),
            "ux_shear_removed_residual_rms_mm": _rms(output["ux_shear_removed_residual"]),
            "uy_shear_removed_residual_rms_mm": _rms(output["uy_shear_removed_residual"]),
        },
        "notes": [
            "rigid_removed subtracts fitted translation and rigid-body rotation while retaining average strain.",
            "shear_removed additionally subtracts the fitted average engineering shear strain for visualization.",
        ],
    }
    return output, summary


def summarize_strain_fields(strain: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for prefix in ("raw", "rigid_removed", "shear_removed"):
        valid = strain.dropna(subset=[f"{prefix}_exx", f"{prefix}_eyy", f"{prefix}_exy"])
        summary[prefix] = {
            "valid_point_count": int(len(valid)),
            "exx_median": _finite_median(valid.get(f"{prefix}_exx", pd.Series(dtype=float))),
            "eyy_median": _finite_median(valid.get(f"{prefix}_eyy", pd.Series(dtype=float))),
            "exy_median": _finite_median(valid.get(f"{prefix}_exy", pd.Series(dtype=float))),
            "gamma_xy_median": _finite_median(valid.get(f"{prefix}_gamma_xy", pd.Series(dtype=float))),
            "max_principal_strain_median": _finite_median(
                valid.get(f"{prefix}_max_principal_strain", pd.Series(dtype=float))
            ),
            "principal_angle_to_y_deg_median": _finite_median(
                valid.get(f"{prefix}_principal_angle_to_y_deg", pd.Series(dtype=float))
            ),
        }
    return summary


def _finite_median(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.median(values))


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def _fit_summary(result: AffineResult, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "center": [float(value) for value in result.center],
        "translation": [float(value) for value in result.translation],
        "H": _matrix_to_list(result.H),
        "epsilon": _matrix_to_list(metrics["epsilon"]),
        "W": _matrix_to_list(metrics["W"]),
        "epsilon_xx": metrics["epsilon_xx"],
        "epsilon_yy": metrics["epsilon_yy"],
        "epsilon_xy": metrics["epsilon_xy"],
        "gamma_xy": metrics["gamma_xy"],
        "omega_z": metrics["omega_z"],
        "principal_strains": metrics["principal_strains"],
        "max_principal_strain": metrics["max_principal_strain"],
        "max_principal_angle_from_x_deg": metrics["max_principal_angle_from_x_deg"],
        "principal_angle_to_y_deg": metrics["principal_angle_to_y_deg"],
        "method": result.method,
        "iterations": result.iterations,
    }


def _matrix_to_list(matrix) -> list[list[float]]:
    array = np.asarray(matrix, dtype=float)
    return [[float(value) for value in row] for row in array]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_sample(sample_dir: Path, output_root: Path, *, write_plots: bool = True) -> list[Path]:
    data, source_files = load_displacement_cloud(sample_dir)
    processed, summary = build_postprocessed_displacement(data)
    strain = build_strain_fields(processed)
    summary["sample"] = sample_dir.name
    summary["source_files"] = source_files
    summary["strain_fields"] = summarize_strain_fields(strain)

    sample_output = output_root / sample_dir.name
    sample_output.mkdir(parents=True, exist_ok=True)
    processed_path = sample_output / f"{sample_dir.name}_displacement_postprocessed.csv"
    strain_path = sample_output / f"{sample_dir.name}_strain_fields.csv"
    summary_path = sample_output / f"{sample_dir.name}_affine_summary.json"
    outlier_path = sample_output / f"{sample_dir.name}_fit_outliers.csv"

    processed.to_csv(processed_path, index=False, encoding="utf-8-sig")
    strain.to_csv(strain_path, index=False, encoding="utf-8-sig")
    processed.loc[~processed["fit_inlier"]].to_csv(outlier_path, index=False, encoding="utf-8-sig")
    write_json(summary_path, summary)

    outputs = [processed_path, strain_path, summary_path, outlier_path]
    if write_plots:
        outputs.extend(render_processed_maps(processed, sample_dir.name, sample_output / "figures"))
        outputs.extend(render_strain_maps(strain, sample_dir.name, sample_output / "figures" / "strain_fields"))
    return outputs


def render_processed_maps(processed: pd.DataFrame, sample_name: str, output_dir: Path) -> list[Path]:
    from visualize_dic_maps import FieldInfo, render_field_map

    rendered: list[Path] = []
    fields = [
        ("ux_raw", FieldInfo("raw_ux_displacement", "raw UX displacement", "mm", "blue")),
        ("uy_raw", FieldInfo("raw_uy_displacement", "raw UY displacement", "mm", "olive")),
        ("ux_rigid_removed", FieldInfo("rigid_removed_ux", "rigid-removed UX", "mm", "blue")),
        ("uy_rigid_removed", FieldInfo("rigid_removed_uy", "rigid-removed UY", "mm", "olive")),
        ("ux_shear_removed", FieldInfo("shear_removed_ux", "shear-removed UX", "mm", "blue")),
        ("uy_shear_removed", FieldInfo("shear_removed_uy", "shear-removed UY", "mm", "olive")),
        ("ux_raw_affine_residual", FieldInfo("raw_affine_residual_ux", "raw affine residual UX", "mm", "pink")),
        ("uy_raw_affine_residual", FieldInfo("raw_affine_residual_uy", "raw affine residual UY", "mm", "pink")),
    ]
    for column, info in fields:
        field = processed[["index", "x", "y", column]].rename(columns={column: "value"})
        rendered.extend(render_field_map(field, info, sample_name, output_dir))
    return rendered


def render_strain_maps(strain: pd.DataFrame, sample_name: str, output_dir: Path) -> list[Path]:
    from visualize_dic_maps import FieldInfo, render_field_map

    rendered: list[Path] = []
    variants = [
        ("raw", "raw"),
        ("rigid_removed", "rigid-removed"),
        ("shear_removed", "shear-removed"),
    ]
    components = [
        ("exx", "Exx strain", "orange"),
        ("eyy", "Eyy strain", "olive"),
        ("exy", "Exy strain", "pink"),
        ("gamma_xy", "engineering shear strain", "pink"),
        ("max_principal_strain", "max principal strain", "orange"),
    ]
    for prefix, label_prefix in variants:
        for component, label, palette_root in components:
            column = f"{prefix}_{component}"
            if column not in strain.columns:
                continue
            field = strain[["index", "x", "y", column]].rename(columns={column: "value"})
            slug = f"{prefix}_{component}" if component.endswith("strain") else f"{prefix}_{component}_strain"
            info = FieldInfo(
                slug,
                f"{label_prefix} {label}",
                "strain",
                palette_root,
            )
            rendered.extend(render_field_map(field, info, sample_name, output_dir))
    return rendered


def process_all(
    data_root: Path,
    output_root: Path,
    *,
    samples: list[str],
    write_plots: bool = True,
) -> tuple[dict[str, list[Path]], Path]:
    outputs_by_sample: dict[str, list[Path]] = {}
    summary_rows: list[dict[str, Any]] = []
    for sample_name in samples:
        sample_dir = data_root / sample_name
        outputs = process_sample(sample_dir, output_root, write_plots=write_plots)
        outputs_by_sample[sample_name] = outputs
        summary_path = output_root / sample_name / f"{sample_name}_affine_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw = summary["raw_fit"]
        rigid = summary["rigid_removed_fit"]
        shear = summary["shear_removed_fit"]
        summary_rows.append(
            {
                "sample": sample_name,
                "point_count": summary["point_count"],
                "fit_inlier_count": summary["fit_inlier_count"],
                "fit_outlier_count": summary["fit_outlier_count"],
                "raw_H_xx": raw["H"][0][0],
                "raw_H_xy": raw["H"][0][1],
                "raw_H_yx": raw["H"][1][0],
                "raw_H_yy": raw["H"][1][1],
                "raw_gamma_xy": raw["gamma_xy"],
                "raw_omega_z": raw["omega_z"],
                "raw_principal_angle_to_y_deg": raw["principal_angle_to_y_deg"],
                "rigid_gamma_xy": rigid["gamma_xy"],
                "shear_gamma_xy": shear["gamma_xy"],
                "raw_affine_rms_mm": summary["residuals"]["raw_affine_rms_mm"],
            }
        )

    summary_csv = output_root / "displacement_affine_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    return outputs_by_sample, summary_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess DIC displacement point-cloud fields.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("1_samples") / "data",
        help="Directory containing sample folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("1_samples") / "data" / "processed",
        help="Directory where postprocessed displacement files will be written.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["longitudinal", "transverse"],
        help="Sample folder names to process.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG/SVG cloud-map rendering and write CSV/JSON outputs only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs_by_sample, summary_csv = process_all(
        args.data_root,
        args.output_root,
        samples=args.samples,
        write_plots=not args.no_plots,
    )
    for sample_name, outputs in outputs_by_sample.items():
        print(f"{sample_name}: wrote {len(outputs)} displacement postprocessing files")
    print(f"summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
