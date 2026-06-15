from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations
import math
import os
from pathlib import Path


RESEARCH_WIDTH = 80.0
RESEARCH_LENGTH = 160.0
PLATE_THICKNESS = 2.0

HOLE_DIAMETER = 8.0
HOLE_RADIUS = 4.0
HOLE_COUNT_UNIFORM = 24
MIN_HOLE_EDGE_SPACING = 8.0
MIN_CENTER_DISTANCE = 16.0
MIN_HOLE_EDGE_TO_RESEARCH_EDGE = 10.0
MIN_CENTER_TO_RESEARCH_EDGE = 14.0

OUTLINE_SEGMENTS_PER_BEZIER = 24


@dataclass(frozen=True)
class Scheme:
    name: str
    grip_width: float
    grip_length: float
    transition_length: float
    research_width: float = RESEARCH_WIDTH
    research_length: float = RESEARCH_LENGTH
    thickness: float = PLATE_THICKNESS

    @property
    def total_length(self) -> float:
        return 2 * self.grip_length + 2 * self.transition_length + self.research_length

    @property
    def center_x(self) -> float:
        return self.research_width / 2.0

    @property
    def research_y_min(self) -> float:
        return self.grip_length + self.transition_length

    @property
    def research_y_max(self) -> float:
        return self.research_y_min + self.research_length

    @property
    def top_grip_y_min(self) -> float:
        return self.total_length - self.grip_length

    @property
    def grip_left_x(self) -> float:
        return self.center_x - self.grip_width / 2.0

    @property
    def grip_right_x(self) -> float:
        return self.center_x + self.grip_width / 2.0


@dataclass(frozen=True)
class Hole:
    x: float
    y: float
    radius: float

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


@dataclass(frozen=True)
class Specimen:
    name: str
    title: str
    scheme: Scheme
    holes: tuple[Hole, ...]
    note: str


def build_schemes() -> list[Scheme]:
    return [
        Scheme(name="scheme1", grip_width=40.0, grip_length=60.0, transition_length=30.0),
        Scheme(name="scheme2", grip_width=35.0, grip_length=50.0, transition_length=20.0),
    ]


def build_specimens() -> list[Specimen]:
    scheme1, scheme2 = build_schemes()
    return [
        Specimen(
            name="scheme1_solid_plate",
            title="scheme1 Solid Plate",
            scheme=scheme1,
            holes=(),
            note="No holes.",
        ),
        Specimen(
            name="scheme1_single_hole_plate",
            title="scheme1 Single-Hole Plate",
            scheme=scheme1,
            holes=build_single_hole(scheme1),
            note="Single centered hole with area equal to 24 holes of 8 mm diameter.",
        ),
        Specimen(
            name="scheme1_uniform_perforated_plate",
            title="scheme1 Uniform Perforated Plate",
            scheme=scheme1,
            holes=build_uniform_holes(scheme1),
            note="Uniform 4 by 6 hole array inside the research region.",
        ),
        Specimen(
            name="scheme2_solid_plate",
            title="scheme2 Solid Plate",
            scheme=scheme2,
            holes=(),
            note="No holes.",
        ),
        Specimen(
            name="scheme2_single_hole_plate",
            title="scheme2 Single-Hole Plate",
            scheme=scheme2,
            holes=build_single_hole(scheme2),
            note="Single centered hole with area equal to 24 holes of 8 mm diameter.",
        ),
        Specimen(
            name="scheme2_uniform_perforated_plate",
            title="scheme2 Uniform Perforated Plate",
            scheme=scheme2,
            holes=build_uniform_holes(scheme2),
            note="Uniform 4 by 6 hole array inside the research region.",
        ),
    ]


def build_single_hole(scheme: Scheme) -> tuple[Hole, ...]:
    diameter = HOLE_DIAMETER * math.sqrt(HOLE_COUNT_UNIFORM)
    return (
        Hole(
            x=scheme.center_x,
            y=(scheme.research_y_min + scheme.research_y_max) / 2.0,
            radius=diameter / 2.0,
        ),
    )


def build_uniform_holes(scheme: Scheme) -> tuple[Hole, ...]:
    x_values = linspace(
        MIN_CENTER_TO_RESEARCH_EDGE,
        scheme.research_width - MIN_CENTER_TO_RESEARCH_EDGE,
        4,
    )
    y_values = linspace(
        scheme.research_y_min + MIN_CENTER_TO_RESEARCH_EDGE,
        scheme.research_y_max - MIN_CENTER_TO_RESEARCH_EDGE,
        6,
    )
    return tuple(Hole(x=x, y=y, radius=HOLE_RADIUS) for y in y_values for x in x_values)


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1.")
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def sample_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    include_start: bool,
) -> list[tuple[float, float]]:
    start_index = 0 if include_start else 1
    return [
        cubic_bezier(p0, p1, p2, p3, index / OUTLINE_SEGMENTS_PER_BEZIER)
        for index in range(start_index, OUTLINE_SEGMENTS_PER_BEZIER + 1)
    ]


def build_outline_points(scheme: Scheme) -> list[tuple[float, float]]:
    lt = scheme.transition_length
    left_grip = scheme.grip_left_x
    right_grip = scheme.grip_right_x
    y0 = 0.0
    y_grip_top = scheme.grip_length
    y_research_bottom = scheme.research_y_min
    y_research_top = scheme.research_y_max
    y_top_grip_bottom = scheme.top_grip_y_min
    y_top = scheme.total_length

    points: list[tuple[float, float]] = [
        (left_grip, y0),
        (right_grip, y0),
        (right_grip, y_grip_top),
    ]
    points.extend(
        sample_bezier(
            (right_grip, y_grip_top),
            (right_grip, y_grip_top + 0.5 * lt),
            (scheme.research_width, y_research_bottom - 0.5 * lt),
            (scheme.research_width, y_research_bottom),
            include_start=False,
        )
    )
    points.append((scheme.research_width, y_research_top))
    points.extend(
        sample_bezier(
            (scheme.research_width, y_research_top),
            (scheme.research_width, y_research_top + 0.5 * lt),
            (right_grip, y_top_grip_bottom - 0.5 * lt),
            (right_grip, y_top_grip_bottom),
            include_start=False,
        )
    )
    points.extend(
        [
            (right_grip, y_top),
            (left_grip, y_top),
            (left_grip, y_top_grip_bottom),
        ]
    )
    points.extend(
        sample_bezier(
            (left_grip, y_top_grip_bottom),
            (left_grip, y_top_grip_bottom - 0.5 * lt),
            (0.0, y_research_top + 0.5 * lt),
            (0.0, y_research_top),
            include_start=False,
        )
    )
    points.append((0.0, y_research_bottom))
    points.extend(
        sample_bezier(
            (0.0, y_research_bottom),
            (0.0, y_research_bottom - 0.5 * lt),
            (left_grip, y_grip_top + 0.5 * lt),
            (left_grip, y_grip_top),
            include_start=False,
        )
    )
    return remove_adjacent_duplicates(points)


def remove_adjacent_duplicates(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in points:
        if not result or math.hypot(point[0] - result[-1][0], point[1] - result[-1][1]) > 1e-9:
            result.append(point)
    return result


def validate_specimen(specimen: Specimen) -> None:
    expected_count = expected_hole_count(specimen)
    if len(specimen.holes) != expected_count:
        raise ValueError(f"{specimen.name} has {len(specimen.holes)} holes; expected {expected_count}.")

    for index, hole in enumerate(specimen.holes, start=1):
        validate_hole_in_research_region(specimen, index, hole)

    indexed_holes = enumerate(specimen.holes, start=1)
    for (left_index, left), (right_index, right) in combinations(indexed_holes, 2):
        center_distance = math.hypot(left.x - right.x, left.y - right.y)
        minimum_distance = left.radius + right.radius + MIN_HOLE_EDGE_SPACING
        if center_distance + 1e-9 < minimum_distance:
            raise ValueError(
                f"{specimen.name} holes {left_index} and {right_index} are too close: "
                f"{center_distance:.6f} mm < {minimum_distance:.6f} mm."
            )

    if "single_hole" in specimen.name:
        total_area = math.pi * specimen.holes[0].radius**2
        reference_area = HOLE_COUNT_UNIFORM * math.pi * HOLE_RADIUS**2
        if not math.isclose(total_area, reference_area, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{specimen.name} single-hole area does not match uniform-hole area.")


def expected_hole_count(specimen: Specimen) -> int:
    if "solid" in specimen.name:
        return 0
    if "single_hole" in specimen.name:
        return 1
    if "uniform" in specimen.name:
        return HOLE_COUNT_UNIFORM
    raise ValueError(f"Unknown specimen name: {specimen.name}")


def validate_hole_in_research_region(specimen: Specimen, index: int, hole: Hole) -> None:
    scheme = specimen.scheme
    left_margin = hole.x - hole.radius
    right_margin = scheme.research_width - hole.x - hole.radius
    bottom_margin = hole.y - hole.radius - scheme.research_y_min
    top_margin = scheme.research_y_max - hole.y - hole.radius
    margin = min(left_margin, right_margin, bottom_margin, top_margin)
    if margin + 1e-9 < MIN_HOLE_EDGE_TO_RESEARCH_EDGE:
        raise ValueError(
            f"{specimen.name} hole {index} violates research-region edge margin: "
            f"{margin:.6f} mm < {MIN_HOLE_EDGE_TO_RESEARCH_EDGE:.6f} mm."
        )


def import_ezdxf():
    os.environ["XDG_CACHE_HOME"] = str(Path(__file__).resolve().parent / ".ezdxf_cache")
    import ezdxf

    return ezdxf


def ensure_layer(doc, name: str, color: int) -> None:
    if name not in doc.layers:
        doc.layers.add(name, color=color)


def write_dxf(specimen: Specimen, output_path: Path) -> Path:
    ezdxf = import_ezdxf()
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    ensure_layer(doc, "CUT_OUTLINE", 7)
    ensure_layer(doc, "CUT_HOLES", 1)

    modelspace = doc.modelspace()
    modelspace.add_lwpolyline(
        build_outline_points(specimen.scheme),
        close=True,
        dxfattribs={"layer": "CUT_OUTLINE"},
    )
    for hole in specimen.holes:
        modelspace.add_circle(
            (hole.x, hole.y),
            radius=hole.radius,
            dxfattribs={"layer": "CUT_HOLES"},
        )

    doc.saveas(str(output_path))
    return output_path


def format_mm(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=0.5e-6):
        value = 0.0
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def write_coordinate_table(specimens: list[Specimen], output_path: Path) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["specimen", "hole_index", "x_mm", "y_mm", "radius_mm", "diameter_mm", "note"])
        for specimen in specimens:
            for index, hole in enumerate(specimen.holes, start=1):
                writer.writerow(
                    [
                        specimen.name,
                        index,
                        format_mm(hole.x),
                        format_mm(hole.y),
                        format_mm(hole.radius),
                        format_mm(hole.diameter),
                        specimen.note,
                    ]
                )
    return output_path


def write_readme(output_path: Path) -> Path:
    content = """# Tensile Sample CAD Outputs

## Manufacturing Notes
- All dimensions are in millimeters.
- Plate thickness is 2 mm.
- DXF files use the `CUT_OUTLINE` layer for the closed outer contour and the `CUT_HOLES` layer for circular through-cuts.
- Use the coordinate table as the hole-center reference for perforated and single-hole samples.

## Scheme Dimensions
- scheme1: 40 mm grip width, 60 mm grip length, 30 mm transition length, 80 mm by 160 mm research region.
- scheme2: 35 mm grip width, 50 mm grip length, 20 mm transition length, 80 mm by 160 mm research region.

## Research Region Notes
- The research region width is 80 mm and length is 160 mm.
- Uniform perforated plates use a 4 by 6 array of 8 mm diameter holes.
- Single-hole plates use one centered hole with area equal to the 24-hole uniform pattern.
- Solid plates contain no holes.
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_outputs(output_dir: os.PathLike[str] | str) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    specimens = build_specimens()
    for specimen in specimens:
        validate_specimen(specimen)

    written_paths: list[Path] = []
    for specimen in specimens:
        written_paths.append(write_dxf(specimen, output_path / f"{specimen.name}.dxf"))
    written_paths.append(write_coordinate_table(specimens, output_path / "tensile_sample_hole_coordinates.csv"))
    written_paths.append(write_readme(output_path / "README.md"))
    return written_paths


def main() -> None:
    for path in write_outputs(Path(__file__).resolve().parent):
        print(path)


if __name__ == "__main__":
    main()
