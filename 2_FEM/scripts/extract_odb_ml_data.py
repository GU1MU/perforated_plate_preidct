from __future__ import print_function

import csv
import io
import json
import math
import os
import re
import sys


_SCRIPT_FILENAME = "extract_odb_ml_data.py"


def _argv_script_candidate(argument):
    if not argument:
        return None
    if argument.lower().startswith("nogui="):
        argument = argument.split("=", 1)[1]
    argument = argument.strip("\"'")
    normalized = argument.replace("\\", os.sep).replace("/", os.sep)
    if os.path.basename(os.path.normpath(normalized)) != _SCRIPT_FILENAME:
        return None
    return normalized


def _cwd_and_ancestors(cwd):
    roots = []
    path = os.path.abspath(cwd)
    while path and path not in roots:
        roots.append(path)
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return roots


def _resolve_script_path(script_file, argv, cwd):
    if script_file:
        return os.path.abspath(script_file)
    if cwd is None:
        cwd = os.getcwd()
    cwd = os.path.abspath(cwd)
    for argument in argv or []:
        candidate = _argv_script_candidate(argument)
        if candidate is None:
            continue
        if os.path.isabs(candidate):
            return os.path.abspath(candidate)
        for root in _cwd_and_ancestors(cwd):
            path = os.path.abspath(os.path.join(root, candidate))
            if os.path.exists(path):
                return path
        return os.path.abspath(os.path.join(cwd, candidate))

    fallback_relatives = [
        os.path.join("2_FEM", "scripts", _SCRIPT_FILENAME),
        os.path.join("scripts", _SCRIPT_FILENAME),
        _SCRIPT_FILENAME,
    ]
    for root in _cwd_and_ancestors(cwd):
        for relative in fallback_relatives:
            path = os.path.abspath(os.path.join(root, relative))
            if os.path.exists(path):
                return path
    return os.path.abspath(os.path.join(cwd, fallback_relatives[0]))


DEFAULT_STEP = "Load"
DEFAULT_FRAME_INDEX = -1
DEFAULT_TOP_SET = "TopEdge"
DEFAULT_BOTTOM_SET = "BottomEdge"
DEFAULT_INSTANCE = "PlateInstance"
DEFAULT_COMPONENT = 2

PLATE_X = 80.0
PLATE_Y = 160.0
HOLE_RADIUS = 4.0
HOLE_COUNT = 24
MIN_CENTER_TO_EDGE = 7.0
NEAR_HOLE_BAND_MM = 1.0

REF = None
VALID_REFS = (None, "solid", "uniform")
GROUP_COUNT = 200
MAX_ODB_PER_RUN = 500
WARM_START = True
PRINT_PROGRESS = True
PROGRESS_BAR_WIDTH = 20

ODB_DIR = os.path.join("odb")
LAYOUT_FILE = os.path.join(
    "..",
    "results",
    "pilot_indicator_sampling",
    "seed_20260609_candidates_100000_target_200",
    "selected_layouts.json",
)
OUTPUT_DIR = os.path.join("..", "results", "odb_ml_data")
SUMMARY_CSV = "odb_ml_summary.csv"
SUMMARY_JSON = "odb_ml_summary.json"
FAILURES_CSV = "odb_ml_failures.csv"
REFERENCE_ODB = os.path.join("odb", "solid_1_plate.odb")

GROUP_ODB_RE = re.compile(r"^([0-9]+)_([0-9]+)_plate[.]odb$")
REF_ODB_RE = re.compile(r"^(solid|uniform)_([0-9]+)_plate[.]odb$")

GROUP_DEFINITIONS = [
    {"id": 1, "cluster": "low", "direction": "x", "bin": "low_x"},
    {"id": 2, "cluster": "low", "direction": "none", "bin": "low_none"},
    {"id": 3, "cluster": "low", "direction": "y", "bin": "low_y"},
    {"id": 4, "cluster": "medium", "direction": "x", "bin": "medium_x"},
    {"id": 5, "cluster": "medium", "direction": "none", "bin": "medium_none"},
    {"id": 6, "cluster": "medium", "direction": "y", "bin": "medium_y"},
    {"id": 7, "cluster": "high", "direction": "x", "bin": "high_x"},
    {"id": 8, "cluster": "high", "direction": "none", "bin": "high_none"},
    {"id": 9, "cluster": "high", "direction": "y", "bin": "high_y"},
]

SCRIPT_PATH = _resolve_script_path(globals().get("__file__"), sys.argv, os.getcwd())
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
FEM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
TEMP_DIR = os.path.join(FEM_ROOT, "temp")


def ensure_directory(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)
    return path


def enter_temp_work_dir(verbose=False):
    ensure_directory(TEMP_DIR)
    os.chdir(TEMP_DIR)
    if verbose:
        print("Working directory: %s" % os.path.abspath(os.getcwd()))
        sys.stdout.flush()
    return TEMP_DIR


def format_progress_message(index, total, path, status="processing"):
    if total < 1:
        percent = 100.0
    else:
        percent = 100.0 * float(index) / float(total)
    filled = int(round(PROGRESS_BAR_WIDTH * percent / 100.0))
    if filled < 0:
        filled = 0
    if filled > PROGRESS_BAR_WIDTH:
        filled = PROGRESS_BAR_WIDTH
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    return "[%d/%d %5.1f%% %s] %s %s" % (
        index,
        total,
        percent,
        bar,
        status,
        os.path.basename(str(path)),
    )


def print_progress(index, total, path, status="processing"):
    if PRINT_PROGRESS:
        print(format_progress_message(index, total, path, status=status))
        sys.stdout.flush()


def print_run_message(message):
    if PRINT_PROGRESS:
        print(message)
        sys.stdout.flush()


def _sort_key(name):
    parsed = parse_odb_name(os.path.basename(str(name)))
    if parsed is None:
        return (9, str(name))
    ref = parsed["ref"]
    if ref is None:
        return (0, parsed["group_index"], parsed["instance_index"], parsed["odb_name"])
    return (1, ref, parsed["instance_index"], parsed["odb_name"])


def parse_odb_name(name):
    base = os.path.basename(str(name))
    match = GROUP_ODB_RE.match(base)
    if match:
        return {
            "odb_name": base,
            "ref": None,
            "group_index": int(match.group(1)),
            "instance_index": int(match.group(2)),
        }
    match = REF_ODB_RE.match(base)
    if match:
        return {
            "odb_name": base,
            "ref": match.group(1),
            "group_index": None,
            "instance_index": int(match.group(2)),
        }
    return None


def filter_odb_names(names, ref=None):
    if ref not in VALID_REFS:
        raise ValueError("unknown REF: %s" % ref)
    selected = []
    for name in names:
        parsed = parse_odb_name(name)
        if parsed is None:
            continue
        if ref is None and parsed["ref"] is None:
            selected.append(os.path.basename(str(name)))
        elif ref is not None and parsed["ref"] == ref:
            selected.append(os.path.basename(str(name)))
    selected.sort(key=lambda item: _sort_key(item))
    return selected


def discover_odb_paths(odb_dir=ODB_DIR, ref=REF):
    if not os.path.isdir(odb_dir):
        return []
    names = filter_odb_names(os.listdir(odb_dir), ref=ref)
    return [os.path.join(odb_dir, name) for name in names]


def limit_group_paths_per_group(odb_paths, group_count=GROUP_COUNT, ref=REF):
    if group_count is None or ref is not None:
        return list(odb_paths)
    count = int(group_count)
    if count < 1:
        raise ValueError("GROUP_COUNT must be a positive integer or None")
    selected = []
    counts = {}
    for path in odb_paths:
        parsed = parse_odb_name(os.path.basename(str(path)))
        if parsed is None or parsed["ref"] is not None:
            continue
        group_index = parsed["group_index"]
        used = counts.get(group_index, 0)
        if used < count:
            selected.append(path)
            counts[group_index] = used + 1
    return selected


def limit_paths_per_run(odb_paths, max_odb_per_run=MAX_ODB_PER_RUN):
    if max_odb_per_run is None:
        return list(odb_paths)
    count = int(max_odb_per_run)
    if count < 1:
        raise ValueError("MAX_ODB_PER_RUN must be a positive integer or None")
    return list(odb_paths)[:count]


def read_json_file(path):
    with io.open(path, "r", encoding="utf-8-sig") as json_file:
        return json.load(json_file)


def linspace(start, stop, count):
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return [start + step * index for index in range(count)]


def build_uniform_holes():
    holes = []
    for y in linspace(MIN_CENTER_TO_EDGE, PLATE_Y - MIN_CENTER_TO_EDGE, 6):
        for x in linspace(MIN_CENTER_TO_EDGE, PLATE_X - MIN_CENTER_TO_EDGE, 4):
            holes.append({"x": x, "y": y, "r": HOLE_RADIUS})
    return holes


def load_layout_payload(path=LAYOUT_FILE):
    payload = read_json_file(path)
    layouts = payload.get("layouts")
    if not isinstance(layouts, list):
        raise ValueError("layout file has no layouts array: %s" % path)
    return payload


def _group_bin_key(group):
    key = group.get("bin")
    if key:
        return key
    cluster = group.get("cluster")
    direction = group.get("direction")
    if cluster and direction:
        return "%s_%s" % (cluster, direction)
    raise ValueError("group %s has no bin or cluster/direction" % group.get("id"))


def _group_lookup_by_bin():
    lookup = {}
    for group in GROUP_DEFINITIONS:
        lookup[_group_bin_key(group)] = group
    return lookup


def _layout_bin_key(layout):
    key = layout.get("bin")
    if key:
        return key
    cluster = layout.get("cluster_label")
    orientation = layout.get("orientation_label")
    if cluster and orientation:
        return "%s_%s" % (cluster, orientation)
    return None


def _layout_group_id(layout):
    key = _layout_bin_key(layout)
    group = _group_lookup_by_bin().get(key)
    if group is None:
        raise ValueError("layout %s has unknown bin: %s" % (layout.get("layout_id"), key))
    return int(group["id"])


def _layouts_by_group_id(layout_payload):
    grouped = {}
    for group in GROUP_DEFINITIONS:
        grouped[int(group["id"])] = []
    if layout_payload is None:
        return grouped
    for layout in layout_payload.get("layouts", []):
        grouped[_layout_group_id(layout)].append(layout)
    return grouped


def _normalized_holes(holes):
    normalized = []
    for hole in holes:
        normalized.append({
            "x": float(hole["x"]),
            "y": float(hole["y"]),
            "r": float(hole.get("r", hole.get("radius", HOLE_RADIUS))),
        })
    return normalized


def holes_for_model(parsed, layout_payload):
    ref = parsed.get("ref")
    if ref == "solid":
        return []
    if ref == "uniform":
        return build_uniform_holes()
    group_index = parsed.get("group_index")
    instance_index = parsed.get("instance_index")
    grouped = _layouts_by_group_id(layout_payload)
    layouts = grouped.get(int(group_index), [])
    position = int(instance_index) - 1
    if position < 0 or position >= len(layouts):
        raise ValueError("layout entry not found for group %s instance %s" % (group_index, instance_index))
    layout = layouts[position]
    if "holes" not in layout:
        raise ValueError(
            "layout entry for group %s instance %s has no holes array"
            % (group_index, instance_index)
        )
    return _normalized_holes(layout["holes"])


def _case_lookup(mapping, name):
    if name in mapping:
        return mapping[name]
    target = name.lower()
    for key in mapping.keys():
        if str(key).lower() == target:
            return mapping[key]
    raise KeyError(name)


def _step(odb, step_name):
    try:
        return _case_lookup(odb.steps, step_name)
    except KeyError:
        raise ValueError("step not found: %s" % step_name)


def _frame(odb, step_name, frame_index):
    step = _step(odb, step_name)
    if not step.frames:
        raise ValueError("step has no frames: %s" % step_name)
    try:
        return step.frames[frame_index]
    except IndexError:
        raise ValueError("frame index out of range: %s" % frame_index)


def _field_output(frame, field_name):
    try:
        return _case_lookup(frame.fieldOutputs, field_name)
    except KeyError:
        raise ValueError(
            "ODB frame has no field output %s. "
            "Make sure the analysis requested %s output." % (field_name, field_name)
        )


def resolve_node_set(odb, set_name, instance_name=DEFAULT_INSTANCE):
    assembly = odb.rootAssembly
    try:
        return _case_lookup(assembly.nodeSets, set_name)
    except KeyError:
        pass

    if instance_name:
        instance = _case_lookup(assembly.instances, instance_name)
        return _case_lookup(instance.nodeSets, set_name)

    for instance in assembly.instances.values():
        try:
            return _case_lookup(instance.nodeSets, set_name)
        except KeyError:
            continue
    raise ValueError("node set not found: %s" % set_name)


def _component_values(field_output, region, component):
    if component < 1:
        raise ValueError("component index is 1-based and must be positive")
    index = component - 1
    subset = field_output.getSubset(region=region)
    values = []
    for value in subset.values:
        try:
            values.append(float(value.data[index]))
        except IndexError:
            raise ValueError("field output has no component %s" % component)
    if not values:
        raise ValueError("field output has no values on node set %s" % region.name)
    return values


def _mean(values):
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / float(len(values))


def _boundary_summary(frame, region, component):
    rf_output = _field_output(frame, "RF")
    u_output = _field_output(frame, "U")
    rf_values = _component_values(rf_output, region, component)
    u_values = _component_values(u_output, region, component)
    return {
        "node_count": len(rf_values),
        "mean_reaction": _mean(rf_values),
        "total_reaction": sum(rf_values),
        "mean_displacement": _mean(u_values),
    }


def compute_equivalent_stiffness(
    odb,
    step_name=DEFAULT_STEP,
    frame_index=DEFAULT_FRAME_INDEX,
    top_set=DEFAULT_TOP_SET,
    bottom_set=DEFAULT_BOTTOM_SET,
    instance_name=DEFAULT_INSTANCE,
    component=DEFAULT_COMPONENT,
):
    frame = _frame(odb, step_name, frame_index)
    top_region = resolve_node_set(odb, top_set, instance_name=instance_name)
    bottom_region = resolve_node_set(odb, bottom_set, instance_name=instance_name)

    top = _boundary_summary(frame, top_region, component)
    bottom = _boundary_summary(frame, bottom_region, component)
    mean_reaction_magnitude = _mean([
        abs(float(top["mean_reaction"])),
        abs(float(bottom["mean_reaction"])),
    ])
    mean_displacement_magnitude = _mean([
        abs(float(top["mean_displacement"])),
        abs(float(bottom["mean_displacement"])),
    ])
    if mean_displacement_magnitude == 0.0:
        equivalent_stiffness = None
        status = "zero_displacement"
    else:
        equivalent_stiffness = mean_reaction_magnitude / mean_displacement_magnitude
        status = "ok"

    result = {
        "status": status,
        "step": step_name,
        "frame_index": frame_index,
        "component": component,
        "top_set": top_set,
        "bottom_set": bottom_set,
        "instance": instance_name or "",
        "top_node_count": int(top["node_count"]),
        "bottom_node_count": int(bottom["node_count"]),
        "top_mean_reaction": float(top["mean_reaction"]),
        "bottom_mean_reaction": float(bottom["mean_reaction"]),
        "top_total_reaction": float(top["total_reaction"]),
        "bottom_total_reaction": float(bottom["total_reaction"]),
        "top_mean_displacement": float(top["mean_displacement"]),
        "bottom_mean_displacement": float(bottom["mean_displacement"]),
        "mean_reaction_magnitude": mean_reaction_magnitude,
        "mean_displacement_magnitude": mean_displacement_magnitude,
        "equivalent_stiffness": equivalent_stiffness,
    }
    if status != "ok":
        result["warning"] = "mean boundary displacement magnitude is zero"
    return result


class CoordinateFieldValue(object):
    def __init__(self, value, xy):
        self.value = value
        self.xy = xy
        self.data = value.data
        if hasattr(value, "maxPrincipal"):
            self.maxPrincipal = value.maxPrincipal
        if hasattr(value, "elementLabel"):
            self.elementLabel = value.elementLabel
        if hasattr(value, "nodeLabel"):
            self.nodeLabel = value.nodeLabel


def max_principal_strain(value):
    if hasattr(value, "maxPrincipal") and value.maxPrincipal is not None:
        return float(value.maxPrincipal)
    data = value.data
    if len(data) < 3:
        raise ValueError("strain value needs maxPrincipal or at least three planar components")
    e11 = float(data[0])
    e22 = float(data[1])
    e12 = float(data[2])
    mean = (e11 + e22) / 2.0
    radius = math.sqrt(((e11 - e22) / 2.0) ** 2 + e12 ** 2)
    return mean + radius


def value_xy(value):
    if hasattr(value, "xy"):
        return (float(value.xy[0]), float(value.xy[1]))
    for attr in ("coordinates", "coord", "centroid"):
        if hasattr(value, attr):
            coords = getattr(value, attr)
            return (float(coords[0]), float(coords[1]))
    raise ValueError("strain value has no coordinates")


def strain_field_summary(values):
    principal = [max_principal_strain(value) for value in values]
    if not principal:
        raise ValueError("strain field has no values")
    return {
        "mean_max_principal_strain": _mean(principal),
        "max_principal_strain": max(principal),
    }


def local_hole_max_principal(values, hole, band_mm=NEAR_HOLE_BAND_MM):
    x0 = float(hole["x"])
    y0 = float(hole["y"])
    radius = float(hole["r"])
    outer = radius + float(band_mm)
    selected = []
    for value in values:
        x, y = value_xy(value)
        distance = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        if distance >= radius and distance <= outer:
            selected.append(max_principal_strain(value))
    if not selected:
        return None
    return max(selected)


def element_centroid_lookup(instance):
    node_coords = {}
    for node in instance.nodes:
        node_coords[int(node.label)] = node.coordinates
    centroids = {}
    for element in instance.elements:
        coords = [node_coords[int(label)] for label in element.connectivity]
        total_x = 0.0
        total_y = 0.0
        for coord in coords:
            total_x += float(coord[0])
            total_y += float(coord[1])
        x = total_x / float(len(coords))
        y = total_y / float(len(coords))
        centroids[int(element.label)] = (x, y)
    return centroids


def _values_need_centroids(values):
    for value in values:
        try:
            value_xy(value)
        except ValueError:
            return True
    return False


def strain_values_from_frame(frame, instance=None):
    try:
        strain_output = _field_output(frame, "E")
    except ValueError:
        strain_output = _field_output(frame, "LE")
    values = list(strain_output.values)
    if not values:
        raise ValueError("strain field has no values")
    if instance is not None and _values_need_centroids(values):
        centroids = element_centroid_lookup(instance)
        wrapped = []
        for value in values:
            if hasattr(value, "elementLabel") and int(value.elementLabel) in centroids:
                wrapped.append(CoordinateFieldValue(value, centroids[int(value.elementLabel)]))
            else:
                wrapped.append(value)
        values = wrapped
    return values


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0, 0.0):
        return None
    return float(numerator) / float(denominator)


def build_model_result(parsed, odb_path, holes, strain_values, stiffness, reference):
    summary = strain_field_summary(strain_values)
    solid_mean = reference.get("solid_mean_max_principal_strain")
    solid_stiffness = reference.get("solid_equivalent_stiffness")
    model_max = summary["max_principal_strain"]
    equivalent_stiffness = stiffness.get("equivalent_stiffness")
    warning = stiffness.get("warning", "")

    nested_holes = []
    for index, hole in enumerate(holes):
        local_peak = local_hole_max_principal(strain_values, hole)
        if local_peak is None:
            warning = _join_warning(warning, "hole %s has no strain values in annular band" % (index + 1))
        nested_holes.append({
            "index": index + 1,
            "x": float(hole["x"]),
            "y": float(hole["y"]),
            "r": float(hole["r"]),
            "local_max_principal_strain": local_peak,
            "strain_concentration_factor": _safe_ratio(local_peak, solid_mean),
        })

    return {
        "odb_name": parsed.get("odb_name") or os.path.basename(str(odb_path)),
        "odb_path": str(odb_path),
        "status": "ok",
        "ref": parsed.get("ref"),
        "group_index": parsed.get("group_index"),
        "instance_index": parsed.get("instance_index"),
        "step": stiffness.get("step", DEFAULT_STEP),
        "frame_index": stiffness.get("frame_index", DEFAULT_FRAME_INDEX),
        "hole_count": len(holes),
        "solid_mean_max_principal_strain": solid_mean,
        "solid_equivalent_stiffness": solid_stiffness,
        "model_max_principal_strain": model_max,
        "max_strain_concentration_factor": _safe_ratio(model_max, solid_mean),
        "equivalent_stiffness": equivalent_stiffness,
        "relative_equivalent_stiffness": _safe_ratio(equivalent_stiffness, solid_stiffness),
        "top_node_count": stiffness.get("top_node_count"),
        "bottom_node_count": stiffness.get("bottom_node_count"),
        "top_mean_reaction": stiffness.get("top_mean_reaction"),
        "bottom_mean_reaction": stiffness.get("bottom_mean_reaction"),
        "top_mean_displacement": stiffness.get("top_mean_displacement"),
        "bottom_mean_displacement": stiffness.get("bottom_mean_displacement"),
        "warning": warning,
        "holes": nested_holes,
    }


def _join_warning(existing, message):
    if existing:
        return "%s; %s" % (existing, message)
    return message


BASE_CSV_FIELDS = [
    "odb_name",
    "odb_path",
    "status",
    "ref",
    "group_index",
    "instance_index",
    "step",
    "frame_index",
    "hole_count",
    "solid_mean_max_principal_strain",
    "solid_equivalent_stiffness",
    "model_max_principal_strain",
    "max_strain_concentration_factor",
    "equivalent_stiffness",
    "relative_equivalent_stiffness",
    "top_node_count",
    "bottom_node_count",
    "top_mean_reaction",
    "bottom_mean_reaction",
    "top_mean_displacement",
    "bottom_mean_displacement",
    "warning",
]

INTEGER_CSV_FIELDS = set([
    "group_index",
    "instance_index",
    "frame_index",
    "hole_count",
    "top_node_count",
    "bottom_node_count",
])

FLOAT_CSV_FIELDS = set([
    "solid_mean_max_principal_strain",
    "solid_equivalent_stiffness",
    "model_max_principal_strain",
    "max_strain_concentration_factor",
    "equivalent_stiffness",
    "relative_equivalent_stiffness",
    "top_mean_reaction",
    "bottom_mean_reaction",
    "top_mean_displacement",
    "bottom_mean_displacement",
])

FAILURE_CSV_FIELDS = ["odb_name", "odb_path", "status", "message"]


def csv_fields():
    fields = list(BASE_CSV_FIELDS)
    for index in range(1, HOLE_COUNT + 1):
        prefix = "hole_%02d" % index
        fields.extend([
            prefix + "_x",
            prefix + "_y",
            prefix + "_local_max_principal_strain",
            prefix + "_strain_concentration_factor",
        ])
    return fields


def flatten_row(row):
    flat = {}
    for field in BASE_CSV_FIELDS:
        value = row.get(field, "")
        if value is None:
            value = ""
        flat[field] = value
    holes = row.get("holes") or []
    for index in range(1, HOLE_COUNT + 1):
        prefix = "hole_%02d" % index
        if index <= len(holes):
            hole = holes[index - 1]
            flat[prefix + "_x"] = hole.get("x", "")
            flat[prefix + "_y"] = hole.get("y", "")
            flat[prefix + "_local_max_principal_strain"] = _blank_none(hole.get("local_max_principal_strain", ""))
            flat[prefix + "_strain_concentration_factor"] = _blank_none(hole.get("strain_concentration_factor", ""))
        else:
            flat[prefix + "_x"] = ""
            flat[prefix + "_y"] = ""
            flat[prefix + "_local_max_principal_strain"] = ""
            flat[prefix + "_strain_concentration_factor"] = ""
    return flat


def _blank_none(value):
    if value is None:
        return ""
    return value


def completed_odb_names(rows):
    completed = set()
    for row in rows:
        if row.get("status") == "ok" and row.get("odb_name"):
            completed.add(row["odb_name"])
    return completed


def read_existing_summary(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        return [_row_from_flat_csv(row) for row in reader]


def _csv_float_or_none(value):
    if value is None or value == "":
        return None
    return float(value)


def _csv_base_value(field, value):
    if value is None or value == "":
        return ""
    if field in INTEGER_CSV_FIELDS:
        return int(float(value))
    if field in FLOAT_CSV_FIELDS:
        return float(value)
    return value


def _row_from_flat_csv(flat):
    row = {}
    for field in BASE_CSV_FIELDS:
        row[field] = _csv_base_value(field, flat.get(field, ""))

    holes = []
    for index in range(1, HOLE_COUNT + 1):
        prefix = "hole_%02d" % index
        x = flat.get(prefix + "_x", "")
        y = flat.get(prefix + "_y", "")
        local_peak = flat.get(prefix + "_local_max_principal_strain", "")
        factor = flat.get(prefix + "_strain_concentration_factor", "")
        if x == "" and y == "" and local_peak == "" and factor == "":
            continue
        holes.append({
            "index": index,
            "x": _csv_float_or_none(x),
            "y": _csv_float_or_none(y),
            "r": HOLE_RADIUS,
            "local_max_principal_strain": _csv_float_or_none(local_peak),
            "strain_concentration_factor": _csv_float_or_none(factor),
        })
    row["holes"] = holes
    return row


def write_summary_csv(rows, output_path):
    ensure_directory(os.path.dirname(os.path.abspath(output_path)))
    with open(output_path, "w") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=csv_fields(),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_row(row))


def write_summary_json(rows, output_path):
    ensure_directory(os.path.dirname(os.path.abspath(output_path)))
    text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    try:
        unicode
    except NameError:
        payload = text
    else:
        payload = unicode(text)
    with io.open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(payload)


def failure_row(path, message):
    return {
        "odb_name": os.path.basename(str(path)),
        "odb_path": str(path),
        "status": "failed",
        "message": str(message),
    }


def write_failures_csv(rows, output_path):
    ensure_directory(os.path.dirname(os.path.abspath(output_path)))
    with open(output_path, "w") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=FAILURE_CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_current_outputs(existing_rows, rows, failures, summary_path, summary_json_path, failures_path):
    all_rows = list(existing_rows) + list(rows)
    write_summary_csv(all_rows, summary_path)
    write_summary_json(all_rows, summary_json_path)
    write_failures_csv(failures, failures_path)
    return all_rows


def paths_after_warm_start(odb_paths, existing_rows, warm_start=WARM_START):
    if not warm_start:
        return list(odb_paths)
    completed = completed_odb_names(existing_rows)
    return [
        path for path in odb_paths
        if os.path.basename(str(path)) not in completed
    ]


def resolve_instance(odb, instance_name=DEFAULT_INSTANCE):
    return _case_lookup(odb.rootAssembly.instances, instance_name)


def open_odb(path):
    try:
        from odbAccess import openOdb
    except ImportError:
        raise RuntimeError("odbAccess is unavailable. Run this script with Abaqus Python.")
    return openOdb(path=str(path), readOnly=True)


def compute_reference(open_odb_func=open_odb):
    odb = open_odb_func(REFERENCE_ODB)
    try:
        frame = _frame(odb, DEFAULT_STEP, DEFAULT_FRAME_INDEX)
        instance = resolve_instance(odb, DEFAULT_INSTANCE)
        strain_values = strain_values_from_frame(frame, instance=instance)
        summary = strain_field_summary(strain_values)
        stiffness = compute_equivalent_stiffness(
            odb,
            step_name=DEFAULT_STEP,
            frame_index=DEFAULT_FRAME_INDEX,
            top_set=DEFAULT_TOP_SET,
            bottom_set=DEFAULT_BOTTOM_SET,
            instance_name=DEFAULT_INSTANCE,
            component=DEFAULT_COMPONENT,
        )
        return {
            "solid_mean_max_principal_strain": summary["mean_max_principal_strain"],
            "solid_equivalent_stiffness": stiffness.get("equivalent_stiffness"),
        }
    finally:
        if hasattr(odb, "close"):
            odb.close()


def process_one_odb(path, layout_payload, reference, open_odb_func=open_odb):
    parsed = parse_odb_name(os.path.basename(str(path)))
    if parsed is None:
        raise ValueError("unsupported ODB name: %s" % os.path.basename(str(path)))
    holes = holes_for_model(parsed, layout_payload)
    odb = open_odb_func(path)
    try:
        frame = _frame(odb, DEFAULT_STEP, DEFAULT_FRAME_INDEX)
        instance = resolve_instance(odb, DEFAULT_INSTANCE)
        strain_values = strain_values_from_frame(frame, instance=instance)
        stiffness = compute_equivalent_stiffness(
            odb,
            step_name=DEFAULT_STEP,
            frame_index=DEFAULT_FRAME_INDEX,
            top_set=DEFAULT_TOP_SET,
            bottom_set=DEFAULT_BOTTOM_SET,
            instance_name=DEFAULT_INSTANCE,
            component=DEFAULT_COMPONENT,
        )
        return build_model_result(parsed, path, holes, strain_values, stiffness, reference)
    finally:
        if hasattr(odb, "close"):
            odb.close()


def run_configured_analysis(open_odb_func=None):
    if open_odb_func is None:
        open_odb_func = open_odb
    enter_temp_work_dir(verbose=PRINT_PROGRESS)
    ensure_directory(OUTPUT_DIR)

    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    summary_json_path = os.path.join(OUTPUT_DIR, SUMMARY_JSON)
    failures_path = os.path.join(OUTPUT_DIR, FAILURES_CSV)

    existing_rows = read_existing_summary(summary_path) if WARM_START else []
    odb_paths = discover_odb_paths(ODB_DIR, ref=REF)
    odb_paths = paths_after_warm_start(odb_paths, existing_rows, warm_start=WARM_START)
    odb_paths = limit_group_paths_per_group(odb_paths, group_count=GROUP_COUNT, ref=REF)
    odb_paths = limit_paths_per_run(odb_paths, max_odb_per_run=MAX_ODB_PER_RUN)
    total_odb_count = len(odb_paths)

    print_run_message("Output directory: %s" % os.path.abspath(OUTPUT_DIR))
    print_run_message("ODB files selected: %d" % total_odb_count)

    layout_payload = None
    if REF is None:
        print_run_message("Layout file: %s" % os.path.abspath(LAYOUT_FILE))
        layout_payload = load_layout_payload(LAYOUT_FILE)
    rows = []
    failures = []
    print_run_message("Reference ODB: %s" % os.path.abspath(REFERENCE_ODB))
    try:
        reference = compute_reference(open_odb_func=open_odb_func)
    except Exception as exc:
        failures.append(
            failure_row(
                REFERENCE_ODB,
                "reference normalization failed: %s" % exc,
            )
        )
        all_rows = write_current_outputs(
            existing_rows,
            rows,
            failures,
            summary_path,
            summary_json_path,
            failures_path,
        )
        return {
            "rows": rows,
            "existing_rows": existing_rows,
            "all_rows": all_rows,
            "failures": failures,
            "summary_csv": summary_path,
            "summary_json": summary_json_path,
            "failures_csv": failures_path,
        }

    all_rows = list(existing_rows)
    for index, path in enumerate(odb_paths, 1):
        print_progress(index, total_odb_count, path, "processing")
        try:
            rows.append(process_one_odb(path, layout_payload, reference, open_odb_func=open_odb_func))
            print_progress(index, total_odb_count, path, "ok")
        except Exception as exc:
            failures.append(failure_row(path, exc))
            print_progress(index, total_odb_count, path, "failed")
        all_rows = write_current_outputs(
            existing_rows,
            rows,
            failures,
            summary_path,
            summary_json_path,
            failures_path,
        )

    if not odb_paths:
        all_rows = write_current_outputs(
            existing_rows,
            rows,
            failures,
            summary_path,
            summary_json_path,
            failures_path,
        )
    return {
        "rows": rows,
        "existing_rows": existing_rows,
        "all_rows": all_rows,
        "failures": failures,
        "summary_csv": summary_path,
        "summary_json": summary_json_path,
        "failures_csv": failures_path,
    }


def main():
    run_configured_analysis()
    return 0


if __name__ == "__main__":
    main()
