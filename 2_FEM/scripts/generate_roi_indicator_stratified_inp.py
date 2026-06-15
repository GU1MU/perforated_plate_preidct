from __future__ import print_function

import io
import json
import math
import os
import sys


try:
    TEXT_TYPE = unicode
except NameError:
    TEXT_TYPE = str

_SCRIPT_FILENAME = "generate_roi_indicator_stratified_inp.py"


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


PLATE_X = 80.0
PLATE_Y = 160.0
PLATE_THICKNESS = 2.0
HOLE_RADIUS = 4.0
HOLE_DIAMETER = 2.0 * HOLE_RADIUS
HOLE_COUNT = 24
MIN_CENTER_DISTANCE = 10.0
MIN_CENTER_TO_EDGE = 7.0

DEFAULT_E = 69000.0
DEFAULT_NU = 0.33
DEFAULT_U = 0.1
DEFAULT_MESH_SIZE = 1.0
ELEMENT_TYPE = "CPS6"

SCRIPT_PATH = _resolve_script_path(globals().get("__file__"), sys.argv, os.getcwd())
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
FEM_ROOT = os.path.join(PROJECT_ROOT, "2_FEM")
TEMP_DIR = os.path.join(FEM_ROOT, "temp")
INP_DIR = os.path.join(TEMP_DIR, "solve_inp")
TEST_WORK_DIR = os.path.join(TEMP_DIR, "tests")
DEFAULT_OUTPUT_DIR = INP_DIR
SELECTED_LAYOUTS_JSON = os.path.join(
    FEM_ROOT,
    "results",
    "pilot_indicator_sampling",
    "seed_20260609_candidates_100000_target_200",
    "selected_layouts.json",
)
E = DEFAULT_E
U = DEFAULT_U
MESH_SIZE = DEFAULT_MESH_SIZE
INSTANCE = 1
REF = "solid"
VALID_REFS = (None, "solid", "uniform")
GROUP_INSTANCE_COUNTS = 150
WARM_START = True

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
REFERENCE_LAYOUTS = {
    "uniform": {
        "columns": 4,
        "rows": 6,
    },
}


def project_root():
    return PROJECT_ROOT


def fem_root():
    return FEM_ROOT


def temp_dir():
    return TEMP_DIR


def inp_dir():
    return INP_DIR


def test_work_dir():
    return TEST_WORK_DIR


def ensure_directory(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def enter_temp_work_dir():
    ensure_directory(TEMP_DIR)
    os.chdir(TEMP_DIR)
    return TEMP_DIR


def read_json_file(path):
    with io.open(path, "r", encoding="utf-8-sig") as json_file:
        return json.load(json_file)


def write_json_file(path, data):
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    text = json.dumps(data, indent=2, sort_keys=True)
    if sys.version_info[0] < 3:
        text = text.decode("ascii")
    with io.open(path, "w", encoding="utf-8") as json_file:
        json_file.write(text)
        json_file.write(u"\n")
    return path


def linspace(start, stop, count):
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [start]
    step = (stop - start) / float(count - 1)
    values = []
    for index in range(count):
        values.append(start + step * index)
    return values


def squared_distance(left, right):
    dx = left["x"] - right["x"]
    dy = left["y"] - right["y"]
    return dx * dx + dy * dy


def validate_holes(holes):
    if len(holes) != HOLE_COUNT:
        raise ValueError("expected %d holes, got %d" % (HOLE_COUNT, len(holes)))

    xmin = MIN_CENTER_TO_EDGE
    xmax = PLATE_X - MIN_CENTER_TO_EDGE
    ymin = MIN_CENTER_TO_EDGE
    ymax = PLATE_Y - MIN_CENTER_TO_EDGE

    for index, hole in enumerate(holes):
        x = hole["x"]
        y = hole["y"]
        if x < xmin or x > xmax or y < ymin or y > ymax:
            raise ValueError("hole %d outside allowed bounds" % index)

    min_distance_sq = MIN_CENTER_DISTANCE * MIN_CENTER_DISTANCE
    for left_index in range(len(holes)):
        for right_index in range(left_index + 1, len(holes)):
            if squared_distance(holes[left_index], holes[right_index]) < min_distance_sq:
                raise ValueError(
                    "holes %d and %d violate minimum center distance"
                    % (left_index, right_index)
                )
    return None


def nearest_neighbor_distances(holes):
    distances = []
    for left_index, left in enumerate(holes):
        nearest_sq = None
        for right_index, right in enumerate(holes):
            if left_index == right_index:
                continue
            dist_sq = squared_distance(left, right)
            if nearest_sq is None or dist_sq < nearest_sq:
                nearest_sq = dist_sq
        distances.append(math.sqrt(nearest_sq))
    return distances


def mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / float(len(values))


def variance(values):
    avg = mean(values)
    total = 0.0
    for value in values:
        total += (value - avg) * (value - avg)
    return total / float(len(values))


def calculate_metrics(holes):
    nearest = nearest_neighbor_distances(holes)
    d_nn_avg = mean(nearest)
    cluster_index = HOLE_DIAMETER / d_nn_avg

    normalized_x = [hole["x"] / PLATE_X for hole in holes]
    normalized_y = [hole["y"] / PLATE_Y for hole in holes]
    var_x = variance(normalized_x)
    var_y = variance(normalized_y)
    denominator = var_x + var_y
    if denominator <= 0.0:
        orientation_index = 0.0
    else:
        orientation_index = (var_x - var_y) / denominator

    return {
        "d_eq": HOLE_DIAMETER,
        "d_nn_avg": d_nn_avg,
        "cluster_index": cluster_index,
        "orientation_index": orientation_index,
        "var_x_norm": var_x,
        "var_y_norm": var_y,
    }


def geometry_contract():
    return {
        "plate_x": PLATE_X,
        "plate_y": PLATE_Y,
        "plate_thickness": PLATE_THICKNESS,
        "hole_radius": HOLE_RADIUS,
        "hole_count": HOLE_COUNT,
        "min_center_distance": MIN_CENTER_DISTANCE,
        "min_center_to_edge": MIN_CENTER_TO_EDGE,
    }


def values_match(actual, expected):
    try:
        return abs(float(actual) - float(expected)) <= 1.0e-9
    except (TypeError, ValueError):
        return actual == expected


def validate_payload_geometry(payload):
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("selected layout geometry is missing")
    expected_geometry = geometry_contract()
    for key in sorted(expected_geometry):
        expected = expected_geometry[key]
        actual = geometry.get(key)
        if not values_match(actual, expected):
            raise ValueError(
                "geometry mismatch for %s: expected %s, got %s"
                % (key, expected, actual)
            )
    return None


def load_layout_payload(path):
    payload = read_json_file(path)
    if payload.get("schema") != "roi_indicator_stratified_layouts_v1":
        raise ValueError("unsupported selected layout schema: %s" % payload.get("schema"))
    validate_payload_geometry(payload)
    layouts = payload.get("layouts", [])
    if not layouts:
        raise ValueError("selected layout file has no layouts")
    for layout in layouts:
        holes = layout.get("holes")
        if holes is None:
            raise ValueError("layout %s has no holes" % layout.get("layout_id"))
        validate_holes(holes)
    return payload


def build_uniform_holes():
    spec = REFERENCE_LAYOUTS["uniform"]
    x_values = linspace(
        MIN_CENTER_TO_EDGE,
        PLATE_X - MIN_CENTER_TO_EDGE,
        spec["columns"],
    )
    y_values = linspace(
        MIN_CENTER_TO_EDGE,
        PLATE_Y - MIN_CENTER_TO_EDGE,
        spec["rows"],
    )
    holes = []
    for y in y_values:
        for x in x_values:
            holes.append({
                "x": x,
                "y": y,
                "r": HOLE_RADIUS,
            })
    validate_holes(holes)
    return holes


def load_reference_holes(ref):
    if ref == "uniform":
        return build_uniform_holes()
    raise ValueError("unknown reference hole layout: %s" % ref)


def group_bin_key(group):
    key = group.get("bin")
    if key:
        return key
    cluster = group.get("cluster")
    direction = group.get("direction")
    if cluster and direction:
        return "%s_%s" % (cluster, direction)
    raise ValueError("group %s has no bin or cluster/direction" % group.get("id"))


def group_lookup_by_bin():
    lookup = {}
    for group in GROUP_DEFINITIONS:
        lookup[group_bin_key(group)] = group
    return lookup


def layout_bin_key(layout):
    key = layout.get("bin")
    if key:
        return key
    cluster_label = layout.get("cluster_label")
    orientation_label = layout.get("orientation_label")
    if cluster_label and orientation_label:
        return "%s_%s" % (cluster_label, orientation_label)
    return None


def group_for_layout(layout):
    key = layout_bin_key(layout)
    group = group_lookup_by_bin().get(key)
    if group is None:
        raise ValueError("layout %s has unknown indicator bin: %s" % (
            layout.get("layout_id"),
            key,
        ))
    return group


def _coerce_instance_count(value, allow_zero):
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError("instance count must be an integer")
    if count != value:
        raise ValueError("instance count must be an integer")
    if allow_zero:
        if count < 0:
            raise ValueError("instance count must be non-negative")
    elif count < 1:
        raise ValueError("instance_count must be at least 1")
    return count


def _is_instance_count_mapping(instance_count):
    return hasattr(instance_count, "get") and hasattr(instance_count, "keys")


def _normalize_group_instance_counts(instance_count):
    group_ids = [group["id"] for group in GROUP_DEFINITIONS]
    counts = {}

    if instance_count is None:
        instance_count = INSTANCE

    if _is_instance_count_mapping(instance_count):
        for group_id in instance_count.keys():
            if group_id not in group_ids:
                raise ValueError("unknown group id in instance counts: %s" % group_id)
        for group in GROUP_DEFINITIONS:
            counts[group["id"]] = _coerce_instance_count(
                instance_count.get(group["id"], 0),
                allow_zero=True,
            )
        if sum(counts.values()) < 1:
            raise ValueError("group instance counts must include at least one instance")
        return counts

    count = _coerce_instance_count(instance_count, allow_zero=False)
    for group in GROUP_DEFINITIONS:
        counts[group["id"]] = count
    return counts


def _parse_group_inp_name(name):
    if not name.endswith("_plate.inp"):
        return None
    stem = name[:-len("_plate.inp")]
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    try:
        group_id = int(parts[0])
        instance_index = int(parts[1])
    except ValueError:
        return None
    return group_id, instance_index


def existing_group_instance_offsets():
    offsets = {}
    for group in GROUP_DEFINITIONS:
        offsets[group["id"]] = 0

    path = inp_dir()
    if not os.path.isdir(path):
        return offsets

    for name in os.listdir(path):
        parsed = _parse_group_inp_name(name)
        if parsed is None:
            continue
        group_id, instance_index = parsed
        if group_id in offsets and instance_index > offsets[group_id]:
            offsets[group_id] = instance_index
    return offsets


def _inp_path(inp_name):
    return os.path.abspath(os.path.join(inp_dir(), inp_name))


def _layouts_by_group_id(payload):
    grouped = {}
    for group in GROUP_DEFINITIONS:
        grouped[group["id"]] = []
    for layout in payload["layouts"]:
        group = group_for_layout(layout)
        grouped[group["id"]].append(layout)
    return grouped


def build_run_plan(payload=None, ref=None, instance_count=None, warm_start=False):
    if ref not in VALID_REFS:
        raise ValueError("unknown run plan reference: %s" % ref)

    ensure_directory(inp_dir())

    if ref is not None:
        instance_total = _coerce_instance_count(
            INSTANCE if instance_count is None else instance_count,
            allow_zero=False,
        )
        plan = []
        for instance_index in range(1, instance_total + 1):
            inp_name = "%s_%d_plate.inp" % (ref, instance_index)
            holes = []
            metrics = None
            if ref != "solid":
                holes = load_reference_holes(ref)
                metrics = calculate_metrics(holes)
            item = {
                "ref": ref,
                "instance_index": instance_index,
                "holes": holes,
                "inp_name": inp_name,
                "inp_path": _inp_path(inp_name),
            }
            if metrics is not None:
                item["metrics"] = metrics
            plan.append(item)
        return plan

    if payload is None:
        raise ValueError("selected layout payload is required when ref is None")

    grouped_layouts = _layouts_by_group_id(payload)
    group_instance_counts = _normalize_group_instance_counts(instance_count)
    offsets = {}
    if warm_start:
        offsets = existing_group_instance_offsets()

    plan = []
    for group in GROUP_DEFINITIONS:
        group_id = group["id"]
        group_bin = group_bin_key(group)
        requested_count = group_instance_counts[group_id]
        if requested_count < 1:
            continue
        layouts = grouped_layouts[group_id]
        if len(layouts) < requested_count:
            raise ValueError(
                "not enough layouts for group %d (%s): expected %d, got %d"
                % (group_id, group_bin, requested_count, len(layouts))
            )
        first_index = offsets.get(group_id, 0) + 1
        for local_index, layout in enumerate(layouts[:requested_count]):
            instance_index = first_index + local_index
            inp_name = "%d_%d_plate.inp" % (group_id, instance_index)
            holes = layout["holes"]
            metrics = layout.get("metrics")
            if metrics is None:
                metrics = calculate_metrics(holes)
            plan.append({
                "layout_id": layout["layout_id"],
                "candidate_index": layout.get("candidate_index"),
                "group_id": group_id,
                "cluster": group["cluster"],
                "direction": group["direction"],
                "cluster_label": layout.get("cluster_label", group["cluster"]),
                "orientation_label": layout.get("orientation_label", group["direction"]),
                "bin": group_bin,
                "instance_index": instance_index,
                "holes": holes,
                "metrics": metrics,
                "inp_name": inp_name,
                "inp_path": _inp_path(inp_name),
            })
    return plan


def build_manifest(payload, plan, material_e, displacement_u, mesh_size):
    groups = []
    sampling_domains = []
    for group in GROUP_DEFINITIONS:
        entry = {
            "id": group["id"],
            "cluster": group["cluster"],
            "direction": group["direction"],
            "bin": group_bin_key(group),
        }
        groups.append(entry)
        sampling_domains.append(dict(entry))

    runs = []
    seeds = []
    for item in plan:
        run = {
            "layout_id": item["layout_id"],
            "candidate_index": item.get("candidate_index"),
            "group_id": item["group_id"],
            "cluster": item["cluster"],
            "direction": item["direction"],
            "cluster_label": item["cluster_label"],
            "orientation_label": item["orientation_label"],
            "bin": item["bin"],
            "instance_index": item["instance_index"],
            "inp_name": item["inp_name"],
            "inp_path": item["inp_path"],
            "metrics": item["metrics"],
            "hole_count": len(item["holes"]),
        }
        runs.append(run)
        seeds.append({
            "inp_name": item["inp_name"],
            "layout_id": item["layout_id"],
            "candidate_index": item.get("candidate_index"),
            "group_id": item["group_id"],
            "instance_index": item["instance_index"],
        })

    return {
        "schema": "roi_indicator_abaqus_manifest_v1",
        "source": {
            "pilot_schema": payload.get("schema"),
            "pilot_seed": payload.get("seed"),
            "pilot_candidate_count": payload.get("candidate_count"),
            "pilot_target_per_bin": payload.get("target_per_bin"),
            "thresholds": payload.get("thresholds"),
            "bin_counts": payload.get("bin_counts"),
        },
        "plate": {
            "x": PLATE_X,
            "y": PLATE_Y,
            "thickness": PLATE_THICKNESS,
        },
        "geometry": geometry_contract(),
        "material": {
            "E": material_e,
            "nu": DEFAULT_NU,
        },
        "load": {
            "u": displacement_u,
        },
        "mesh_size": mesh_size,
        "mesh_size_default": DEFAULT_MESH_SIZE,
        "defaults": {
            "material_e": DEFAULT_E,
            "displacement_u": DEFAULT_U,
            "mesh_size": DEFAULT_MESH_SIZE,
        },
        "element_type": ELEMENT_TYPE,
        "constraints": {
            "hole_count": HOLE_COUNT,
            "hole_radius": HOLE_RADIUS,
            "min_center_distance": MIN_CENTER_DISTANCE,
            "min_center_to_edge": MIN_CENTER_TO_EDGE,
        },
        "groups": groups,
        "sampling_domains": sampling_domains,
        "references": REFERENCE_LAYOUTS,
        "runs": runs,
        "seeds": seeds,
    }


def _merge_entries_by_inp_name(existing_entries, new_entries):
    merged = []
    positions = {}
    for entry in existing_entries:
        name = entry.get("inp_name")
        if name is None:
            merged.append(entry)
            continue
        positions[name] = len(merged)
        merged.append(entry)

    for entry in new_entries:
        name = entry.get("inp_name")
        if name in positions:
            merged[positions[name]] = entry
        else:
            positions[name] = len(merged)
            merged.append(entry)
    return merged


def merge_manifest(existing_manifest, new_manifest):
    if not existing_manifest:
        return new_manifest

    merged = dict(new_manifest)
    merged["runs"] = _merge_entries_by_inp_name(
        existing_manifest.get("runs", []),
        new_manifest.get("runs", []),
    )
    merged["seeds"] = _merge_entries_by_inp_name(
        existing_manifest.get("seeds", []),
        new_manifest.get("seeds", []),
    )
    return merged


def load_existing_manifest(path):
    if not os.path.isfile(path):
        return None
    return read_json_file(path)


def abaqus_job_name_from_inp_path(inp_path):
    stem = os.path.splitext(os.path.basename(inp_path))[0]
    name_chars = []
    for char in stem:
        if char.isalnum() or char == "_":
            name_chars.append(char)
        else:
            name_chars.append("_")
    name = "".join(name_chars)
    if not name:
        name = "job"
    if not name[0].isalpha():
        name = "job_" + name
    if sys.version_info[0] < 3 and isinstance(name, TEXT_TYPE):
        name = name.encode("ascii")
    return name


def move_generated_inp(source_path, target_path):
    source_abs = os.path.abspath(source_path)
    target_abs = os.path.abspath(target_path)
    ensure_directory(os.path.dirname(target_abs))
    if source_abs == target_abs:
        return target_abs
    if os.path.exists(target_abs):
        os.remove(target_abs)
    os.rename(source_abs, target_abs)
    return target_abs


def write_inp_with_abaqus(run_item, material_e, displacement_u, mesh_size):
    enter_temp_work_dir()

    from abaqus import mdb
    from abaqusConstants import TWO_D_PLANAR, DEFORMABLE_BODY, ON, OFF, UNSET, STANDARD, TRI, CPS6
    from mesh import ElemType

    job_name = abaqus_job_name_from_inp_path(run_item["inp_path"])
    model_name = "Model_%s" % job_name
    if model_name in mdb.models:
        del mdb.models[model_name]
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    model = mdb.Model(name=model_name)
    sketch = model.ConstrainedSketch(
        name="PlateProfile",
        sheetSize=2.0 * max(PLATE_X, PLATE_Y),
    )
    sketch.rectangle(point1=(0.0, 0.0), point2=(PLATE_X, PLATE_Y))
    for hole in run_item["holes"]:
        x = hole["x"]
        y = hole["y"]
        r = hole.get("r", HOLE_RADIUS)
        sketch.CircleByCenterPerimeter(center=(x, y), point1=(x + r, y))

    part = model.Part(
        name="Plate",
        dimensionality=TWO_D_PLANAR,
        type=DEFORMABLE_BODY,
    )
    part.BaseShell(sketch=sketch)
    faces = part.faces[:]

    material = model.Material(name="PlateMaterial")
    material.Elastic(table=((material_e, DEFAULT_NU),))
    model.HomogeneousSolidSection(
        name="PlateSection",
        material="PlateMaterial",
        thickness=PLATE_THICKNESS,
    )
    face_set = part.Set(faces=faces, name="PlateFace")
    part.SectionAssignment(region=face_set, sectionName="PlateSection")

    tol = 1.0e-6
    bottom_edges = part.edges.getByBoundingBox(
        xMin=-tol,
        yMin=-tol,
        zMin=-tol,
        xMax=PLATE_X + tol,
        yMax=tol,
        zMax=tol,
    )
    top_edges = part.edges.getByBoundingBox(
        xMin=-tol,
        yMin=PLATE_Y - tol,
        zMin=-tol,
        xMax=PLATE_X + tol,
        yMax=PLATE_Y + tol,
        zMax=tol,
    )
    lower_left_vertices = part.vertices.getByBoundingBox(
        xMin=-tol,
        yMin=-tol,
        zMin=-tol,
        xMax=tol,
        yMax=tol,
        zMax=tol,
    )
    part.Set(edges=bottom_edges, name="BottomEdge")
    part.Set(edges=top_edges, name="TopEdge")
    part.Set(vertices=lower_left_vertices, name="LowerLeftVertex")

    assembly = model.rootAssembly
    instance = assembly.Instance(name="PlateInstance", part=part, dependent=ON)
    model.StaticStep(name="Load", previous="Initial")
    if hasattr(model, "fieldOutputRequests"):
        model.fieldOutputRequests["F-Output-1"].setValues(variables=("S", "E", "U", "RF"))
    model.DisplacementBC(
        name="BottomUY",
        createStepName="Load",
        region=instance.sets["BottomEdge"],
        u1=UNSET,
        u2=-0.5 * displacement_u,
    )
    model.DisplacementBC(
        name="TopUY",
        createStepName="Load",
        region=instance.sets["TopEdge"],
        u1=UNSET,
        u2=0.5 * displacement_u,
    )
    model.DisplacementBC(
        name="FixLowerLeftUX",
        createStepName="Load",
        region=instance.sets["LowerLeftVertex"],
        u1=0.0,
        u2=UNSET,
    )

    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.setMeshControls(regions=faces, elemShape=TRI)
    elem_type = ElemType(elemCode=CPS6, elemLibrary=STANDARD)
    part.setElementType(regions=(faces,), elemTypes=(elem_type,))
    part.generateMesh()

    job = mdb.Job(name=job_name, model=model.name)
    job.writeInput(consistencyChecking=OFF)
    return move_generated_inp(job_name + ".inp", run_item["inp_path"])


def main():
    ensure_directory(inp_dir())
    payload = None
    if REF is None:
        payload = load_layout_payload(SELECTED_LAYOUTS_JSON)
    plan = build_run_plan(
        payload,
        ref=REF,
        instance_count=GROUP_INSTANCE_COUNTS if REF is None else INSTANCE,
        warm_start=(REF is None and WARM_START),
    )
    enter_temp_work_dir()
    for item in plan:
        saved_path = write_inp_with_abaqus(item, E, U, MESH_SIZE)
        if saved_path is None:
            saved_path = item["inp_path"]
        print("Saved inp: %s" % saved_path)

    if REF is not None:
        return None

    manifest = build_manifest(
        payload,
        plan,
        material_e=E,
        displacement_u=U,
        mesh_size=MESH_SIZE,
    )
    manifest_path = os.path.join(inp_dir(), "group_manifest.json")
    if WARM_START:
        manifest = merge_manifest(load_existing_manifest(manifest_path), manifest)
    write_json_file(manifest_path, manifest)
    return manifest_path


if __name__ == "__main__":
    main()
