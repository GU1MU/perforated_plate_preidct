from __future__ import print_function

import csv
import hashlib
import io
import json
import os
import random
import sys


_SCRIPT_FILENAME = "generate_perforated_plate_inp.py"


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


_SCRIPT_PATH = _resolve_script_path(globals().get("__file__"), sys.argv, os.getcwd())
_SCRIPT_DIR = os.path.dirname(_SCRIPT_PATH)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
_FEM_ROOT = os.path.join(_PROJECT_ROOT, "2_FEM")
_TEMP_DIR = os.path.join(_FEM_ROOT, "temp")
_INP_DIR = os.path.join(_TEMP_DIR, "solve_inp")
_TEST_WORK_DIR = os.path.join(_TEMP_DIR, "tests")

PLATE_X = 150.0
PLATE_Y = 200.0
PLATE_THICKNESS = 2.0
HOLE_RADIUS = 5.0
HOLE_COUNT = 40
MIN_CENTER_DISTANCE = 14.0
MIN_CENTER_TO_EDGE = 10.0
DEFAULT_E = 2100.0
DEFAULT_NU = 0.33
DEFAULT_U = 2
DEFAULT_MESH_SIZE = 2.5
E = DEFAULT_E
U = DEFAULT_U
MESH_SIZE = DEFAULT_MESH_SIZE
ELEMENT_TYPE = "CPS6"
INSTANCE = 1
REF = None
WARM_START = True
GROUP_INSTANCE_COUNTS = {
    2: 9,
    5: 7,
    8: 4,
}
MAX_ATTEMPTS_PER_HOLE = 200000
MAX_LAYOUT_RESTARTS = 1000

REFERENCE_SPECIMENS = {
    "transverse": "transverse_dense_perforated_plate",
    "longitudinal": "longitudinal_dense_perforated_plate",
}


VALID_REFS = (None, "solid", "transverse", "longitudinal")


GROUP_DEFINITIONS = [
    {
        "id": 1,
        "cluster": "low",
        "direction": "x",
        "x_range": (10.0, 140.0),
        "y_range": (40.0, 160.0),
    },
    {
        "id": 2,
        "cluster": "low",
        "direction": "none",
        "x_range": (17.5, 132.5),
        "y_range": (23.33333333333333, 176.66666666666669),
    },
    {
        "id": 3,
        "cluster": "low",
        "direction": "y",
        "x_range": (20.0, 130.0),
        "y_range": (10.0, 190.0),
    },
    {
        "id": 4,
        "cluster": "medium",
        "direction": "x",
        "x_range": (10.0, 140.0),
        "y_range": (55.0, 145.0),
    },
    {
        "id": 5,
        "cluster": "medium",
        "direction": "none",
        "x_range": (25.0, 125.0),
        "y_range": (33.33333333333333, 166.66666666666669),
    },
    {
        "id": 6,
        "cluster": "medium",
        "direction": "y",
        "x_range": (32.5, 117.5),
        "y_range": (15.0, 185.0),
    },
    {
        "id": 7,
        "cluster": "high",
        "direction": "x",
        "x_range": (10.0, 140.0),
        "y_range": (67.5, 132.5),
    },
    {
        "id": 8,
        "cluster": "high",
        "direction": "none",
        "x_range": (31.25, 118.75),
        "y_range": (41.66666666666667, 158.33333333333331),
    },
    {
        "id": 9,
        "cluster": "high",
        "direction": "y",
        "x_range": (42.5, 107.5),
        "y_range": (10.0, 190.0),
    },
]


def project_root():
    return _PROJECT_ROOT


def fem_root():
    return _FEM_ROOT


def temp_dir():
    return _TEMP_DIR


def inp_dir():
    return _INP_DIR


def test_work_dir():
    return _TEST_WORK_DIR


def reference_hole_csv_path():
    return os.path.join(
        project_root(), "1_samples", "CAD drawing", "sample_hole_coordinates.csv"
    )


def enter_temp_work_dir():
    path = temp_dir()
    ensure_directory(path)
    os.chdir(path)
    return path


def ensure_directory(path):
    if not path:
        return path
    if os.path.isdir(path):
        return path
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    return path


def write_json_file(path, data):
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    text = json.dumps(data, indent=2, sort_keys=True)
    if sys.version_info[0] < 3:
        text = text.decode("ascii")
    with io.open(path, "w", encoding="utf-8") as json_file:
        json_file.write(text)
        json_file.write(u"\n")
    return path


def read_json_file(path):
    with io.open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _seed_text_for_group(group, instance_index, restart_index):
    return "%s:%s:%s" % (group["id"], instance_index, restart_index)


def _seed_for_group(group, instance_index, restart_index):
    seed_text = _seed_text_for_group(group, instance_index, restart_index)
    seed_digest = hashlib.md5(seed_text.encode("ascii")).hexdigest()
    return int(seed_digest[:12], 16)


def _is_far_enough_from_existing(candidate, holes):
    min_distance_sq = MIN_CENTER_DISTANCE * MIN_CENTER_DISTANCE
    for hole in holes:
        dx = candidate["x"] - hole["x"]
        dy = candidate["y"] - hole["y"]
        if dx * dx + dy * dy < min_distance_sq:
            return False
    return True


def _generate_holes_for_group_with_metadata(group, instance_index):
    xmin, xmax = group["x_range"]
    ymin, ymax = group["y_range"]
    best_count = 0

    for restart_index in range(MAX_LAYOUT_RESTARTS):
        seed = _seed_for_group(group, instance_index, restart_index)
        rng = random.Random(seed)
        holes = []

        while len(holes) < HOLE_COUNT:
            accepted = False
            for _attempt in range(MAX_ATTEMPTS_PER_HOLE):
                candidate = {
                    "x": rng.uniform(xmin, xmax),
                    "y": rng.uniform(ymin, ymax),
                    "r": HOLE_RADIUS,
                }
                if _is_far_enough_from_existing(candidate, holes):
                    holes.append(candidate)
                    accepted = True
                    break
            if not accepted:
                break

        if len(holes) == HOLE_COUNT:
            validate_holes(holes)
            return holes, {
                "seed": seed,
                "seed_text": _seed_text_for_group(group, instance_index, restart_index),
                "restart_index": restart_index,
            }

        if len(holes) > best_count:
            best_count = len(holes)

    raise ValueError(
        "could not generate group %d instance %s: accepted %d holes"
        % (group["id"], instance_index, best_count)
    )


def generate_holes_for_group(group, instance_index):
    holes, _metadata = _generate_holes_for_group_with_metadata(group, instance_index)
    return holes


def _open_csv_read(path):
    if sys.version_info[0] < 3:
        return open(path, "rb")
    return open(path, "r", newline="")


def load_reference_holes(ref):
    if ref not in REFERENCE_SPECIMENS:
        raise ValueError("unknown reference hole layout: %s" % ref)

    specimen = REFERENCE_SPECIMENS[ref]
    holes = []
    with _open_csv_read(reference_hole_csv_path()) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["specimen"] != specimen:
                continue
            holes.append({
                "x": float(row["y_mm"]),
                "y": float(row["x_mm"]),
                "r": HOLE_RADIUS,
            })

    validate_holes(holes)
    return holes


def _group_manifest_entry(group):
    return {
        "id": group["id"],
        "cluster": group["cluster"],
        "direction": group["direction"],
        "x_range": [group["x_range"][0], group["x_range"][1]],
        "y_range": [group["y_range"][0], group["y_range"][1]],
    }


def _inp_path(inp_name):
    return os.path.abspath(os.path.join(inp_dir(), inp_name))


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


def _parse_random_inp_name(name):
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
        parsed = _parse_random_inp_name(name)
        if parsed is None:
            continue
        group_id, instance_index = parsed
        if group_id in offsets and instance_index > offsets[group_id]:
            offsets[group_id] = instance_index
    return offsets


def build_run_plan(ref=None, instance_count=1, warm_start=False):
    if ref not in VALID_REFS:
        raise ValueError("unknown run plan reference: %s" % ref)

    plan = []
    if ref is None:
        group_instance_counts = _normalize_group_instance_counts(instance_count)
        offsets = {}
        if warm_start:
            offsets = existing_group_instance_offsets()
        for group in GROUP_DEFINITIONS:
            first_index = offsets.get(group["id"], 0) + 1
            last_index = offsets.get(group["id"], 0) + group_instance_counts[group["id"]]
            for instance_index in range(first_index, last_index + 1):
                holes, metadata = _generate_holes_for_group_with_metadata(
                    group, instance_index
                )
                inp_name = "%d_%d_plate.inp" % (group["id"], instance_index)
                item = {
                    "ref": None,
                    "group_id": group["id"],
                    "cluster": group["cluster"],
                    "direction": group["direction"],
                    "instance_index": instance_index,
                    "holes": holes,
                    "inp_name": inp_name,
                    "inp_path": _inp_path(inp_name),
                    "seed": metadata["seed"],
                    "seed_text": metadata["seed_text"],
                    "restart_index": metadata["restart_index"],
                }
                plan.append(item)
        return plan

    instance_total = _coerce_instance_count(instance_count, allow_zero=False)
    for instance_index in range(1, instance_total + 1):
        if ref == "solid":
            inp_name = "solid_%d_plate.inp" % instance_index
            plan.append({
                "ref": ref,
                "instance_index": instance_index,
                "holes": [],
                "inp_name": inp_name,
                "inp_path": _inp_path(inp_name),
            })
        else:
            inp_name = "%s_%d_plate.inp" % (ref, instance_index)
            plan.append({
                "ref": ref,
                "instance_index": instance_index,
                "holes": load_reference_holes(ref),
                "inp_name": inp_name,
                "inp_path": _inp_path(inp_name),
            })
    return plan


def build_manifest(
    plan,
    material_e=DEFAULT_E,
    displacement_u=DEFAULT_U,
    mesh_size=DEFAULT_MESH_SIZE,
):
    groups = []
    sampling_domains = []
    for group in GROUP_DEFINITIONS:
        entry = _group_manifest_entry(group)
        groups.append(entry)
        sampling_domains.append({
            "group_id": group["id"],
            "cluster": group["cluster"],
            "direction": group["direction"],
            "x_range": [group["x_range"][0], group["x_range"][1]],
            "y_range": [group["y_range"][0], group["y_range"][1]],
        })

    runs = []
    seeds = []
    for item in plan:
        run = {
            "inp_name": item["inp_name"],
            "inp_path": item["inp_path"],
            "instance_index": item["instance_index"],
            "ref": item.get("ref"),
            "hole_count": len(item["holes"]),
        }
        if "group_id" in item:
            run["group_id"] = item["group_id"]
            run["cluster"] = item["cluster"]
            run["direction"] = item["direction"]
        if "seed" in item:
            run["seed"] = item["seed"]
            run["seed_text"] = item["seed_text"]
            run["restart_index"] = item["restart_index"]
            seeds.append({
                "inp_name": item["inp_name"],
                "group_id": item["group_id"],
                "instance_index": item["instance_index"],
                "seed": item["seed"],
                "seed_text": item["seed_text"],
                "restart_index": item["restart_index"],
            })
        runs.append(run)

    return {
        "plate": {
            "x": PLATE_X,
            "y": PLATE_Y,
            "thickness": PLATE_THICKNESS,
        },
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
        "references": REFERENCE_SPECIMENS,
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


def validate_holes(holes):
    if len(holes) != HOLE_COUNT:
        raise ValueError("expected 40 holes")

    xmin = MIN_CENTER_TO_EDGE
    xmax = PLATE_X - MIN_CENTER_TO_EDGE
    ymin = MIN_CENTER_TO_EDGE
    ymax = PLATE_Y - MIN_CENTER_TO_EDGE

    for index, hole in enumerate(holes):
        x = hole["x"]
        y = hole["y"]
        if x < xmin or x > xmax or y < ymin or y > ymax:
            raise ValueError("hole center outside allowed bounds at index %d" % index)

    min_distance_sq = MIN_CENTER_DISTANCE * MIN_CENTER_DISTANCE
    for left_index in range(len(holes)):
        left = holes[left_index]
        for right_index in range(left_index + 1, len(holes)):
            right = holes[right_index]
            dx = left["x"] - right["x"]
            dy = left["y"] - right["y"]
            if dx * dx + dy * dy < min_distance_sq:
                raise ValueError(
                    "hole centers too close at indices %d and %d"
                    % (left_index, right_index)
                )
    return None


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
    return name


def _move_generated_inp(source_path, target_path):
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
    return _move_generated_inp(job_name + ".inp", run_item["inp_path"])


def main():
    ensure_directory(inp_dir())
    if REF is None:
        instance_count = GROUP_INSTANCE_COUNTS
    else:
        instance_count = INSTANCE
    plan = build_run_plan(
        ref=REF,
        instance_count=instance_count,
        warm_start=(REF is None and WARM_START),
    )
    enter_temp_work_dir()
    for run_item in plan:
        saved_path = write_inp_with_abaqus(run_item, E, U, MESH_SIZE)
        if saved_path is None:
            saved_path = run_item["inp_path"]
        print("Saved inp: %s" % saved_path)

    if REF is not None:
        return None

    manifest = build_manifest(
        plan,
        material_e=E,
        displacement_u=U,
        mesh_size=MESH_SIZE,
    )
    manifest_path = os.path.join(inp_dir(), "group_manifest.json")
    if REF is None and WARM_START:
        manifest = merge_manifest(load_existing_manifest(manifest_path), manifest)
    write_json_file(manifest_path, manifest)
    return manifest_path


if __name__ == "__main__":
    main()
