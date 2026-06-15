from __future__ import print_function

import csv
import io
import json
import os


DEFAULT_STEP = "Load"
DEFAULT_TOP_SET = "TopEdge"
DEFAULT_BOTTOM_SET = "BottomEdge"
DEFAULT_INSTANCE = "PlateInstance"
DEFAULT_COMPONENT = 2

# Script configuration. Abaqus is expected to run with cwd at 2_FEM/temp.
ODB_PATHS = [
    os.path.join("odb", "uniform_1_plate.odb"),
]
FRAMES = "all"  # Examples: [0, 1, -1], or "all"
STEP_NAME = DEFAULT_STEP
TOP_SET = DEFAULT_TOP_SET
BOTTOM_SET = DEFAULT_BOTTOM_SET
INSTANCE_NAME = DEFAULT_INSTANCE
COMPONENT = DEFAULT_COMPONENT
OUTPUT_DIR = os.path.join("..", "results", "stiffness")
WRITE_JSON = True
WRITE_CSV = True


def _case_lookup(mapping, name):
    if name in mapping:
        return mapping[name]
    target = name.lower()
    for key in mapping.keys():
        if str(key).lower() == target:
            return mapping[key]
    raise KeyError(name)


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


def _frame(odb, step_name, frame_index):
    step = _step(odb, step_name)
    if not step.frames:
        raise ValueError("step has no frames: %s" % step_name)
    try:
        return step.frames[frame_index]
    except IndexError:
        raise ValueError("frame index out of range: %s" % frame_index)


def _step(odb, step_name):
    try:
        return _case_lookup(odb.steps, step_name)
    except KeyError:
        raise ValueError("step not found: %s" % step_name)


def _field_output(frame, field_name):
    try:
        return _case_lookup(frame.fieldOutputs, field_name)
    except KeyError:
        raise ValueError(
            "ODB frame has no field output %s. "
            "Make sure the analysis requested %s output." % (field_name, field_name)
        )


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


def normalize_frame_indices(odb, step_name, frames):
    step = _step(odb, step_name)
    frame_count = len(step.frames)
    if frame_count < 1:
        raise ValueError("step has no frames: %s" % step_name)

    try:
        string_types = (basestring,)
    except NameError:
        string_types = (str,)

    if isinstance(frames, string_types):
        if frames.lower() != "all":
            raise ValueError('FRAMES string must be "all"')
        return list(range(frame_count))

    if isinstance(frames, int):
        frames = [frames]

    normalized = []
    for frame_index in frames:
        index = int(frame_index)
        if index < 0:
            index = frame_count + index
        if index < 0 or index >= frame_count:
            raise ValueError("frame index out of range: %s" % frame_index)
        normalized.append(index)
    return normalized


def compute_stiffness_for_frames(
    odb,
    step_name=DEFAULT_STEP,
    frame_indices=None,
    top_set=DEFAULT_TOP_SET,
    bottom_set=DEFAULT_BOTTOM_SET,
    instance_name=DEFAULT_INSTANCE,
    component=DEFAULT_COMPONENT,
):
    if frame_indices is None:
        frame_indices = [-1]
    resolved_indices = normalize_frame_indices(odb, step_name, frame_indices)
    if isinstance(frame_indices, int):
        requested_indices = [frame_indices]
    elif isinstance(frame_indices, str):
        requested_indices = resolved_indices
    else:
        requested_indices = list(frame_indices)

    results = []
    for position, resolved_index in enumerate(resolved_indices):
        result = compute_equivalent_stiffness(
            odb,
            step_name=step_name,
            frame_index=resolved_index,
            top_set=top_set,
            bottom_set=bottom_set,
            instance_name=instance_name,
            component=component,
        )
        result["requested_frame_index"] = requested_indices[position]
        results.append(result)
    return results


def compute_equivalent_stiffness(
    odb,
    step_name=DEFAULT_STEP,
    frame_index=-1,
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
    total_reaction_magnitude = _mean([
        abs(float(top["total_reaction"])),
        abs(float(bottom["total_reaction"])),
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
        "total_reaction_magnitude": total_reaction_magnitude,
        "top_mean_displacement": float(top["mean_displacement"]),
        "bottom_mean_displacement": float(bottom["mean_displacement"]),
        "mean_reaction_magnitude": mean_reaction_magnitude,
        "mean_displacement_magnitude": mean_displacement_magnitude,
        "equivalent_stiffness": equivalent_stiffness,
    }
    if status != "ok":
        result["warning"] = "mean boundary displacement magnitude is zero"
    return result


def open_odb(path):
    try:
        from odbAccess import openOdb
    except ImportError:
        raise RuntimeError("odbAccess is unavailable. Run this script with Abaqus Python.")
    return openOdb(path=str(path), readOnly=True)


def write_result_json(result, output_path):
    output_path = str(output_path)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    try:
        unicode
    except NameError:
        text = payload
    else:
        text = unicode(payload)
    with io.open(output_path, "w", encoding="utf-8") as result_file:
        result_file.write(text)


CSV_FIELDS = [
    "odb_path",
    "status",
    "step",
    "requested_frame_index",
    "frame_index",
    "component",
    "top_set",
    "bottom_set",
    "instance",
    "top_node_count",
    "bottom_node_count",
    "top_mean_reaction",
    "bottom_mean_reaction",
    "top_total_reaction",
    "bottom_total_reaction",
    "total_reaction_magnitude",
    "top_mean_displacement",
    "bottom_mean_displacement",
    "mean_reaction_magnitude",
    "mean_displacement_magnitude",
    "equivalent_stiffness",
    "warning",
]


def write_result_csv(results, output_path):
    output_path = str(output_path)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(output_path, "w") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def print_result(result):
    print("Equivalent stiffness")
    print("--------------------")
    print("step: %s" % result["step"])
    print("status: %s" % result.get("status", "ok"))
    print("frame_index: %s" % result["frame_index"])
    print("component: %s" % result["component"])
    print("top_set: %s" % result["top_set"])
    print("bottom_set: %s" % result["bottom_set"])
    print("top_mean_reaction: %.9g" % result["top_mean_reaction"])
    print("bottom_mean_reaction: %.9g" % result["bottom_mean_reaction"])
    print("top_total_reaction: %.9g" % result["top_total_reaction"])
    print("bottom_total_reaction: %.9g" % result["bottom_total_reaction"])
    print("total_reaction_magnitude: %.9g" % result["total_reaction_magnitude"])
    print("top_mean_displacement: %.9g" % result["top_mean_displacement"])
    print("bottom_mean_displacement: %.9g" % result["bottom_mean_displacement"])
    print("mean_reaction_magnitude: %.9g" % result["mean_reaction_magnitude"])
    print("mean_displacement_magnitude: %.9g" % result["mean_displacement_magnitude"])
    if result["equivalent_stiffness"] is None:
        print("equivalent_stiffness: not available")
    else:
        print("equivalent_stiffness: %.9g" % result["equivalent_stiffness"])
    if result.get("warning"):
        print("warning: %s" % result["warning"])


def _absolute_from_cwd(path):
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.abspath(str(path))


def _odb_stem(odb_path):
    return os.path.splitext(os.path.basename(str(odb_path)))[0]


def run_configured_analysis(
    odb_paths=None,
    frames=None,
    step_name=STEP_NAME,
    top_set=TOP_SET,
    bottom_set=BOTTOM_SET,
    instance_name=INSTANCE_NAME,
    component=COMPONENT,
    output_dir=OUTPUT_DIR,
    write_json=WRITE_JSON,
    write_csv=WRITE_CSV,
    open_odb_func=None,
):
    if odb_paths is None:
        odb_paths = ODB_PATHS
    if frames is None:
        frames = FRAMES
    if open_odb_func is None:
        open_odb_func = open_odb

    output_dir = _absolute_from_cwd(output_dir)
    all_results = []
    for odb_path in odb_paths:
        absolute_odb_path = _absolute_from_cwd(odb_path)
        odb = open_odb_func(absolute_odb_path)
        try:
            results = compute_stiffness_for_frames(
                odb,
                step_name=step_name,
                frame_indices=frames,
                top_set=top_set,
                bottom_set=bottom_set,
                instance_name=instance_name,
                component=component,
            )
        finally:
            if hasattr(odb, "close"):
                odb.close()

        for result in results:
            result["odb_path"] = absolute_odb_path
            print_result(result)
        stem = _odb_stem(absolute_odb_path)
        if write_json:
            write_result_json(results, os.path.join(output_dir, stem + "_stiffness.json"))
        if write_csv:
            write_result_csv(results, os.path.join(output_dir, stem + "_stiffness.csv"))
        all_results.extend(results)
    return all_results


def main():
    run_configured_analysis()
    return 0


if __name__ == "__main__":
    main()
