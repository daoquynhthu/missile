import argparse
import json
from pathlib import Path

import numpy as np
from stl import mesh


DEFAULT_CONFIG = {
    "nose_width": 0.030,
    "nose_height": 0.020,
    "nose_length": 0.020,
    "body_length": 0.190,
    "boattail_length": 0.055,
    "boattail_scale": 0.82,
    "superellipse_power": 4.0,
    "segments": 64,
    "nose_rings": 20,
    "fin_root_chord": 0.082,
    "fin_tip_chord": 0.038,
    "fin_span": 0.040,
    "fin_sweep_deg": 22.0,
    "fin_thickness": 0.0015,
    "fin_count": 4,
    "fin_x_offset": 0.0,
    "canard_root_chord": 0.022,
    "canard_tip_chord": 0.010,
    "canard_span": 0.018,
    "canard_sweep_deg": 18.0,
    "canard_thickness": 0.0012,
    "canard_count": 4,
    "canard_x_le": 0.060,
}


def profile_points(width, height, segments, power):
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    y = (width * 0.5) * np.sign(np.cos(angles)) * np.abs(np.cos(angles)) ** (2.0 / power)
    z = (height * 0.5) * np.sign(np.sin(angles)) * np.abs(np.sin(angles)) ** (2.0 / power)
    return np.column_stack([y, z])


def section_vertices(x_coord, width, height, segments, power):
    yz = profile_points(width, height, segments, power)
    x = np.full((segments, 1), x_coord)
    return np.hstack([x, yz])


def loft_between_sections(section_a, section_b):
    faces = []
    segments = len(section_a)
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, idx + segments])
        faces.append([nxt, nxt + segments, idx + segments])
    vertices = np.vstack([section_a, section_b])
    return vertices, faces


def add_end_cap(vertices, faces, section, reverse):
    center = np.array([[section[0, 0], 0.0, 0.0]])
    center_idx = len(vertices)
    vertices = np.vstack([vertices, center])
    base_offset = len(vertices) - 1 - len(section)
    for idx in range(len(section)):
        nxt = (idx + 1) % len(section)
        if reverse:
            faces.append([center_idx, base_offset + nxt, base_offset + idx])
        else:
            faces.append([center_idx, base_offset + idx, base_offset + nxt])
    return vertices, faces


def create_rounded_square_nose(width, height, length, segments, rings, power):
    ring_list = []
    for ring_idx in range(1, rings + 1):
        u = ring_idx / rings
        scale = np.sin(u * np.pi * 0.5)
        x_coord = length * (1.0 - np.cos(u * np.pi * 0.5))
        ring_list.append(section_vertices(x_coord, width * scale, height * scale, segments, power))

    vertices = ring_list[0]
    faces = []
    tip = np.array([[0.0, 0.0, 0.0]])
    vertices = np.vstack([tip, vertices])
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([0, idx + 1, nxt + 1])

    offset = 1
    prev = ring_list[0]
    for ring in ring_list[1:]:
        part_v, part_f = loft_between_sections(prev, ring)
        faces.extend((np.array(part_f) + offset).tolist())
        vertices = np.vstack([vertices, part_v[len(prev):]])
        offset += len(prev)
        prev = ring
    return vertices, faces, ring_list[-1]


def create_body_sections(start_x, body_length, boattail_length, width, height, boattail_scale, segments, power):
    sections = [section_vertices(start_x, width, height, segments, power)]
    body_end_x = start_x + body_length
    sections.append(section_vertices(body_end_x, width, height, segments, power))
    tail_end_x = body_end_x + boattail_length
    sections.append(section_vertices(tail_end_x, width * boattail_scale, height * boattail_scale, segments, power))
    return sections


def build_sections_mesh(sections):
    vertices = sections[0]
    faces = []
    offset = 0
    prev = sections[0]
    for section in sections[1:]:
        part_v, part_f = loft_between_sections(prev, section)
        faces.extend((np.array(part_f) + offset).tolist())
        vertices = np.vstack([vertices, part_v[len(prev):]])
        offset += len(prev)
        prev = section
    vertices, faces = add_end_cap(vertices, faces, sections[-1], True)
    return vertices, faces


def create_trapezoidal_fin(root_chord, tip_chord, span, sweep_deg, thickness, offset_x, angle_deg):
    dx_tip = span * np.tan(np.radians(sweep_deg))
    local_vertices = np.array([
        [0.0, -thickness * 0.5, 0.0],
        [root_chord, -thickness * 0.5, 0.0],
        [root_chord, thickness * 0.5, 0.0],
        [0.0, thickness * 0.5, 0.0],
        [dx_tip, -thickness * 0.5, span],
        [dx_tip + tip_chord, -thickness * 0.5, span],
        [dx_tip + tip_chord, thickness * 0.5, span],
        [dx_tip, thickness * 0.5, span],
    ])
    rotation = np.radians(angle_deg)
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rotation), -np.sin(rotation)],
        [0.0, np.sin(rotation), np.cos(rotation)],
    ])
    vertices = local_vertices @ rot.T
    vertices[:, 0] += offset_x
    faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ]
    return vertices, faces


def surface_offset(width, height, angle_deg, power):
    angle = np.radians(angle_deg)
    y_coord = (width * 0.5) * np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** (2.0 / power)
    z_coord = (height * 0.5) * np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** (2.0 / power)
    return y_coord, z_coord


def append_surface_group(groups_v, groups_f, vertices, faces, current_count):
    groups_v.append(vertices)
    groups_f.append(np.array(faces) + current_count)
    return current_count + len(vertices)


def load_config(config_path):
    config = dict(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def generate_dart_mesh(config):
    nose_width = float(config["nose_width"])
    nose_height = float(config["nose_height"])
    nose_length = float(config["nose_length"])
    body_length = float(config["body_length"])
    boattail_length = float(config["boattail_length"])
    boattail_scale = float(config["boattail_scale"])
    power = float(config["superellipse_power"])
    segments = int(config["segments"])
    rings = int(config["nose_rings"])

    nose_v, nose_f, nose_base = create_rounded_square_nose(
        nose_width, nose_height, nose_length, segments, rings, power
    )
    sections = create_body_sections(
        nose_base[0, 0], body_length, boattail_length,
        nose_width, nose_height, boattail_scale, segments, power
    )
    body_v, body_f = build_sections_mesh([nose_base] + sections[1:])

    groups_v = [nose_v, body_v]
    groups_f = [np.array(nose_f), np.array(body_f) + len(nose_v)]
    current_count = len(nose_v) + len(body_v)

    total_length = nose_length + body_length + boattail_length
    fin_offset_x = total_length - float(config["fin_root_chord"]) - float(config.get("fin_x_offset", 0.0))
    for fin_idx in range(int(config["fin_count"])):
        angle_deg = 360.0 * fin_idx / int(config["fin_count"])
        fin_v, fin_f = create_trapezoidal_fin(
            float(config["fin_root_chord"]),
            float(config["fin_tip_chord"]),
            float(config["fin_span"]),
            float(config["fin_sweep_deg"]),
            float(config["fin_thickness"]),
            fin_offset_x,
            angle_deg,
        )
        y_off, z_off = surface_offset(nose_width, nose_height, angle_deg, power)
        fin_v[:, 1] += y_off
        fin_v[:, 2] += z_off
        current_count = append_surface_group(groups_v, groups_f, fin_v, fin_f, current_count)

    if int(config["canard_count"]) > 0 and float(config["canard_span"]) > 0.0:
        for canard_idx in range(int(config["canard_count"])):
            angle_deg = 360.0 * canard_idx / int(config["canard_count"])
            canard_v, canard_f = create_trapezoidal_fin(
                float(config["canard_root_chord"]),
                float(config["canard_tip_chord"]),
                float(config["canard_span"]),
                float(config["canard_sweep_deg"]),
                float(config["canard_thickness"]),
                float(config["canard_x_le"]),
                angle_deg,
            )
            y_off, z_off = surface_offset(nose_width, nose_height, angle_deg, power)
            canard_v[:, 1] += y_off
            canard_v[:, 2] += z_off
            current_count = append_surface_group(groups_v, groups_f, canard_v, canard_f, current_count)

    return np.vstack(groups_v), np.vstack(groups_f)


def save_stl(vertices, faces, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dart_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for face_idx, face in enumerate(faces):
        for vert_idx in range(3):
            dart_mesh.vectors[face_idx][vert_idx] = vertices[face[vert_idx], :]
    dart_mesh.save(str(output_path))
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", default="", help="geometry config json path")
    parser.add_argument("--output", default="dart_v5_full_square.stl", help="output STL path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_json if args.config_json else None)
    vertices, faces = generate_dart_mesh(config)
    output_path = save_stl(vertices, faces, args.output)
    print(f"Generated STL: {output_path}")


if __name__ == "__main__":
    main()
