# SPDX-License-Identifier: Apache-2.0
"""OptiX worker, executed inside the ultrasound-simulator container.

One JSON request/reply per frame. Native-library stdout is redirected to stderr
so it cannot corrupt the protocol. All coordinates are in mesh millimetres.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path


def load_triangles(path):
    """Read the renderer's OBJ surface, preserving its millimetre coordinates."""
    import numpy as np
    vertices, faces = [], []
    with open(path) as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                ids = [int(x.split("/")[0]) for x in parts[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                faces.extend((ids[0], ids[j], ids[j + 1]) for j in range(1, len(ids) - 1))
    if not faces:
        raise ValueError("Skin mesh has no faces")
    return np.asarray(vertices, dtype=float)[np.asarray(faces)]


def closest_surface_point(point, triangles):
    """Exact closest point on triangle faces/edges, not just mesh vertices."""
    import numpy as np
    point = np.asarray(point, dtype=float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm2 = (normal * normal).sum(-1)
    projection = point - normal * (((point - a) * normal).sum(-1) / np.maximum(norm2, 1e-30))[:, None]
    v = projection - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (v * ab).sum(-1), (v * ac).sum(-1)
    denominator = np.maximum(norm2, 1e-30)
    u = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (norm2 > 1e-30) & (u >= 0) & (w >= 0) & (u + w <= 1)
    candidates = [projection]
    distances = [np.where(inside, ((projection - point)**2).sum(-1), np.inf)]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.clip(((point - start) * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-30), 0, 1)
        candidate = start + t[:, None] * edge
        candidates.append(candidate)
        distances.append(((candidate - point)**2).sum(-1))
    distances = np.asarray(distances)
    kind, face = np.unravel_index(np.argmin(distances), distances.shape)
    return candidates[kind][face], float(np.sqrt(distances[kind, face]))


class SkinProximity:
    """Fast local surface approximation: query only 16 nearby triangle centres."""

    def __init__(self, triangles):
        from scipy.spatial import cKDTree
        self.triangles = triangles
        self.tree = cKDTree(triangles.mean(axis=1))
        self.neighbors = min(16, len(triangles))

    def closest(self, point):
        import numpy as np
        _, indices = self.tree.query(point, k=self.neighbors)
        return closest_surface_point(point, self.triangles[np.atleast_1d(indices)])


# Match the geometry passed explicitly to CurvilinearProbe below (millimetres).
PROBE_RADIUS_MM = 45.0
PROBE_SECTOR_DEG = 73.0
PROBE_HEIGHT_MM = 7.0


def acoustic_face_points(position, rotation_xyz_rad):
    """Small grid on the convex emitting face, transformed into mesh coordinates."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    angles, elevation = np.meshgrid(
        np.linspace(-PROBE_SECTOR_DEG / 2, PROBE_SECTOR_DEG / 2, 17) * np.pi / 180,
        [-PROBE_HEIGHT_MM / 2, 0, PROBE_HEIGHT_MM / 2],
    )
    local = np.stack((PROBE_RADIUS_MM * np.sin(angles), elevation,
                      PROBE_RADIUS_MM * (np.cos(angles) - 1)), axis=-1).reshape(-1, 3)
    return Rotation.from_euler("xyz", rotation_xyz_rad).apply(local) + np.asarray(position)


def acoustic_face_distance(skin, position, rotation_xyz_rad):
    import numpy as np
    points = acoustic_face_points(position, rotation_xyz_rad)
    _, indices = skin.tree.query(points, k=skin.neighbors)
    triangles = skin.triangles[np.asarray(indices).reshape(-1)]
    # Evaluate all point/nearby-triangle pairs in one vectorized operation.
    return closest_surface_point(np.repeat(points, skin.neighbors, axis=0), triangles)[1]


def main() -> None:
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import numpy as np
    import ultrasound_simulator.cuda as rs

    materials = rs.Materials()
    world = rs.World("water")
    meshes = []
    for name, material in (
        ("Tumor1", "fat"),
        ("Tumor2", "water"),
        ("Liver", "liver"),
        ("Skin", "fat"),
        ("Bone", "bone"),
        ("Vessels", "water"),
        ("Gallbladder", "water"),
        ("Spleen", "liver"),
        ("Heart", "liver"),
        ("Stomach", "water"),
        ("Pancreas", "liver"),
        ("Small_bowel", "water"),
        ("Colon", "water"),
    ):
        path = Path("/opt/ultrasound-mesh") / f"{name}.obj"
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = rs.Mesh(str(path), materials.get_index(material))
        meshes.append(mesh)
        world.add(mesh)
    skin = None
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    params = rs.SimParams()
    params.conv_psf = True
    params.buffer_size = 4096
    params.t_far = 180.0
    params.enable_cuda_timing = False
    params.b_mode_size = (256, 256)
    print(json.dumps({"ready": True}), file=protocol)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("skin_distance_threshold_mm") is not None:
                if skin is None:
                    skin = SkinProximity(load_triangles("/opt/ultrasound-mesh/Skin.obj"))
                distance = acoustic_face_distance(skin, request["position_mm"], request["rotation_xyz_rad"])
                if not distance < request["skin_distance_threshold_mm"]:
                    print(json.dumps({"inactive": True, "skin_distance_mm": distance}), file=protocol)
                    continue
            probe = rs.CurvilinearProbe(
                rs.Pose(
                    np.asarray(request["position_mm"], dtype=np.float32),
                    np.asarray(request["rotation_xyz_rad"], dtype=np.float32),
                ),
                radius=PROBE_RADIUS_MM, sector_angle=PROBE_SECTOR_DEG,
                elevational_height=PROBE_HEIGHT_MM,
            )
            frame = np.asarray(simulator.simulate(probe, params), dtype="<f4")
            if frame.shape != (256, 256):
                raise ValueError("invalid ultrasound frame")
            # Upstream log compression divides by the echo quantile, which is
            # zero when the probe misses all tissue. Map those samples and the
            # outside-sector negative FLT_MAX sentinel to a finite noise floor.
            frame = np.maximum(np.nan_to_num(frame, nan=-120.0, posinf=-120.0, neginf=-120.0), -120.0)
            reply = {"shape": list(frame.shape), "db": base64.b64encode(frame.tobytes()).decode("ascii")}
        except (ValueError, KeyError, RuntimeError) as exc:
            reply = {"error": str(exc)}
        print(json.dumps(reply), file=protocol)


if __name__ == "__main__":
    main()
