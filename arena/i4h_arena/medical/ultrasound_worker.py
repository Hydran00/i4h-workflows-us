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
            probe = rs.CurvilinearProbe(
                rs.Pose(
                    np.asarray(request["position_mm"], dtype=np.float32),
                    np.asarray(request["rotation_xyz_rad"], dtype=np.float32),
                )
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
