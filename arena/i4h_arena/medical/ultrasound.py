# SPDX-License-Identifier: Apache-2.0
"""Pose conversion and an isolated real OptiX ultrasound renderer."""

from __future__ import annotations

import base64
import json
import selectors
import subprocess
import uuid
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def probe_in_mesh(probe_pos, probe_quat_xyzw, mesh_pos, mesh_quat_xyzw):
    """Isaac Lab world metre/xyzw poses -> mesh millimetres/extrinsic XYZ radians.

    Inputs are the existing ee_to_us_transform and mesh_to_organ_transform
    target world poses. OptiX make_rotation uses Rz @ Ry @ Rx.
    """
    values = np.concatenate([probe_pos, probe_quat_xyzw, mesh_pos, mesh_quat_xyzw])
    if not np.isfinite(values).all():
        raise ValueError("ultrasound transforms must be finite")
    probe = Rotation.from_quat(np.asarray(probe_quat_xyzw))
    mesh = Rotation.from_quat(np.asarray(mesh_quat_xyzw))
    position = mesh.inv().apply(np.asarray(probe_pos) - np.asarray(mesh_pos)) * 1000.0
    angles = (mesh.inv() * probe).as_euler("xyz")
    return position, angles


def bmode_rgb(frame: np.ndarray) -> np.ndarray:
    """Same fixed -60..0 dB display window as the upstream liver demo."""
    gray = np.rint(np.clip((frame + 60.0) / 60.0, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


class UltrasoundRenderer:
    """Own one disposable GPU container; no network service or host Python ABI coupling."""

    def __init__(self, image: str, gpu: str = "0", timeout: float = 120.0):
        self.timeout = timeout
        self.name = f"i4h-ultrasound-{uuid.uuid4().hex[:12]}"
        worker = Path(__file__).with_name("ultrasound_worker.py").resolve()
        self.process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                self.name,
                "--gpus",
                f"device={gpu}",
                "--network",
                "none",
                "--mount",
                f"type=bind,src={worker},dst=/ultrasound_worker.py,readonly",
                "--entrypoint",
                "python3",
                image,
                "-u",
                "/ultrasound_worker.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self._pending = bytearray()
        try:
            if self._receive() != {"ready": True}:
                raise RuntimeError("ultrasound worker did not initialize")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        import os
        import time

        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("ultrasound worker timed out; inspect its stderr log")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("ultrasound worker exited; inspect Docker/OptiX errors above")
                self._pending.extend(data)
                if len(self._pending) > 1024 * 1024:
                    raise RuntimeError("ultrasound reply exceeds frame limit")
        line, _, rest = self._pending.partition(b"\n")
        self._pending = bytearray(rest)
        return json.loads(line)

    def render(self, position_mm, rotation_xyz_rad) -> np.ndarray:
        request = json.dumps({"position_mm": list(position_mm), "rotation_xyz_rad": list(rotation_xyz_rad)})
        self.process.stdin.write((request + "\n").encode())
        reply = self._receive()
        if "error" in reply:
            raise RuntimeError(f"ultrasound: {reply['error']}")
        if reply.get("shape") != [256, 256]:
            raise ValueError("unexpected ultrasound frame shape")
        frame = np.frombuffer(base64.b64decode(reply["db"], validate=True), dtype="<f4").reshape(256, 256).copy()
        if not np.isfinite(frame).all():
            raise ValueError("nonfinite ultrasound frame")
        return frame

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "stop", "-t", "1", self.name], capture_output=True, timeout=10, check=False)
            self.process.wait(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()
