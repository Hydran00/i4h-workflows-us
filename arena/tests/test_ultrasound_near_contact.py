"""Exercise the sensor acquisition gate without launching Isaac or Docker."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_missing_contact_still_acquires_near_skin():
    path = Path(__file__).parents[1] / "i4h_arena/sensors/ultrasound.py"
    tree = ast.parse(path.read_text())
    sensor_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UltrasoundSensor")
    update = next(node for node in sensor_class.body if isinstance(node, ast.FunctionDef) and node.name == "_update_buffers_impl")

    class Renderer:
        def __init__(self, *_args):
            self.thresholds = []

        def render(self, _position, _angles, *, skin_distance_threshold_m):
            self.thresholds.append(skin_distance_threshold_m)
            return np.ones((2, 2), dtype=np.float32) if skin_distance_threshold_m == 0.03 else None

    namespace = {
        "np": np, "torch": torch, "logging": __import__("logging"),
        "wp": SimpleNamespace(to_torch=lambda mask: mask),
        "to_torch": lambda value: value,
        "probe_in_mesh": lambda *_args: ([0, 0, 0], [0, 0, 0]),
        "bmode_rgb": lambda frame: np.repeat(frame[..., None], 3, axis=-1).astype(np.uint8),
        "UltrasoundRenderer": Renderer,
    }
    exec(compile(ast.Module(body=[update], type_ignores=[]), str(path), "exec"), namespace)
    pose = SimpleNamespace(target_pos_w=torch.zeros((1, 1, 3)), target_quat_w=torch.tensor([[[0., 0., 0., 1.]]]))
    scene = {
        "ee_to_us_transform": SimpleNamespace(data=pose),
        "mesh_to_organ_transform": SimpleNamespace(data=pose),
        "ee_frame": SimpleNamespace(data=pose),
        "contact_probe_organs": SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.zeros((1, 1, 1, 3)),
            contact_pos_w=torch.zeros((1, 1, 1, 3)),
        )),
    }
    sensor = SimpleNamespace(
        _scene=scene, _renderer=None, _device="cpu",
        cfg=SimpleNamespace(activation_height_m=None, contact_sensor="contact_probe_organs",
                            contact_force_threshold_n=1e-5, skin_distance_threshold_m=0.03,
                            image="unused", gpu="0"),
        _data=SimpleNamespace(output={"rgb": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
                                          "bmode_db": torch.zeros((1, 2, 2, 1))},
                              frame_id=torch.zeros(1, dtype=torch.int64)),
    )
    namespace["_update_buffers_impl"](sensor, torch.tensor([True]))
    assert sensor._renderer.thresholds == [0.03]
    assert sensor._data.frame_id.tolist() == [1]
    assert sensor._data.output["rgb"].any()
