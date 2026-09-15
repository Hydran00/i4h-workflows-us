# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Isaac/OptiX coordinate and display boundaries."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from i4h_arena.medical.ultrasound import bmode_rgb, probe_in_mesh


def xyzw(rotation):
    return rotation.as_quat()


def test_probe_pose_is_relative_to_rotated_translated_anatomy():
    mesh = Rotation.from_euler("z", 90, degrees=True)
    relative = Rotation.from_euler("xyz", [0.2, -0.4, 0.8])
    origin = np.array([1.0, 2.0, 3.0])
    position, angles = probe_in_mesh(
        origin + mesh.apply([0.01, 0.02, 0.03]),
        xyzw(mesh * relative),
        origin,
        xyzw(mesh),
    )
    np.testing.assert_allclose(position, [10, 20, 30], atol=1e-10)
    np.testing.assert_allclose(Rotation.from_euler("xyz", angles).as_matrix(), relative.as_matrix(), atol=1e-10)


def test_moving_entire_scene_preserves_ultrasound_pose():
    rotation = Rotation.from_euler("xyz", [0.4, 0.3, -0.2])
    identity = [0, 0, 0, 1]
    initial = probe_in_mesh([0.2, 0.1, 0.3], identity, [0.1, 0, 0], identity)
    shift = np.array([4.0, 3.0, 2.0])
    moved = probe_in_mesh(
        rotation.apply([0.2, 0.1, 0.3]) + shift, xyzw(rotation), rotation.apply([0.1, 0, 0]) + shift, xyzw(rotation)
    )
    np.testing.assert_allclose(initial, moved, atol=1e-10)


def test_invalid_pose_rejected():
    with pytest.raises(ValueError, match="finite"):
        probe_in_mesh([np.nan, 0, 0], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0, 1])


def test_bmode_window_clips_native_background_sentinel():
    rgb = bmode_rgb(np.array([[-np.finfo(np.float32).max, -60, -30, 0, 10]], dtype=np.float32))
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, :, 0], [0, 0, 128, 255, 255])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])


def test_ultrasound_is_optional_and_scene_configuration_is_idempotent():
    from types import SimpleNamespace

    from i4h_common.manifest import SceneSpec

    from i4h_arena.scenes.panda_phantom import PandaPhantomScene

    spec = SceneSpec(
        name="panda_phantom",
        impl="unused",
        embodiment="panda",
        action_space="ee_pose",
        dof=6,
        cameras=("room", "wrist"),
    )
    args = SimpleNamespace(ultrasound=False)
    scene = PandaPhantomScene(spec, args)
    scene.configure_args(args)
    assert scene.spec.cameras == ("room", "wrist")
    assert scene.default_sensor_views() == ()
    args.ultrasound = True
    scene.configure_args(args)
    scene.configure_args(args)
    assert scene.spec.cameras == ("room", "wrist", "ultrasound")
    assert scene.default_sensor_views() == ("ultrasound",)
    scene.close()  # No container was started during lightweight configuration.


def test_isaac_xyzw_identity_keeps_world_beam_direction():
    position, angles = probe_in_mesh([0, 0, 0.1], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0, 1])
    np.testing.assert_allclose(position, [0, 0, 100])
    np.testing.assert_allclose(angles, [0, 0, 0])


def test_isaac_xyzw_quarter_turn_rotates_mesh_translation():
    q = np.sqrt(0.5)
    position, angles = probe_in_mesh([0, 0, 0.1], [q, 0, 0, q], [0, 0, 0], [q, 0, 0, q])
    np.testing.assert_allclose(position, [0, 100, 0], atol=1e-10)
    np.testing.assert_allclose(angles, [0, 0, 0], atol=1e-10)
