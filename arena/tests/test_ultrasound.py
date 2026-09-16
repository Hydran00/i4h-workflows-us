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


def test_contact_gate_requires_measured_force_and_closes_on_separation():
    from i4h_arena.medical.ultrasound import contact_mask
    forces = np.zeros((3, 1, 1, 3))
    forces[1, 0, 0, 2] = 0.11
    forces[2, 0, 0, 2] = 0.09
    np.testing.assert_array_equal(contact_mask(forces), [False, True, False])
    forces[:] = 0
    assert not contact_mask(forces).any()


def test_closest_skin_surface_includes_triangle_interiors_and_edges(tmp_path):
    from i4h_arena.medical.ultrasound_worker import closest_surface_point, load_triangles
    mesh = tmp_path / 'Skin.obj'
    mesh.write_text('v 0 0 0\nv 100 0 0\nv 0 100 0\nf 1 2 3\n')
    triangles = load_triangles(mesh)
    for height in (4.9, 5.0, 5.1, -4.9):
        closest, distance = closest_surface_point([20, 20, height], triangles)
        np.testing.assert_allclose(closest, [20, 20, 0], atol=1e-12)
        assert distance == pytest.approx(abs(height))
        assert (distance < 5) == (abs(height) < 5)
    closest, distance = closest_surface_point([-3, -4, 0], triangles)
    np.testing.assert_allclose(closest, [0, 0, 0])
    assert distance == 5
    closest, distance = closest_surface_point([50, -2, 0], triangles)
    np.testing.assert_allclose(closest, [50, 0, 0])
    assert distance == 2


def test_fast_skin_proximity_queries_local_surface():
    from i4h_arena.medical.ultrasound_worker import SkinProximity, closest_surface_point
    # Uniform surface triangles: local query preserves face-interior distances.
    triangles = np.array([[[x, y, 0], [x + 1, y, 0], [x, y + 1, 0]]
                          for x in range(20) for y in range(20)], dtype=float)
    skin = SkinProximity(triangles)
    assert skin.neighbors == 16
    for point in ([10.2, 10.2, 4.9], [0.2, 0.2, 5.1], [-1, -1, 0]):
        actual_point, actual_distance = skin.closest(point)
        expected_point, expected_distance = closest_surface_point(point, triangles)
        np.testing.assert_allclose(actual_point, expected_point)
        assert actual_distance == pytest.approx(expected_distance)


def test_curved_probe_edge_can_remain_close_when_center_is_far():
    from i4h_arena.medical.ultrasound_worker import SkinProximity, acoustic_face_distance, acoustic_face_points
    triangles = np.array([[[-100, -100, 0], [100, -100, 0], [100, 100, 0]],
                          [[-100, -100, 0], [100, 100, 0], [-100, 100, 0]]], dtype=float)
    skin = SkinProximity(triangles)
    rotation = [0, np.pi / 6, 0]
    face = acoustic_face_points([0, 0, 0], rotation)
    position = [0, 0, 1 - face[:, 2].min()]
    assert skin.closest(position)[1] > 5
    assert acoustic_face_distance(skin, position, rotation) == pytest.approx(1)
    assert acoustic_face_distance(skin, [0, 0, 60], rotation) > 5
    assert len(face) == 51


def test_wrench_gate_any_signed_component_and_estimated_torque():
    from i4h_arena.medical.ultrasound import contact_wrench_mask
    force = np.zeros((5, 1, 1, 3))
    force[0, 0, 0, 0] = -2e-5
    force[1, 0, 0, 1] = 1e-5
    force[2, 0, 0, 1] = 9e-6
    points = np.zeros_like(force)
    points[2, 0, 0, 0] = 2  # moment 1.8e-5, although force is below threshold
    points[3] = np.nan  # no contact points when separated
    force[4, 0, 0, 2] = 2e-5
    points[4] = np.nan  # force alone still activates
    np.testing.assert_array_equal(contact_wrench_mask(force, points, np.zeros((5, 3))),
                                  [True, False, True, False, True])
