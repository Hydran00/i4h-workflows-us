# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from i4h_arena.adapters.scene_view import ArenaSceneView


class _ArrayView:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def numpy(self):
        return self._value


class _XformAsset:
    def get_world_poses(self):
        return _ArrayView([[1.0, 2.0, 3.0]]), _ArrayView([[0.1, 0.2, 0.3, 0.9]])


class _Robot:
    data = SimpleNamespace(
        root_pos_w=np.array([[0.0, -2.4, 0.7923]], dtype=np.float32),
        root_quat_w=np.array([[0.0, 0.0, 0.7071068, 0.7071068]], dtype=np.float32),
        root_lin_vel_w=np.array([[0.0, 0.1, 0.0]], dtype=np.float32),
    )


def test_object_reads_static_xform_world_pose():
    scene = {"tray": _XformAsset()}
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=1))

    state = ArenaSceneView(env, objects=("tray",)).object("tray")

    np.testing.assert_allclose(state.pose.pos, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(state.pose.quat, [[0.9, 0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(state.lin_vel, np.zeros((1, 3), dtype=np.float32))


def test_robot_root_reads_floating_base_state():
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": _Robot()}, num_envs=1))

    state = ArenaSceneView(env).robot_root()

    np.testing.assert_allclose(state.pose.pos, [[0.0, -2.4, 0.7923]])
    np.testing.assert_allclose(state.pose.quat, [[0.7071068, 0.0, 0.0, 0.7071068]])
    np.testing.assert_allclose(state.lin_vel, [[0.0, 0.1, 0.0]])


def test_contact_prefers_filtered_force_matrix():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.array([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),
            net_forces_w=np.array([[[0.0, 0.0, 100.0]]], dtype=np.float32),
        )
    )
    scene = SimpleNamespace(sensors={"contact_robot_table": sensor})
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=1))

    assert not ArenaSceneView(env).contact("robot", "table").any()


def test_contact_aggregates_filtered_sensor_family():
    quiet = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.zeros((1, 1, 1, 3), dtype=np.float32),
            net_forces_w=np.zeros((1, 1, 3), dtype=np.float32),
        )
    )
    touching = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=np.array([[[[0.0, 0.0, 2.0]]]], dtype=np.float32),
            net_forces_w=np.zeros((1, 1, 3), dtype=np.float32),
        )
    )
    scene = SimpleNamespace(
        sensors={
            "contact_robot_table__pelvis": quiet,
            "contact_robot_table__left_hand": touching,
        }
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=1))

    assert ArenaSceneView(env).contact("table", "robot").all()


def test_scene_owned_footprint_is_batched():
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={}, num_envs=2))
    view = ArenaSceneView(env, footprint_half_extents={"table": (0.64, 0.4)})

    np.testing.assert_allclose(
        view.footprint_half_extents("table"),
        [[0.64, 0.4], [0.64, 0.4]],
    )


def test_camera_reports_the_sensors_own_frame_id():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={"rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            frame_id=np.array([7], dtype=np.int64),
        )
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"ultrasound": sensor}, num_envs=1))

    frame = ArenaSceneView(env, cameras=("ultrasound",)).camera("ultrasound")

    assert frame is not None
    assert frame.frame_num == 7


def test_camera_defaults_frame_id_to_zero_without_a_sensor_clock():
    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)}))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"room_camera": sensor}, num_envs=1))

    frame = ArenaSceneView(env, cameras=("room",), camera_aliases={"room": "room_camera"}).camera("room")

    assert frame is not None
    assert frame.frame_num == 0


def test_phantom_recording_reads_world_calibrated_frames():
    organ = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=np.array([[1., 2., 3.]]), root_quat_w=np.array([[0., 0., 0., 1.]])))
    mesh = SimpleNamespace(data=SimpleNamespace(
        target_pos_w=np.array([[[4., 5., 6.]]]),
        target_quat_w=np.array([[[1., 0., 0., 0.]]])))
    probe = SimpleNamespace(data=SimpleNamespace(
        target_pos_w=np.array([[[7., 8., 9.]]]),
        target_quat_w=np.array([[[0., 0., 0., 1.]]])))
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        scene={'organs': organ, 'mesh_to_organ_transform': mesh, 'ee_to_us_transform': probe},
        common_step_counter=12, step_dt=.02))
    state = ArenaSceneView(env).phantom_recording_state()
    np.testing.assert_allclose(state['phantom_pose'], [1, 2, 3, 1, 0, 0, 0])
    np.testing.assert_allclose(state['mesh_pose'], [4, 5, 6, 0, 1, 0, 0])
    np.testing.assert_allclose(state['ultrasound_probe_pose'], [7, 8, 9, 1, 0, 0, 0])
    assert state['timestamps'] == .24
    mesh.data.target_pos_w[:] = 0
    assert state['mesh_pose'][0] == 4  # snapshots survive simulator buffer updates
