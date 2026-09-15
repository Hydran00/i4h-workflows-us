# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from i4h_common.types import Pose

from i4h_arena.adapters.actuation import ArenaActuation, RobotSlice


def test_seed_does_not_alias_the_home_joint_reference() -> None:
    home = np.array([[0.0, -1.6, 1.4, 1.5, -1.8, 0.0]], dtype=np.float32)
    actuation = ArenaActuation(num_envs=1, action_dim=6)

    actuation.seed(home)
    actuation.set_joint_targets(np.ones((1, 6), dtype=np.float32))

    np.testing.assert_allclose(home, [[0.0, -1.6, 1.4, 1.5, -1.8, 0.0]])
    np.testing.assert_allclose(actuation.numpy(), np.ones((1, 6), dtype=np.float32))


def test_hold_repeats_the_previous_joint_position_target() -> None:
    target = np.arange(6, dtype=np.float32).reshape(1, 6)
    actuation = ArenaActuation(num_envs=1, action_dim=6)
    actuation.set_joint_targets(target)
    actuation.tensor()
    actuation.set_joint_targets(np.zeros((1, 6), dtype=np.float32))

    actuation.hold()

    np.testing.assert_allclose(actuation.numpy(), target)


def test_hold_zeros_a_relative_cartesian_delta() -> None:
    actuation = ArenaActuation(
        num_envs=1,
        action_dim=6,
        action_space="ee_pose",
        slices=(RobotSlice("robot", 0, 6, gripper_index=None),),
        relative_ee=True,
    )
    actuation.set_ee_delta(np.ones((1, 6), dtype=np.float32))
    actuation.tensor()

    actuation.hold()

    np.testing.assert_allclose(actuation.numpy(), np.zeros((1, 6), dtype=np.float32))


def test_last_ee_target_reports_the_absolute_pose_not_the_delta() -> None:
    actuation = ArenaActuation(
        num_envs=1,
        action_dim=7,
        action_space="ee_pose",
        slices=(RobotSlice("robot", 0, 7, gripper_index=None),),
        relative_ee=False,
    )
    assert actuation.last_ee_target() is None

    pose = Pose(pos=np.array([[0.1, 0.2, 0.3]]), quat=np.array([[1.0, 0.0, 0.0, 0.0]]))
    actuation.set_ee_target(pose)

    target = actuation.last_ee_target()
    assert target is not None
    np.testing.assert_allclose(target.pos, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(target.quat, [[1.0, 0.0, 0.0, 0.0]])
    assert actuation.last_ee_target("other_robot") is None


def test_relative_orientation_uses_shortest_arc_for_either_quaternion_sign():
    from types import SimpleNamespace

    current = Pose(pos=np.zeros((1, 3)), quat=np.array([[1., 0, 0, 0]]))
    view = SimpleNamespace(tcp=lambda _: current)
    actuation = ArenaActuation(num_envs=1, action_dim=6, action_space='ee_pose',
                               slices=(RobotSlice('robot', 0, 6, gripper_index=None),),
                               relative_ee=True, view=view)
    angle = np.radians(30)
    quaternion = np.array([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
    for sign in (1, -1):
        actuation.set_ee_target(Pose(pos=np.array([[.01, .02, .03]]), quat=sign * quaternion))
        np.testing.assert_allclose(actuation.numpy(), [[.01, .02, .03, 0, 0, angle]], atol=1e-7)
