# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Servo the end effector to a Cartesian target and hold until it arrives.

Success requires the measured TCP to be within tolerance: a commanded pose
is not an achieved pose, and treating the two as equal is how a workflow
continues with the tool nowhere near where it believes it is."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from i4h_common.types import Pose, quat_mul
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext

logger = logging.getLogger("i4h_tasks.ik.move_to_pose")


def slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc interpolation between batched ``wxyz`` quaternions.

    Nlerp with a hemisphere fix rather than true slerp: for the small
    orientation deltas these nodes command, the difference is below joint
    resolution and this has no trig and no near-parallel singularity.
    """
    b = np.where((np.sum(a * b, axis=-1, keepdims=True) < 0.0), -b, b)
    out = a * (1.0 - alpha) + b * alpha
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.divide(out, norm, out=np.tile(np.array([1.0, 0, 0, 0], np.float32), (len(out), 1)), where=norm > 1e-8)


class MoveToPose(Task):
    """Servo the end effector to a Cartesian target and hold until it arrives.

    Success requires the measured TCP to be within ``position_tolerance`` — a
    commanded pose is not an achieved pose, and treating the two as equal is how
    a workflow silently continues with the tool nowhere near where it thinks it is.
    """

    requires = {"action_space": "ee_pose"}

    @dataclass
    class Inputs:
        target: Pose
        offset: Pose | None = None

    @dataclass
    class Outputs:
        reached: bool = False
        tcp: Pose | None = None

    def __init__(
        self,
        *,
        duration_s: float = 1.0,
        position_tolerance: float = 0.005,
        orientation_tolerance: float | None = None,
        interpolate_orientation: bool = True,
        settle_timeout_s: float = 2.0,
        position_stall_timeout_s: float | None = None,
        robot: str = "robot",
        gripper: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.duration_s = duration_s
        self.position_tolerance = position_tolerance
        #: Radians; None (default) keeps the original position-only check, so
        #: existing callers are unaffected. Measured via the quaternion dot
        #: product, so a near-antipodal q/-q pair still reads as ~0 error.
        self.orientation_tolerance = orientation_tolerance
        self.interpolate_orientation = interpolate_orientation
        self.settle_timeout_s = settle_timeout_s
        if position_stall_timeout_s is not None and (
            not np.isfinite(position_stall_timeout_s) or position_stall_timeout_s <= 0
        ):
            raise ValueError("position_stall_timeout_s must be positive and finite")
        self.position_stall_timeout_s = position_stall_timeout_s
        self.robot = robot
        self.gripper = gripper
        self._start: Pose | None = None
        self._goal: Pose | None = None
        self._steps = 1
        self._tick = 0
        self._reached = False

    def on_enter(self, ctx: TickContext, inputs: Inputs) -> None:
        target = getattr(inputs, "target", None)
        if target is None:
            raise ValueError(f"{self.name}: no target pose; wire it from a locate node")
        offset = getattr(inputs, "offset", None)
        if offset is not None:
            target = Pose(pos=target.pos + offset.pos, quat=quat_mul(target.quat, offset.quat))
        self._start = ctx.scene.tcp(self.robot)
        self._goal = self._resolve_goal(target)
        self._steps = max(1, round(self.duration_s / ctx.dt))
        self._tick = 0
        self._reached = False
        self._position_best = None
        self._position_stale_steps = None
        self._waiting_for = None
        logger.debug(
            "%s target: start=%s goal=%s",
            self.name,
            self._start.pos.round(5).tolist(),
            self._goal.pos.round(5).tolist(),
        )

    def _resolve_goal(self, target: Pose) -> Pose:
        return target

    def tick(self, ctx: TickContext) -> Status:
        assert self._start is not None and self._goal is not None
        self._tick += 1
        alpha = min(1.0, self._tick / self._steps)
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        quat = (self._goal.quat if not self.interpolate_orientation or eased >= 1.0
                else slerp(self._start.quat, self._goal.quat, eased))
        command = Pose(
            pos=self._start.pos + (self._goal.pos - self._start.pos) * eased,
            # At the endpoint preserve the target's exact quaternion sign.
            # q and -q are the same rotation, but IsaacLab's DLS orientation
            # error follows the supplied sign and the controller sends
            # the command-manager quaternion unchanged.
            quat=quat,
        )
        ctx.act.set_ee_target(command, self.robot)
        if self.gripper is not None:
            ctx.act.set_gripper(self.gripper, self.robot)

        if self._tick < self._steps:
            return Status.RUNNING

        current = ctx.scene.tcp(self.robot)
        error = current.distance_to(self._goal)
        orientation_error = self._orientation_error_rad(current)
        position_ok = bool((error < self.position_tolerance).all())
        orientation_ok = self.orientation_tolerance is None or bool(
            (orientation_error < self.orientation_tolerance).all()
        )
        if position_ok and orientation_ok:
            self._reached = True
            return Status.SUCCESS
        waiting_for = "orientation" if position_ok else "position"
        if self.position_stall_timeout_s is not None:
            if waiting_for != self._waiting_for:
                logger.info(
                    "%s waiting_for=%s xyz_error_m=%s orientation_error_rad=%s current=%s target=%s",
                    self.name, waiting_for, (self._goal.pos - current.pos).round(5).tolist(),
                    orientation_error.round(5).tolist(), current.pos.round(5).tolist(),
                    self._goal.pos.round(5).tolist(),
                )
                self._waiting_for = waiting_for
            if self._position_best is None:
                self._position_best = np.array(error, copy=True)
                self._position_stale_steps = np.zeros_like(error, dtype=int)
            else:
                # Rotation alone must not count as translational progress.
                progressed = error < self._position_best - 0.001
                reached = error < self.position_tolerance
                self._position_best = np.where(progressed | reached, error, self._position_best)
                self._position_stale_steps = np.where(
                    progressed | reached, 0, self._position_stale_steps + 1)
                stalled = (~reached) & (
                    self._position_stale_steps * ctx.dt >= self.position_stall_timeout_s)
                if np.any(stalled):
                    logger.warning(
                        "%s translation stalled: xyz_error_m=%s current=%s target=%s; failing attempt",
                        self.name, (self._goal.pos - current.pos).round(5).tolist(),
                        current.pos.round(5).tolist(), self._goal.pos.round(5).tolist(),
                    )
                    return Status.FAILURE
        # Keep commanding the goal while the controller closes the gap; give up
        # only after the settle budget so a stuck arm fails rather than hangs.
        overrun = (self._tick - self._steps) * ctx.dt
        if overrun < self.settle_timeout_s:
            return Status.RUNNING
        logger.warning(
            "%s failed to reach target: error_m=%s tolerance_m=%.4f orientation_error_rad=%s"
            " orientation_tolerance_rad=%s current=%s target=%s",
            self.name,
            np.asarray(error).round(5).tolist(),
            self.position_tolerance,
            np.asarray(orientation_error).round(5).tolist(),
            self.orientation_tolerance,
            current.pos.round(5).tolist(),
            self._goal.pos.round(5).tolist(),
        )
        return Status.FAILURE

    def _orientation_error_rad(self, current: Pose) -> np.ndarray:
        """Angle between current and goal orientation; sign-invariant (q and -q match)."""
        dot = np.sum(current.quat * self._goal.quat, axis=-1)
        return 2.0 * np.arccos(np.clip(np.abs(dot), -1.0, 1.0))

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(reached=self._reached, tcp=ctx.scene.tcp(self.robot))
