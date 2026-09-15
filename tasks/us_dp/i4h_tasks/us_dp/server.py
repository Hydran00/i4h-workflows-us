# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve XYZ spline prefixes from real, synchronized checkpoint-compatible observation histories.

The remote proxy collects history at the checkpoint cadence during execution.
Orientation is fixed at rollout start; IK and contact control remain scene-owned.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np
from i4h_common.bus.messages import ObsFrame, decode
from i4h_common.server import ActionContract, PolicyServer, Session

logger = logging.getLogger("i4h_tasks.us_dp")

def _quat_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of ``i4h_engine.remote._axis_angle_to_quat``: wxyz quat to rotation vector."""
    w = float(np.clip(quat_wxyz[0], -1.0, 1.0))
    sin_half = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = quat_wxyz[1:4] / sin_half
    return (axis * angle).astype(np.float32)


class UsDpServer(PolicyServer):
    """Serve ``us_dp/*`` tasks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._runners: dict[str, Any] = {}
        self._fixed_rotvec: dict[str, np.ndarray] = {}
        self._zero_images = os.environ.get("US_DP_ZERO_IMAGES", "0") == "1"
        logger.info("Ultrasound input ablation: zero_images=%s", self._zero_images)
        self._single_plan = os.environ.get("US_DP_SINGLE_PLAN", "0") == "1"
        self._plan_end: dict[str, int] = {}

    def _key(self, session: Session) -> str:
        return session.checkpoint or str(session.model.get("checkpoint", ""))

    def load(self, session: Session) -> None:
        from us_dp.deployment.inference import RecedingHorizonPolicy

        key = self._key(session)
        if not key:
            raise ValueError(f"{session.task_id}: no checkpoint in the manifest and no --checkpoint")
        if key not in self._runners:
            import torch

            device = os.environ.get("US_DP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            logger.info("loading us_dp checkpoint %s (device=%s)", key, device)
            self._runners[key] = RecedingHorizonPolicy(key, device=device)
        # A new session is a new episode: the receding-horizon history must not
        # span across a reset, even when the same checkpoint (and therefore the
        # same cached runner) serves the next episode too.
        self._runners[key].reset()
        self._fixed_rotvec.pop(session.task_uid, None)
        self._plan_end.pop(session.task_uid, None)

    def unload(self, session: Session) -> None:
        pass  # the loaded checkpoint stays cached across episodes; only its history resets (see load)

    def is_done(self, session: Session, frame: ObsFrame) -> bool:
        end = self._plan_end.get(session.task_uid)
        return self._single_plan and end is not None and frame.step >= end

    def action_contract(self, session: Session) -> ActionContract:
        # 6 columns: absolute world position (3) + held orientation as a
        # rotation vector (3) -- the scene's ee_pose action space is 6-dof
        # (matches openpi_pi0's own contract here), not the 7-value pos_quat.
        config = self._runners[self._key(session)].config
        return ActionContract(
            space="ee_pose", layout="pos_axis_angle", dof=6,
            observation_history=config.history, observation_sample_hz=config.sample_hz,
        )

    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        from us_dp.common.geometry import rotation_matrix
        from us_dp.common.state import STATE_FIELDS, validate_state_fields

        if self._single_plan and session.task_uid in self._plan_end:
            if self.is_done(session, frame):
                logger.info("Single-plan diagnostic completed; no replanning (not a scan-quality verdict)")
            return None
        key = self._key(session)
        runner = self._runners.get(key)
        if runner is None:
            logger.warning("session %s step=%d: no runner loaded for checkpoint %r", session.task_uid, frame.step, key)
            return None
        fields = runner.metadata["state_fields"]
        validate_state_fields(fields)
        if frame.dt <= 0:
            logger.warning("session %s step=%d: frame.dt=%r, cannot derive a timestamp", session.task_uid, frame.step, frame.dt)
            return None
        frames = [decode(item, ObsFrame) for item in frame.history]
        if len(frames) != runner.config.history:
            raise ValueError("Expected a real observation history from the remote proxy")
        if frames[-1].step != frame.step:
            raise ValueError("History must end at the current observation")
        # Replace the rolling window, preserving the rollout's fixed orientation.
        runner.images.clear()
        runner.states.clear()
        runner.last_timestamp = None
        robot_name = session.model.get("robot", "robot")
        for item in frames:
            ultrasound = session.images(item).get("ultrasound")
            ee = session.ee_poses(item).get(robot_name)
            if ultrasound is None or ee is None:
                raise ValueError("History requires ultrasound and measured TCP pose")
            if item.dt != frame.dt:
                raise ValueError("History timebase changed during rollout")
            position, quat = ee[:3].astype(np.float32), ee[3:7].astype(np.float32)
            rotation = rotation_matrix(quat, "wxyz")
            state = np.concatenate((position, rotation[:, 0], rotation[:, 1]))
            if fields == STATE_FIELDS:
                ids = [item.state_names.index(f"panda_joint{i}") for i in range(1, 8)]
                q = np.asarray(item.state, dtype=np.float32)[ids]
                dq = np.asarray(item.state_velocities, dtype=np.float32)[ids]
                state = np.concatenate((q, dq, state))
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3], pose[:3, 3] = rotation, position
            gray = np.rint(ultrasound.mean(axis=-1)).astype(np.uint8)
            if self._zero_images:
                gray = np.zeros_like(gray)
            runner.observe(gray, state, pose, item.step * item.dt)
            self._fixed_rotvec.setdefault(session.task_uid, _quat_to_axis_angle(quat))
        options = {"horizon_seconds": runner.config.prediction_seconds} if self._single_plan else {}
        plan = runner.plan(control_hz=1.0 / frame.dt, **options)
        positions = plan["positions_world"].astype(np.float32)
        if self._single_plan:
            self._plan_end[session.task_uid] = frame.step + len(positions)
            delta = np.diff(np.vstack((pose[:3, 3], positions)), axis=0)
            logger.info(
                "Single-plan diagnostic: %d actions, %.3f s; first displacement=%.6f m, max step=%.6f m",
                len(positions), len(positions) * frame.dt,
                np.linalg.norm(delta[0]), np.linalg.norm(delta, axis=1).max(),
            )
        if self._single_plan and os.environ.get("I4H_RUN_DIR"):
            from pathlib import Path
            from us_dp.deployment.plot_trajectory import save_trajectory

            try:
                path = Path(os.environ["I4H_RUN_DIR"]) / "trajectories" / (
                    f"episode_{session.episode_index:04d}_step_{frame.step:06d}"
                )
                saved = save_trajectory(path, pose[:3, 3], positions,
                                        np.arange(1, len(positions) + 1) * frame.dt)
                logger.info("Predicted 3D trajectory saved: %s", saved)
            except ImportError:
                logger.warning("Install matplotlib to save the predicted 3D trajectory")
        rotvec = np.tile(self._fixed_rotvec[session.task_uid], (len(positions), 1))
        return np.concatenate((positions, rotvec), axis=-1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="us-dp-server")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help="load this task's checkpoint at startup instead of on the first spec",
    )
    parser.add_argument("--checkpoint", default="", help="checkpoint override used while preloading")
    parser.add_argument("--preload-only", action="store_true", help="load requested checkpoints and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[us-dp] %(message)s",
    )
    server = UsDpServer(namespace=args.namespace)
    if args.preload_only:
        if not args.preload:
            parser.error("--preload-only requires --preload")
        server.preload_only(tuple(args.preload), checkpoint=args.checkpoint)
        return 0
    server.serve_forever(
        preload=tuple(args.preload),
        preload_checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
