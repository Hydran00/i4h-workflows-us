# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka Panda with an ultrasound probe over an abdominal phantom.

The only ee_pose scene with a Vention table rather than Props/Table, and the
only one whose default backend is openpi PI0 — neither fact appears anywhere in
this class, because the backend is a manifest lookup.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from i4h_common.config import get_robot_config

from i4h_arena.adapters.actuation import RobotSlice
from i4h_arena.scenes.base import Scene


class PandaPhantomScene(Scene):
    name = "panda_phantom"

    def configure_args(self, args):
        super().configure_args(args)
        if getattr(args, "ultrasound", False) and "ultrasound" not in self.spec.cameras:
            self.spec = replace(self.spec, cameras=(*self.spec.cameras, "ultrasound"))

    def make_view(self, env):
        if getattr(self.args, "ultrasound", False):
            self._ultrasound_sensor = env.unwrapped.scene["ultrasound"]
            self._ultrasound_sensor.bind_scene(env.unwrapped.scene)
        return super().make_view(env)

    def default_sensor_views(self):
        return ("ultrasound",) if getattr(self.args, "ultrasound", False) else ()

    def sensor_view_titles(self):
        return {"ultrasound": "Ultrasound B-mode (OptiX)"}

    def close(self):
        sensor = getattr(self, "_ultrasound_sensor", None)
        if sensor is not None:
            sensor.close()

    def register_assets(self) -> None:
        import i4h_arena.assets.panda_phantom  # noqa: F401

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.panda_phantom import make_assets
        from i4h_arena.embodiments.franka import FrankaUltrasoundEmbodiment
        from i4h_arena.envcfg.panda_phantom import PandaPhantomEnvCfg

        steps = self.args.episode_steps or self.spec.max_steps
        embodiment = FrankaUltrasoundEmbodiment(enable_cameras=bool(getattr(self.args, "enable_cameras", True)))

        assets = make_assets()
        if getattr(self.args, "ultrasound", False):
            from i4h_arena.assets.config_asset import ConfigAsset
            from i4h_arena.sensors.ultrasound import UltrasoundSensorCfg

            assets.append(
                ConfigAsset(
                    "ultrasound",
                    UltrasoundSensorCfg(
                        prim_path="{ENV_REGEX_NS}/organs/UltrasoundSensor",
                        # Match the control loop's own rate: a slower sensor
                        # clock left every recorded frame between real updates
                        # a duplicate of the last one (measurable via
                        # obs/ultrasound_frame_id in the recording). Empirically
                        # (not from IsaacLab's SensorBase scheduling docs) the
                        # sensor's own update clock runs at half of what
                        # 1/control_hz would suggest, hence the extra factor.
                        update_period=1.0 / (2 * self.spec.control_hz),
                        # ~1cm above the world-Z the probe actually converges
                        # to at contact (measured 0.196-0.197m across several
                        # recorded runs, consistently, via obs/measured_ee_pose
                        # at the end of the make_contact segment). A first
                        # attempt reused the workflow's own coarse
                        # _ULTRASOUND_CONTACT_Z_MAX_M=0.22 gate directly, which
                        # activated too early (confirmed visually); a second
                        # attempt overcorrected to 0.13, which is BELOW the
                        # reachable range, so the sensor never activated at
                        # all. 0.20 sits just above the measured convergence
                        # height instead of a boundary borrowed from elsewhere.
                        activation_height_m=0.20,
                        image=self.args.ultrasound_image,
                        gpu=self.args.ultrasound_gpu,
                    ),
                )
            )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=ArenaScene(assets=assets),
            # PandaPhantomEnvCfg takes no env_spacing: the phantom scene is a
            # single workspace, not a tiled grid.
            task=PandaPhantomEnvCfg(episode_length_s=max(10.0, (steps + 1) / self.spec.control_hz)),
        )

    def home_joints(self, env: Any) -> np.ndarray | None:
        home = get_robot_config(self.spec.embodiment).home_joint_pos_rad
        if not home:
            return None
        return np.tile(np.asarray(home, dtype=np.float32), (int(env.unwrapped.num_envs), 1))

    def tcp_body(self) -> str | None:
        return "TCP"

    def tcp_sensors(self) -> dict[str, str]:
        return {"robot": "ee_frame"}

    def camera_aliases(self) -> dict[str, str]:
        return {"room": "room_camera", "wrist": "wrist_camera"}

    def relative_ee(self) -> bool:
        # The Panda uses a 6-D relative Cartesian controller. ArenaActuation
        # converts scripted absolute targets and passes policy deltas directly.
        return True

    def robot_slices(self, env: Any) -> tuple[RobotSlice, ...]:
        # Probe, not a gripper: no jaw column to claim.
        width = int(env.action_space.shape[-1])
        return (RobotSlice("robot", 0, width, gripper_index=None),)

    def make_actuation(self, env, view=None):
        from i4h_common.types import Pose

        actuation = super().make_actuation(env, view)
        if getattr(self, "_scan_start_target", None) is not None:
            position, orientation = self._scan_start_target
            # Keep the contact preload when the runner rebuilds its command buffer.
            actuation.set_ee_target(Pose(pos=position.copy(), quat=orientation.copy()), "robot")
        return actuation

    def on_reset(self, env: Any, view: Any, on_progress=None) -> None:
        """Align and autonomously land before engine.start and recorder.begin_episode."""
        import logging

        import torch
        from i4h_common.types import quat_rotate
        from i4h_common.ultrasound_scan import APPROACH, CONTACT, CONTACT_SETTLE_S, scan_orientation
        from isaaclab.utils.math import compute_pose_error

        from i4h_arena.envcfg.panda_phantom import probe_phantom_contact, reset_ultrasound_success_state
        from i4h_arena.tensor_utils import to_torch

        unwrapped = env.unwrapped
        self._scan_start_target = None
        unwrapped._ultrasound_preparing = True
        device = unwrapped.device
        num_envs = unwrapped.num_envs
        phantom = view.object("organs").pose
        orientation_wxyz = scan_orientation(phantom.quat)
        # Public Pose is wxyz; this pinned IsaacLab build uses xyzw.
        target_quat = torch.as_tensor(orientation_wxyz[:, [1, 2, 3, 0]], device=device)
        approach_world = phantom.pos + quat_rotate(phantom.quat, np.broadcast_to(APPROACH, phantom.pos.shape))
        contact_world = phantom.pos + quat_rotate(phantom.quat, np.broadcast_to(CONTACT, phantom.pos.shape))
        target_pos = torch.as_tensor(approach_world, device=device, dtype=torch.float32)

        def servo(position):
            data = unwrapped.scene["ee_frame"].data
            current_pos = to_torch(data.target_pos_w)[:, 0, :]
            current_quat = to_torch(data.target_quat_w)[:, 0, :]
            dp, dr = compute_pose_error(current_pos, current_quat, position, target_quat, rot_error_type="axis_angle")
            # Bounded Cartesian increments; the existing IK/controller owns joint actuation.
            # This pre-roll never appears in the recording, so it can move as
            # fast as the controller tracks cleanly; these caps only bound a
            # single step's command, not the pre-roll's overall duration.
            dp = dp * (0.04 / torch.linalg.vector_norm(dp, dim=-1, keepdim=True).clamp_min(0.04))
            dr = dr * (0.3 / torch.linalg.vector_norm(dr, dim=-1, keepdim=True).clamp_min(0.3))
            with torch.no_grad():
                env.step(torch.cat((dp, dr), dim=-1))
            if on_progress is not None:
                on_progress()
            return dp, dr

        # Rotate above the phantom and converge before descending.
        for _ in range(250):
            servo(target_pos)
            data = unwrapped.scene["ee_frame"].data
            dp, dr = compute_pose_error(to_torch(data.target_pos_w)[:, 0], to_torch(data.target_quat_w)[:, 0],
                                       target_pos, target_quat, rot_error_type="axis_angle")
            if bool(((dp.norm(dim=-1) < 0.015) & (dr.norm(dim=-1) < 0.05)).all()):
                break
        else:
            raise RuntimeError("Probe could not reach the aligned pre-contact pose; episode not started")

        if getattr(self.args, "mode", None) in ("rule-based", "policy"):
            stable = torch.zeros(num_envs, dtype=torch.long, device=device)
            required = max(1, round(CONTACT_SETTLE_S / unwrapped.step_dt))
            bottom = torch.as_tensor(contact_world, device=device, dtype=torch.float32)
            # Slow autonomous descent; stop lowering each probe as soon as it contacts.
            for _ in range(500):
                touching = probe_phantom_contact(unwrapped)
                target_pos[:, 2] = torch.where(touching, target_pos[:, 2],
                    torch.maximum(target_pos[:, 2] - 0.08 * unwrapped.step_dt, bottom[:, 2]))
                servo(target_pos)
                touching = probe_phantom_contact(unwrapped)
                stable = torch.where(touching, stable + 1, torch.zeros_like(stable))
                if bool((stable >= required).all()):
                    break
            else:
                raise RuntimeError(f"No stable probe/phantom contact: tcp={to_torch(unwrapped.scene['ee_frame'].data.target_pos_w)[:, 0].cpu().tolist()} target={target_pos.cpu().tolist()} forces={to_torch(unwrapped.scene['contact_probe_organs'].data.force_matrix_w).cpu().tolist()}; episode not started")
            logging.getLogger("i4h_arena.ultrasound").info(
                "Scan starts after physical contact: fixed_quat_wxyz=%s position=%s",
                orientation_wxyz.tolist(), to_torch(unwrapped.scene["ee_frame"].data.target_pos_w)[:, 0].cpu().tolist(),
            )
        self._scan_start_target = (target_pos.detach().cpu().numpy(), orientation_wxyz.copy())
        env_ids = torch.arange(num_envs, device=device)
        reset_ultrasound_success_state(unwrapped, env_ids)
        unwrapped._ultrasound_preparing = False
        view.invalidate()
