# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ultrasound liver-scan task.

Single-file env recipe — MDP helpers (events, observations, rewards), small
configclasses (Terminations, Rewards, Events, Observations), and the
:class:`PandaPhantomEnvCfg` recipe consumed by the env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.tasks.task_base import TaskBase

from i4h_arena.tensor_utils import to_torch

_ULTRASOUND_TARGET_TOLERANCE_M = 0.03
_ULTRASOUND_MAX_TCP_SPEED_M_S = 0.005


# ---------- Reset events ------------------------------------------------------


def reset_panda_joints_by_fraction_of_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fraction: float = 0.1,
) -> None:
    """Reset Panda joints with offsets sampled from a fraction of the joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = to_torch(asset.data.default_joint_pos)[env_ids].clone()
    joint_vel = to_torch(asset.data.default_joint_vel)[env_ids].clone()
    joint_limits = to_torch(asset.data.default_joint_limits)[env_ids].clone()
    joint_sample_ranges = joint_limits * fraction

    lower = joint_sample_ranges[:, :, 0]
    upper = joint_sample_ranges[:, :, 1]
    joint_pos_delta = torch.rand(joint_pos.shape, device=joint_pos.device) * (upper - lower) + lower
    joint_pos += joint_pos_delta
    joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])

    asset.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
    # Teleporting joint state does not replace targets left by the failed attempt.
    asset.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
    asset.set_joint_effort_target_index(target=torch.zeros_like(joint_pos), env_ids=env_ids)


def reset_ultrasound_success_state(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Clear the previous episode's success diagnostics."""
    if hasattr(env, "_ultrasound_success_last"):
        delattr(env, "_ultrasound_success_last")
    if hasattr(env, "_ultrasound_prev_tcp_pos"):
        delattr(env, "_ultrasound_prev_tcp_pos")


# ---------- Observations ------------------------------------------------------


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("organs"),
) -> torch.Tensor:
    """Organ XYZ position expressed in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    root_state_w = to_torch(robot.data.root_state_w)
    object_pos_b, _ = subtract_frame_transforms(
        root_state_w[:, :3], root_state_w[:, 3:7], to_torch(obj.data.root_pos_w)[:, :3]
    )
    return object_pos_b


# ---------- Rewards -----------------------------------------------------------


def object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("organs"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Inverse-square reward for approaching the organ scan target above the phantom."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = to_torch(obj.data.root_pos_w)
    target_pos_w = object_pos_w + torch.tensor([0.0, -0.25, 1.0], device=object_pos_w.device)
    ee_w = to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    distance = torch.norm(target_pos_w - ee_w, dim=1)

    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the EE z+x axes with the organ goal frame."""
    ee_frame_quat = to_torch(env.scene["ee_frame"].data.target_quat_w)[..., 0, :]
    goal_frame_quat = to_torch(env.scene["goal_frame"].data.target_quat_w)[..., 0, :]
    ee_rot = matrix_from_quat(ee_frame_quat)
    goal_rot = matrix_from_quat(goal_frame_quat)
    goal_x, goal_z = goal_rot[..., 0], goal_rot[..., 2]
    ee_x, ee_z = ee_rot[..., 0], ee_rot[..., 2]
    align_z = torch.bmm(ee_z.unsqueeze(1), goal_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), goal_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


# ---------- Terminations ------------------------------------------------------


def probe_phantom_contact(env):
    """Physical contact on the probe body, filtered to the phantom only."""
    from i4h_common.ultrasound_scan import CONTACT_FORCE_THRESHOLD_N

    forces = to_torch(env.scene["contact_probe_organs"].data.force_matrix_w)
    return torch.linalg.vector_norm(forces, dim=-1).flatten(1).amax(dim=1) >= CONTACT_FORCE_THRESHOLD_N


def ultrasound_scan_success(
    env: ManagerBasedRLEnv,
    target_tolerance_m: float = _ULTRASOUND_TARGET_TOLERANCE_M,
    max_tcp_speed_m_s: float = _ULTRASOUND_MAX_TCP_SPEED_M_S,
) -> torch.Tensor:
    """Complete only within 5 cm of the goal and below 1 cm/s TCP speed."""
    if getattr(env, "_ultrasound_preparing", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    target_pos, ee_pos, alignment = _ultrasound_success_metrics(env)
    distance = torch.linalg.norm(target_pos - ee_pos, dim=-1)
    previous = getattr(env, "_ultrasound_prev_tcp_pos", None)
    speed = (
        torch.linalg.norm(ee_pos - previous, dim=-1) / env.step_dt
        if previous is not None else torch.full_like(distance, float("inf"))
    )
    env._ultrasound_prev_tcp_pos = ee_pos.detach().clone()
    success = (distance < target_tolerance_m) & (speed < max_tcp_speed_m_s)
    env._ultrasound_success_last = {
        "target_pos": target_pos.detach().clone(),
        "ee_pos": ee_pos.detach().clone(),
        "distance": distance.detach().clone(),
        "tcp_speed_m_s": speed.detach().clone(),
        "alignment": alignment.detach().clone(),
        "success": success.detach().clone(),
        "target_tolerance_m": target_tolerance_m,
        "max_tcp_speed_m_s": max_tcp_speed_m_s,
    }
    return success


def _ultrasound_success_metrics(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scene = env.scene
    target_pos = to_torch(scene["goal_frame"].data.target_pos_w)[:, 0, :]
    ee_pos = to_torch(scene["ee_frame"].data.target_pos_w)[:, 0, :]
    ee_rot = matrix_from_quat(to_torch(scene["ee_frame"].data.target_quat_w)[:, 0, :])
    goal_rot = matrix_from_quat(to_torch(scene["goal_frame"].data.target_quat_w)[:, 0, :])
    align_z = torch.sum(ee_rot[..., 2] * goal_rot[..., 2], dim=-1)
    align_x = torch.sum(ee_rot[..., 0] * goal_rot[..., 0], dim=-1)
    alignment = torch.minimum(align_z, align_x)
    return target_pos, ee_pos, alignment


# ---------- Env-cfg recipe ----------------------------------------------------


@configclass
class _EventsCfg:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_success_state = EventTermCfg(func=reset_ultrasound_success_state, mode="reset")
    reset_object_position = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.0, -0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("organs"),
        },
    )
    reset_joint_position = EventTermCfg(
        func=reset_panda_joints_by_fraction_of_limits,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]), "fraction": 0.01},
    )
    # Real ultrasound probes glide on gel-lubricated skin; without an explicit low-friction
    # material both bodies fall back to PhysX/USD defaults, which is high enough for the
    # probe to catch on the phantom surface instead of sliding along it.
    organs_physics_material = EventTermCfg(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("organs"),
            "static_friction_range": (0.08, 0.08),
            "dynamic_friction_range": (0.05, 0.05),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )
    probe_physics_material = EventTermCfg(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "static_friction_range": (0.08, 0.08),
            "dynamic_friction_range": (0.05, 0.05),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )


@configclass
class _RewardsCfg:
    reaching_object = RewardTermCfg(func=object_ee_distance, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewardTermCfg(func=align_ee_handle, weight=2.5)
    alive = RewardTermCfg(func=base_mdp.is_alive, weight=0.1)
    action_rate_l2 = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewardTermCfg(func=base_mdp.joint_vel_l2, weight=-0.0001)


def ultrasound_time_out(env):
    """Preparation has its own bounded loops, outside the scan episode budget."""
    if getattr(env, "_ultrasound_preparing", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return base_mdp.time_out(env)


@configclass
class _TerminationsCfg:
    time_out = TerminationTermCfg(func=ultrasound_time_out, time_out=True)
    success = TerminationTermCfg(func=ultrasound_scan_success, time_out=False)


class PandaPhantomEnvCfg(TaskBase):
    """Arena task wrapper for the Franka liver-scan ultrasound objective."""

    def __init__(self, episode_length_s: float = 5.0):
        super().__init__(episode_length_s=episode_length_s, task_description="Perform a liver ultrasound.")

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return _TerminationsCfg()

    def get_events_cfg(self):
        return _EventsCfg()

    def get_rewards_cfg(self):
        return _RewardsCfg()

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.5, 1.3, 1.0), lookat=(0.0, 0.0, 0.0))

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg
