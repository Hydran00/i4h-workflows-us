# SPDX-License-Identifier: Apache-2.0
"""Camera-compatible B-mode sensor driven by the scene's calibrated frames."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.configclass import configclass

from i4h_arena.medical.ultrasound import UltrasoundRenderer, bmode_rgb, probe_in_mesh
from i4h_arena.tensor_utils import to_torch


class UltrasoundSensor(SensorBase):
    def __init__(self, cfg):
        self._renderer = None
        self._scene = None
        self._data = SimpleNamespace(output={}, frame_id=None)
        super().__init__(cfg)

    @property
    def data(self):
        self._update_outdated_buffers()
        return self._data

    def bind_scene(self, scene):
        self._scene = scene

    def reset(self, env_ids=None, env_mask=None):
        """Clear the last frame/frame_id for the reset envs.

        Without this, a reader (and obs/ultrasound_frame_id) would see the
        previous episode's last B-mode image and count carried into the new
        one, since the base class only resets its own outdated-timestamp
        bookkeeping, not this sensor's own buffers.
        """
        super().reset(env_ids, env_mask)
        if self._data.frame_id is None:
            return  # Not yet initialized: nothing to clear.
        mask = self._resolve_indices_and_mask(env_ids, env_mask)
        indices = np.flatnonzero(wp.to_torch(mask).cpu().numpy())
        self._data.output["rgb"][indices] = 0
        self._data.output["bmode_db"][indices] = -60.0
        self._data.frame_id[indices] = 0

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data.output = {
            "rgb": torch.zeros((self._num_envs, 256, 256, 3), dtype=torch.uint8, device=self._device),
            "bmode_db": torch.full((self._num_envs, 256, 256, 1), -60.0, device=self._device),
        }
        self._data.frame_id = torch.zeros(self._num_envs, dtype=torch.int64, device=self._device)

    def _update_buffers_impl(self, env_mask):
        if self._scene is None:
            return  # Environment initialization precedes Scene.make_view binding.
        if self._renderer is None:
            self._renderer = UltrasoundRenderer(self.cfg.image, self.cfg.gpu)
        probe = self._scene["ee_to_us_transform"].data
        mesh = self._scene["mesh_to_organ_transform"].data
        poses = [
            to_torch(value).detach().cpu().numpy()[:, 0]
            for value in (
                probe.target_pos_w,
                probe.target_quat_w,
                mesh.target_pos_w,
                mesh.target_quat_w,
            )
        ]
        probe_pos_w = poses[0]
        for index in np.flatnonzero(wp.to_torch(env_mask).cpu().numpy()):
            if self.cfg.activation_height_m is not None and probe_pos_w[index, 2] > self.cfg.activation_height_m:
                # Still well above the surface: a real probe would form no
                # image here. Leave frame_id untouched so a reader can tell
                # (obs/ultrasound_frame_id) that no genuine scan happened yet,
                # rather than rendering a physically meaningless frame.
                continue
            position, angles = probe_in_mesh(*(value[index] for value in poses))
            if int(self._data.frame_id[index]) == 0:
                logging.getLogger(__name__).info("ultrasound mesh pose mm=%s xyz_rad=%s", position, angles)
            frame = self._renderer.render(position, angles)
            self._data.output["rgb"][index] = torch.as_tensor(bmode_rgb(frame), device=self._device)
            self._data.output["bmode_db"][index, ..., 0] = torch.as_tensor(frame, device=self._device)
            self._data.frame_id[index] += 1

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


@configclass
class UltrasoundSensorCfg(SensorBaseCfg):
    class_type: type = UltrasoundSensor
    image: str = "i4h_sim_build:ultrasound-simulator"
    gpu: str = "0"
    #: World-Z (metres) above which no B-mode is rendered (None = always on).
    #: The scene sets this from its own calibrated contact height, not a
    #: value this generic sensor should guess.
    activation_height_m: float | None = None
