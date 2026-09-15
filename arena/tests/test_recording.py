# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from i4h_arena.recording.hdf5 import EpisodeRecorder


class _Frame:
    def __init__(self, value: int, frame_num: int = 0) -> None:
        self.value = value
        self.frame_num = frame_num

    def to_array(self) -> np.ndarray:
        return np.full((32, 48, 3), self.value, dtype=np.uint8)


class _View:
    def __init__(self) -> None:
        self.step = 0

    def joints(self) -> SimpleNamespace:
        return SimpleNamespace(pos=np.full((1, 6), self.step, dtype=np.float32),
                               vel=np.full((1, 6), self.step * 0.5, dtype=np.float32))

    def tcp(self, _robot: str = "robot") -> SimpleNamespace:
        return SimpleNamespace(
            pos=np.full((1, 3), self.step, dtype=np.float32),
            quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 1)),
        )

    def camera(self, _name: str) -> _Frame:
        return _Frame(self.step)


class _SlowClockView(_View):
    """A sensor (e.g. ultrasound) that renders on its own schedule, slower than the control loop."""

    def camera(self, _name: str) -> _Frame:
        return _Frame(self.step, frame_num=self.step // 3)


def _result(*, succeeded: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        attempt=1,
        succeeded=succeeded,
        status=SimpleNamespace(value="succeeded" if succeeded else "failed"),
    )


def test_streams_camera_frames_and_commits_episode(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="policy", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    for step in range(70):
        view.step = step
        recorder.on_step(np.full((1, 6), step, dtype=np.float32), view)

    recorder._drain_frames()
    with h5py.File(path, "r") as handle:
        assert handle["data/_attempt/obs/room"].shape == (70, 32, 48, 3)
        assert handle["data/_attempt/obs/room"].chunks == (1, 32, 48, 3)
        assert handle["data/_attempt/obs/room"].compression == "lzf"

    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert "_attempt" not in handle["data"]
        assert handle["data/demo_0/actions"].shape == (70, 6)
        assert handle["data/demo_0/obs/joint_pos"].shape == (70, 6)
        assert handle["data/demo_0/obs/room"].shape == (70, 32, 48, 3)
        assert np.all(handle["data/demo_0/obs/room"][-1] == 69)


class _MedicalView(_View):
    """A sensor that also exposes the pre-display signal, as fluoroscopy does."""

    def sensor_signal(self, _name: str, output: str) -> np.ndarray | None:
        if output != "attenuation":
            return None
        return np.full((32, 48, 1), 0.25 * self.step, dtype=np.float32)


def test_records_the_display_independent_signal_beside_the_image(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("fluoroscopy",))
    view = _MedicalView()

    recorder.begin_episode(0, 1)
    for step in range(4):
        view.step = step
        recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        assert obs["fluoroscopy"].shape == (4, 32, 48, 3)
        signal = obs["fluoroscopy_attenuation"]
        assert signal.shape == (4, 32, 48, 1)
        assert signal.dtype == np.float32
        # Full precision, not quantized through an 8-bit image.
        assert np.allclose(signal[-1], 0.75)


def test_a_view_without_a_signal_records_images_only(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert list(handle["data/demo_0/obs"]) == ["joint_pos", "joint_vel", "measured_ee_pose", "room"]


def test_discards_temporary_episode(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="policy", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(succeeded=False), keep=False)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert list(handle["data"]) == []


def test_records_measured_and_commanded_ee_pose(tmp_path) -> None:
    from i4h_common.types import Pose

    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="rule-based", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _View()

    recorder.begin_episode(0, 1)
    # Step 0: no active Cartesian target (e.g. a locate/hold node).
    view.step = 0
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view, None)
    # Step 1: a task is driving an explicit absolute target.
    view.step = 1
    commanded = Pose(pos=np.array([[1.0, 2.0, 3.0]]), quat=np.array([[1.0, 0.0, 0.0, 0.0]]))
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view, commanded)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        measured = handle["data/demo_0/obs/measured_ee_pose"][()]
        np.testing.assert_allclose(measured[0, :3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(measured[1, :3], [1.0, 1.0, 1.0])
        commanded_recorded = handle["data/demo_0/obs/commanded_ee_pose"][()]
        assert np.all(np.isnan(commanded_recorded[0]))
        np.testing.assert_allclose(commanded_recorded[1], [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])


def test_records_frame_id_for_a_sensor_on_its_own_clock(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="rule-based", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("ultrasound",))
    view = _SlowClockView()

    recorder.begin_episode(0, 1)
    for step in range(6):
        view.step = step
        recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        frame_id = handle["data/demo_0/obs/ultrasound_frame_id"][()]
        np.testing.assert_array_equal(frame_id, [0, 0, 0, 1, 1, 1])


def test_omits_frame_id_for_a_plain_camera(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    for step in range(3):
        view.step = step
        recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert "room_frame_id" not in handle["data/demo_0/obs"]


def test_omits_commanded_ee_pose_when_never_used(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert "commanded_ee_pose" not in handle["data/demo_0/obs"]
        assert "measured_ee_pose" in handle["data/demo_0/obs"]


def test_reset_discards_pre_reset_samples(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.full((1, 6), 3.0, dtype=np.float32), view)
    recorder.restart_episode(node="drive", task_id="teleop/drive")
    recorder.on_step(np.full((1, 6), 7.0, dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["data/demo_0/actions"][:], np.full((1, 6), 7.0))
        assert handle["data/demo_0/segments"][0]["start"] == 0


class _PhantomView(_View):
    def phantom_recording_state(self):
        pose = np.array([self.step, 2., 3., 1., 0., 0., 0.])
        return {'phantom_pose': pose.copy(), 'mesh_pose': pose.copy(),
                'ultrasound_probe_pose': pose.copy(), 'timestamps': np.asarray(self.step * .02)}


def test_records_phantom_per_sample_and_clears_on_restart(tmp_path):
    path = tmp_path / 'phantom.hdf5'
    workflow = SimpleNamespace(name='ultrasound_liver_scan', mode='rule-based', scene='panda_phantom')
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=('ultrasound',))
    view = _PhantomView()
    try:
        recorder.begin_episode(0, 1)
        view.step = 99
        recorder.on_step(np.zeros((1, 6)), view)
        recorder.restart_episode()
        for step in range(3):
            view.step = step
            recorder.on_step(np.zeros((1, 6)), view)
        recorder.end_episode(_result(), keep=True)
        recorder.begin_episode(1, 1)
        view.step = 10
        recorder.on_step(np.zeros((1, 6)), view)
        recorder.end_episode(_result(), keep=True)
    finally:
        recorder.close()
    with h5py.File(path) as handle:
        obs = handle['data/demo_0/obs']
        for key in ('phantom_pose', 'mesh_pose', 'ultrasound_probe_pose'):
            assert obs[key].shape == (3, 7)
            assert obs[key].attrs['quaternion_order'] == 'wxyz'
            assert obs[key].attrs['frame'] == 'world'
            np.testing.assert_allclose(obs[key][:, 0], [0, 1, 2])
        np.testing.assert_allclose(obs['timestamps'][:], [0, .02, .04])
        assert handle['data/demo_1/obs/mesh_pose'].shape == (1, 7)
        assert handle['data/demo_1/obs/mesh_pose'][0, 0] == 10
