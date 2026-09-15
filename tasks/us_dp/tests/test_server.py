from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from i4h_common.bus.messages import ObsFrame, decode, encode
from i4h_common.server import Session

from i4h_tasks.us_dp.server import UsDpServer


class Runner:
    config = SimpleNamespace(history=3, sample_hz=10, execution_seconds=0.4)

    def __init__(self, pose_only=False):
        from us_dp.common.state import POSE_FIELDS, STATE_FIELDS
        self.metadata = {"state_fields": POSE_FIELDS if pose_only else STATE_FIELDS}
        self.images, self.states = deque(), deque()
        self.last_timestamp = None
        self.received = []

    def observe(self, image, state, pose, timestamp):
        self.received.append((image, state, pose, timestamp))

    def plan(self, control_hz):
        assert control_hz == 50
        return {"positions_world": np.zeros((20, 3), dtype=np.float32)}


def frame(step, angle=0):
    return ObsFrame(
        step=step, dt=0.02,
        state_names=[f"panda_joint{i}" for i in range(1, 8)],
        state=list(np.arange(7, dtype=float)), state_velocities=[0.5] * 7,
        ee_robots=["robot"], ee_pose=[0.5, 0, 0.2, np.cos(angle / 2), 0, 0, np.sin(angle / 2)],
        images={"ultrasound": np.full((4, 4, 3), step, np.uint8).tobytes()},
        image_shapes={"ultrasound": [4, 4, 3]},
    )


@pytest.mark.parametrize("zero_images", [False, True])
@pytest.mark.parametrize("pose_only", [False, True])
def test_server_uses_real_full_state_history_and_fixed_orientation(pose_only, zero_images):
    server = object.__new__(UsDpServer)
    runner = Runner(pose_only)
    server._runners = {"checkpoint": runner}
    server._fixed_rotvec = {}
    server._zero_images = zero_images
    server._single_plan = False
    server._plan_end = {}
    session = Session(task_uid="episode", task_id="us_dp/test", run_id="run", episode_index=0,
                      prompt="", checkpoint="checkpoint")
    contract = server.action_contract(session)
    assert contract.observation_history == 3 and contract.observation_sample_hz == 10
    request = frame(10)
    request.history = [encode(frame(step, 0.4)) for step in [0, 5, 10]]
    first = server.infer(session, request)
    assert first.shape == (20, 6)
    np.testing.assert_allclose(first[:, 3:], np.tile([0, 0, 0.4], (20, 1)), atol=1e-6)
    assert [item[3] for item in runner.received] == [0, 0.1, 0.2]
    assert [item[0][0, 0] for item in runner.received] == ([0, 0, 0] if zero_images else [0, 5, 10])
    if zero_images:
        assert all(np.count_nonzero(item[0]) == 0 for item in runner.received)
    assert decode(request.history[-1], ObsFrame).images == frame(10).images
    state = runner.received[-1][1]
    assert state.shape == ((9,) if pose_only else (23,))
    if not pose_only:
        np.testing.assert_array_equal(state[:7], np.arange(7))
        np.testing.assert_array_equal(state[7:14], np.full(7, 0.5))
    np.testing.assert_allclose(state[-9:-6], [0.5, 0, 0.2])
    np.testing.assert_allclose(state[-6:], [np.cos(0.4), np.sin(0.4), 0, -np.sin(0.4), np.cos(0.4), 0], atol=1e-6)
    request = frame(30, 0.8)
    request.history = [encode(frame(step, 0.8)) for step in [20, 25, 30]]
    second = server.infer(session, request)
    np.testing.assert_array_equal(second[:, 3:], first[:, 3:])
    with pytest.raises(ValueError, match="real observation history"):
        server.infer(session, frame(35))


def test_single_plan_executes_full_horizon_then_finishes_without_inference():
    server = object.__new__(UsDpServer)
    runner = Runner(pose_only=True)
    runner.config = SimpleNamespace(history=3, sample_hz=50, prediction_seconds=2.0)
    calls = []
    def plan(**kwargs):
        calls.append(kwargs)
        return {"positions_world": np.zeros((100, 3), dtype=np.float32)}
    runner.plan = plan
    server._runners = {"checkpoint": runner}
    server._fixed_rotvec = {}
    server._zero_images = False
    server._single_plan = True
    server._plan_end = {}
    session = Session(task_uid="episode", task_id="us_dp/test", run_id="run", episode_index=0,
                      prompt="", checkpoint="checkpoint")
    request = frame(2)
    request.history = [encode(frame(step)) for step in [0, 1, 2]]
    assert server.infer(session, request).shape == (100, 6)
    assert not server.is_done(session, request)
    assert not server.is_done(session, frame(101))
    assert server.infer(session, frame(102)) is None
    assert server.is_done(session, frame(102))
    assert calls == [{"control_hz": 50.0, "horizon_seconds": 2.0}]
