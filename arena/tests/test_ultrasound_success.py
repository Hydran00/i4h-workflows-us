"""CPU checks for the ultrasound scan's distance-only completion rule."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from pytest import approx


def _success_predicate():
    source = Path(__file__).parents[1] / "i4h_arena/envcfg/panda_phantom.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ultrasound_scan_success")
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    node.args.defaults = []
    namespace = {
        "torch": torch,
        "_ultrasound_success_metrics": lambda env: (env.target, env.tcp, env.alignment),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ultrasound_scan_success"]


def test_scan_success_requires_distance_and_low_measured_speed():
    success = _success_predicate()
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        target=torch.zeros(3, 3),
        tcp=torch.tensor([[0.049, 0, 0], [0.049, 0, 0], [0.051, 0, 0]]),
        alignment=torch.tensor([0.0, 1.0, 1.0]),
    )
    # The first sample has no previous pose from which to measure speed.
    assert success(env, 0.05, 0.01).tolist() == [False, False, False]
    env.tcp = torch.tensor([[0.049, 0, 0], [0.050, 0, 0], [0.051, 0, 0]])
    assert success(env, 0.05, 0.01).tolist() == [True, False, False]
    assert env._ultrasound_success_last["tcp_speed_m_s"].tolist() == approx([0.0, 0.05, 0.0], abs=1e-6)
    env.tcp[0, 0] = 0.05
    assert success(env, 0.05, 0.01).tolist() == [False, False, False]


def test_preparation_cannot_finish_scan_even_at_target():
    success = _success_predicate()
    env = SimpleNamespace(num_envs=1, device="cpu", _ultrasound_preparing=True)
    assert not success(env, 0.05, 0.01).any()
