"""Exercise reset helpers without starting Isaac or importing its cfg graph."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def helper(name, namespace):
    path = Path(__file__).parents[1] / 'i4h_arena/envcfg/panda_phantom.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    node.args.defaults = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_preparation_cannot_timeout_but_scan_can():
    timeout = helper('ultrasound_time_out', {'torch': torch, 'base_mdp': SimpleNamespace(time_out=lambda env: torch.tensor([True]))})
    env = SimpleNamespace(_ultrasound_preparing=True, num_envs=1, device='cpu')
    assert not timeout(env).any()
    env._ultrasound_preparing = False
    assert timeout(env).all()


def test_joint_reset_replaces_failed_attempt_motor_targets():
    calls = {}
    asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(1, 7),
        default_joint_vel=torch.zeros(1, 7), default_joint_limits=torch.tensor([[[-2., 2.]] * 7])))
    for name in ('write_joint_state_to_sim_index', 'set_joint_position_target_index',
                 'set_joint_velocity_target_index', 'set_joint_effort_target_index'):
        setattr(asset, name, lambda _name=name, **kwargs: calls.update({_name: kwargs}))
    reset = helper('reset_panda_joints_by_fraction_of_limits', {'torch': torch, 'to_torch': lambda x: x})
    reset(SimpleNamespace(scene={'robot': asset}), torch.tensor([0]), SimpleNamespace(name='robot'), 0.01)
    torch.testing.assert_close(calls['set_joint_position_target_index']['target'], calls['write_joint_state_to_sim_index']['position'])
    assert not calls['set_joint_velocity_target_index']['target'].any()
    assert not calls['set_joint_effort_target_index']['target'].any()
