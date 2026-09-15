# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka Panda sweeps an ultrasound probe across an abdominal phantom.

The only openpi PI0 env. Nothing about the workflow says so — the backend is a
manifest lookup, so swapping to a GR00T checkpoint would be a one-string edit.
"""

from __future__ import annotations

import numpy as np

from i4h_common.ultrasound_scan import PROBE_SCAN_LOCAL_WXYZ, SWEEP
from i4h_engine.graph import TaskGraph, node
from i4h_engine.interface import Workflow
from i4h_tasks.basic.control.hold import Hold
from i4h_tasks.basic.control.wait_until import WaitUntil
from i4h_tasks.basic.perception.locate import Locate
from i4h_tasks.ik.approach import Approach
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy
from i4h_workflow_modes.replay import replay
from i4h_workflow_modes.teleop import teleop

def rule_based() -> TaskGraph:
    """Scan XYZ after the Scene has aligned and landed the probe before recording."""
    locate = node(Locate("organs", name="locate"))
    graph = TaskGraph(description="Fixed-orientation XYZ sweep starting in physical contact.").flow(locate)
    previous = locate
    last_sweep_index = len(SWEEP) - 1
    for index, offset in enumerate(SWEEP):
        is_final_pose = index == last_sweep_index
        waypoint = node(
            Approach(
                standoff=offset,
                local_standoff=True,
                orientation=PROBE_SCAN_LOCAL_WXYZ,
                local_orientation=True,
                interpolate_orientation=False,
                duration_s=1.0,
                position_stall_timeout_s=1.0,
                # Only the final pose actually reached on the liver (the last
                # sweep waypoint) needs tighter precision; the earlier
                # waypoints are just passed through on the way there.
                # orientation_tolerance is new (MoveToPose had no orientation
                # check at all before); 10 degrees is an untested starting
                # point, not a value tuned against real convergence data.
                position_tolerance=0.03 if is_final_pose else 0.05,
                orientation_tolerance=np.radians(10.0) if is_final_pose else None,
                settle_timeout_s=4.0,
                name=f"sweep_{index}",
            )
        )
        graph.flow(previous >> waypoint)
        graph.wire(locate.out.pose, waypoint.in_.target)
        previous = waypoint

    # Allow the existing scan-duration gate to finish within the unchanged scene cap.
    hold = node(Hold(0.2, name="hold"))
    verify = node(WaitUntil(success, timeout_s=5.0, name="verify_scan"))
    graph.flow(previous >> hold >> verify)
    return graph


def success(ctx) -> object:
    return ctx.scene.termination("success")


WORKFLOW = Workflow(
    scene="panda_phantom",
    success=success,
    modes={
        # task_id override (--task-id) selects which backend serves this mode,
        # e.g. us_dp/ultrasound_liver_scan for the diffusion-over-splines
        # policy instead of the default openpi_pi0 checkpoint.
        "policy": lambda task_id="openpi_pi0/ultrasound_liver_scan": policy(task_id, until=success),
        "rule-based": rule_based,
        "teleop": lambda device="keyboard", **kwargs: teleop(device, until=success, **kwargs),
        "replay": replay,
        "idle": idle,
    },
)
