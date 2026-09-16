"""Check plan marker updates without starting Isaac Sim."""

import numpy as np

from i4h_arena.adapters.scene_view import ArenaSceneView


class Markers:
    def __init__(self):
        self.visible = False
        self.calls = []

    def is_visible(self):
        return self.visible

    def set_visibility(self, visible):
        self.visible = visible

    def visualize(self, *, translations, marker_indices):
        self.calls.append((translations.copy(), marker_indices.copy()))


def test_policy_path_marker_colors_and_cleanup():
    view = object.__new__(ArenaSceneView)
    view._policy_path_markers = Markers()
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    view.visualize_policy_path(points, current_index=1)
    actual, indices = view._policy_path_markers.calls[-1]
    np.testing.assert_array_equal(actual, points)
    np.testing.assert_array_equal(indices, [0, 1, 2])
    assert view._policy_path_markers.visible
    view.clear_policy_path()
    assert not view._policy_path_markers.visible
