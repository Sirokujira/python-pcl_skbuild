"""Runtime tests for hypothesis verification: GreedyVerification,
PapazovHV and GlobalHypothesesVerification (PCL's copied with its
inverted normal test fixed).

Each verifier gets the same job: a scene with one object in it, and two
candidate placements of that object — the right one and one in empty
space. The right one must be kept and the wrong one rejected.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def _box_surface(n, rng):
    """Points on the visible faces of a 0.1 m box (top and two sides)."""
    faces = []
    m = n // 3
    top = rng.rand(m, 3) * [0.1, 0.1, 0.0] + [0.0, 0.0, 0.1]
    side_x = rng.rand(m, 3) * [0.0, 0.1, 0.1]
    side_y = rng.rand(n - 2 * m, 3) * [0.1, 0.0, 0.1]
    faces.extend([top, side_x, side_y])
    return np.vstack(faces)


@pytest.fixture
def scene_and_hypotheses():
    rng = np.random.RandomState(0)
    obj = _box_surface(1500, rng)
    floor = rng.rand(4000, 3) * [0.6, 0.6, 0.0] + [-0.2, -0.2, 0.0]
    placed = obj + [0.1, 0.1, 0.0]
    scene = pcl.PointCloud(np.vstack([floor, placed]).astype(np.float32))
    right = pcl.PointCloud(
        (placed + rng.randn(*placed.shape) * 0.0005).astype(np.float32))
    wrong = pcl.PointCloud((obj + [0.0, 0.0, 0.5]).astype(np.float32))
    return scene, [right, wrong]


def _verify(hv, scene, models):
    hv.set_SceneCloud(scene)
    hv.add_Models(models)
    return hv.verify()


def test_greedy_keeps_the_right_hypothesis(scene_and_hypotheses):
    scene, models = scene_and_hypotheses
    assert _verify(pcl.GreedyVerification(), scene, models) == [True, False]


def test_papazov_keeps_the_right_hypothesis(scene_and_hypotheses):
    scene, models = scene_and_hypotheses
    assert _verify(pcl.PapazovHV(), scene, models) == [True, False]


def test_mask_is_kept_and_reset(scene_and_hypotheses):
    scene, models = scene_and_hypotheses
    hv = pcl.GreedyVerification()
    assert hv.get_Mask() is None
    result = _verify(hv, scene, models)
    assert hv.get_Mask() == result
    hv.add_Models(models[:1])
    assert hv.get_Mask() is None


def test_verifiers_check_their_inputs(scene_and_hypotheses):
    scene, models = scene_and_hypotheses
    hv = pcl.PapazovHV()
    with pytest.raises(RuntimeError):
        hv.verify()
    hv.set_SceneCloud(scene)
    with pytest.raises(RuntimeError):
        hv.verify()
    with pytest.raises(TypeError):
        hv.add_Models([np.zeros((3, 3))])
    with pytest.raises(ValueError):
        hv.add_Models([pcl.PointCloud()])
    with pytest.raises(ValueError):
        hv.set_Resolution(0)
    with pytest.raises(ValueError):
        hv.set_SupportThreshold(1.5)
    with pytest.raises(ValueError):
        pcl.GreedyVerification(lambda_=0)


def test_settings_round_trip():
    hv = pcl.PapazovHV()
    hv.set_Resolution(0.01)
    assert hv.get_Resolution() == pytest.approx(0.01)
    hv.set_InlierThreshold(0.02)
    assert hv.get_InlierThreshold() == pytest.approx(0.02)
    hv.set_OcclusionThreshold(0.03)
    assert hv.get_OcclusionThreshold() == pytest.approx(0.03)
    assert pcl.GreedyVerification(lambda_=2.0).get_Lambda() == pytest.approx(
        2.0)


def test_global_verification_keeps_the_right_hypothesis(
        scene_and_hypotheses):
    """PCL's own keeps both here: its addModel drops the model points
    whose normals are finite (measured in plain C++, 1.14)."""
    scene, models = scene_and_hypotheses
    hv = pcl.GlobalHypothesesVerification()
    assert _verify(hv, scene, models) == [True, False]


def test_global_verification_picks_the_objects_that_are_there():
    """Two boxes on a table, and two copies of the box where there is
    nothing: one on the empty table, one in the air."""
    rng = np.random.RandomState(1)
    obj = _box_surface(1500, rng)
    floor = rng.rand(6000, 3) * [0.8, 0.8, 0.0] + [-0.4, -0.4, 0.0]
    at = [np.array(p) for p in ([0.0, 0.0, 0.0], [-0.25, 0.05, 0.0],
                                [0.1, -0.3, 0.0], [0.5, 0.5, 0.5])]
    scene = pcl.PointCloud(
        np.vstack([floor, obj + at[0], obj + at[1]]).astype(np.float32))
    models = [pcl.PointCloud((obj + a).astype(np.float32)) for a in at]
    hv = pcl.GlobalHypothesesVerification()
    assert _verify(hv, scene, models) == [True, True, False, False]


def test_global_verification_settings(scene_and_hypotheses):
    scene, models = scene_and_hypotheses
    hv = pcl.GlobalHypothesesVerification()
    hv.set_Regularizer(3.0)
    hv.set_RadiusClutter(0.03)
    hv.set_ClutterRegularizer(5.0)
    hv.set_RadiusNormals(0.02)
    hv.set_DetectClutter(False)
    hv.set_MaxIterations(2000)
    hv.set_InitialTemp(500.0)
    hv.set_ResolutionOccupancyGrid(0.01)
    assert _verify(hv, scene, models) == [True, False]
    for setter, bad in ((hv.set_Regularizer, -1), (hv.set_RadiusClutter, 0),
                        (hv.set_ClutterRegularizer, -1),
                        (hv.set_RadiusNormals, 0), (hv.set_MaxIterations, 0),
                        (hv.set_InitialTemp, 0),
                        (hv.set_ResolutionOccupancyGrid, 0)):
        with pytest.raises(ValueError):
            setter(bad)
