"""The disk-backed octree. Only built where PCL has outofcore (it needs
VTK), so these skip otherwise."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._outofcore",
                    reason="this PCL has no outofcore module")


def _cloud(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0, 10, (n, 3)).astype(np.float32)
    pts[:, 2] *= 0.5
    return pts


def test_tree_answers_queries_straight_away(tmp_path):
    """PCL's own tree returns nothing until it is closed and reopened."""
    path = tmp_path / "tree" / "tree.oct_idx"
    tree = pcl.OutofcoreOctree.create(path, (0, 0, 0), (10, 10, 10), depth=3)
    pts = _cloud()
    assert tree.add(pcl.PointCloud(pts)) == len(pts)
    assert tree.depth == 3
    assert tree.get_VoxelSideLength() == pytest.approx(1.25)
    assert tree.get_NumPointsAtDepth(3) == len(pts)
    inside = np.all(pts <= 5, axis=1).sum()
    got = tree.query((0, 0, 0), (5, 5, 5)).to_array()
    assert len(got) == inside and np.all(got <= 5)
    # more points later, then reopen from disk
    tree.add(pcl.PointCloud(_cloud(5000, seed=1)))
    again = pcl.OutofcoreOctree.open(path)
    assert again.query((0, 0, 0), (10, 10, 10)).size == len(pts) + 5000
    lo, hi = again.get_BoundingBox()
    assert np.allclose(lo, 0) and np.allclose(hi, 10)
    centres = again.get_OccupiedVoxelCenters()
    assert centres.shape[1] == 3 and len(centres) == 8 * 8 * 4


def test_level_of_detail(tmp_path):
    path = tmp_path / "lod" / "tree.oct_idx"
    tree = pcl.OutofcoreOctree.create(path, (0, 0, 0), (10, 10, 10), depth=3)
    tree.add(pcl.PointCloud(_cloud()))
    assert tree.get_NumPointsAtDepth(1) == 0
    tree.build_LOD()
    coarse = tree.query((0, 0, 0), (10, 10, 10), depth=1)
    assert 0 < coarse.size < 20000


def test_arguments(tmp_path):
    path = tmp_path / "t" / "tree.oct_idx"
    with pytest.raises(ValueError, match="oct_idx"):
        pcl.OutofcoreOctree.create(tmp_path / "x.idx", (0, 0, 0), (1, 1, 1), 2)
    with pytest.raises(ValueError, match="exceed"):
        pcl.OutofcoreOctree.create(path, (0, 0, 0), (1, 0, 1), 2)
    pcl.OutofcoreOctree.create(path, (0, 0, 0), (1, 1, 1), 2)
    with pytest.raises(ValueError, match="exists"):
        pcl.OutofcoreOctree.create(path, (0, 0, 0), (1, 1, 1), 2)
    with pytest.raises(IOError):
        pcl.OutofcoreOctree.open(tmp_path / "missing.oct_idx")
