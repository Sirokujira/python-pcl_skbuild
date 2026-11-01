"""Organized connected components + comparators, multi-plane refinement,
CPC, labelled euclidean clusters, GrabCut and the unary classifier."""

import math

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._segmentation_extra")

H, W, F = 60, 80, 60.0


def _rays():
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    return (u + 0.5 - W / 2) / F, (v + 0.5 - H / 2) / F


def _wall_and_floor():
    """A wall at z = 2 and a floor at y = 0.5 (camera y points down), seen
    by a pinhole camera: rows below ~45 see the floor."""
    dx, dy = _rays()
    t_wall = np.full(dx.shape, 2.0)
    with np.errstate(divide="ignore"):
        t_floor = np.where(dy > 0, 0.5 / dy, np.inf)
    t = np.minimum(t_wall, t_floor)
    xyz = np.dstack([dx * t, dy * t, t]).astype(np.float32)
    floor = t_floor < t_wall
    return xyz, floor


def _organized(xyz):
    cloud = pcl.PointCloud()
    cloud.from_organized_array(np.ascontiguousarray(xyz, dtype=np.float32))
    return cloud


def _normals(cloud):
    ne = pcl.IntegralImageNormalEstimation(cloud)
    ne.set_NormalEstimationMethod(pcl.COVARIANCE_MATRIX)
    ne.set_NormalSmoothingSize(5.0)
    return ne, ne.compute_cloud()


def _big(components, n=50):
    return [c for c in components if len(c) >= n]


@pytest.fixture(scope="module")
def scene():
    xyz, floor = _wall_and_floor()
    cloud = _organized(xyz)
    ne, normals = _normals(cloud)
    return cloud, normals, ne, floor


# --- organized connected components --------------------------------------------


def test_euclidean_comparator_splits_depth_layers():
    depth = np.full((H, W), 1.0, np.float32)
    depth[:, W // 2:] = 2.0
    dx, dy = _rays()
    cloud = _organized(np.dstack([dx * depth, dy * depth, depth]))
    seg = pcl.OrganizedConnectedComponentSegmentation(
        pcl.EuclideanClusterComparator(0.05))
    seg.set_InputCloud(cloud)
    labels, components = seg.segment()
    assert labels.shape == (H, W)
    big = _big(components)
    assert len(big) == 2
    assert sorted(len(c) for c in big) == [H * W // 2, H * W // 2]
    # the label image agrees with the component lists
    for c in big:
        assert len(set(labels.flat[c].tolist())) == 1


def test_invalid_pixels_get_no_label():
    depth = np.full((H, W), 1.0, np.float32)
    depth[10:20, 10:20] = np.nan
    dx, dy = _rays()
    cloud = _organized(np.dstack([dx * depth, dy * depth, depth]))
    seg = pcl.OrganizedConnectedComponentSegmentation(
        pcl.EuclideanClusterComparator(0.05))
    seg.set_InputCloud(cloud)
    labels, components = seg.segment()
    assert np.all(labels[10:20, 10:20] == -1)
    assert len(_big(components)) == 1


def test_excluded_labels_join_nothing():
    """PCL 1.11+ keeps ONLY the labels in its exclude set; the wrapper
    passes the set that excludes, whichever way PCL reads it."""
    assert isinstance(pcl.comparator_reads_exclude_set_as_exclusion(), bool)
    depth = np.full((H, W), 1.0, np.float32)
    dx, dy = _rays()
    cloud = _organized(np.dstack([dx * depth, dy * depth, depth]))
    marks = np.zeros((H, W), np.int64)
    marks[:, W // 2:] = 7
    comparator = pcl.EuclideanClusterComparator(0.05)
    comparator.set_Labels(marks)
    comparator.set_ExcludeLabels([7])
    assert comparator.get_ExcludeLabels() == [7]
    seg = pcl.OrganizedConnectedComponentSegmentation(comparator)
    seg.set_InputCloud(cloud)
    labels, components = seg.segment()
    big = _big(components)
    assert len(big) == 1
    assert np.all(np.asarray(big[0]) % W < W // 2)
    # every excluded pixel is a component of its own
    right = labels[:, W // 2:]
    assert len(np.unique(right)) == right.size


def test_plane_comparator_finds_wall_and_floor(scene):
    cloud, normals, _, floor = scene
    seg = pcl.OrganizedConnectedComponentSegmentation(
        pcl.PlaneCoefficientComparator(normals, math.radians(3.0), 0.02,
                                       False))
    seg.set_InputCloud(cloud)
    labels, components = seg.segment()
    big = sorted(_big(components, 200), key=len, reverse=True)
    assert len(big) == 2
    wall, ground = big
    assert np.mean(floor.flat[wall]) < 0.02
    assert np.mean(floor.flat[ground]) > 0.98


def test_plane_comparator_default_angle_is_honoured(scene):
    """PCL's constructor default would accept normals ~88 degrees apart
    and merge wall and floor; the wrapper always applies the angle."""
    cloud, normals, _, _ = scene
    comparator = pcl.PlaneCoefficientComparator(normals, distance_threshold=5.0)
    assert comparator.get_AngularThreshold() == pytest.approx(math.radians(2))
    seg = pcl.OrganizedConnectedComponentSegmentation(comparator)
    seg.set_InputCloud(cloud)
    _, components = seg.segment()
    assert len(_big(components, 200)) == 2


def test_euclidean_plane_comparator(scene):
    cloud, normals, _, floor = scene
    seg = pcl.OrganizedConnectedComponentSegmentation(
        pcl.EuclideanPlaneCoefficientComparator(normals, math.radians(3.0),
                                                0.1))
    seg.set_InputCloud(cloud)
    _, components = seg.segment()
    big = _big(components, 200)
    assert len(big) == 2


def test_edge_aware_comparator_uses_the_distance_map(scene):
    cloud, normals, ne, floor = scene
    distance_map = ne.get_DistanceMap()
    assert distance_map.shape == (H, W)
    assert np.all(np.isfinite(distance_map))
    comparator = pcl.EdgeAwarePlaneComparator(
        normals, distance_map, math.radians(3.0), 0.02, False,
        euclidean_distance_threshold=0.2, distance_map_threshold=2.0)
    seg = pcl.OrganizedConnectedComponentSegmentation(comparator)
    seg.set_InputCloud(cloud)
    _, components = seg.segment()
    assert len(_big(components, 200)) == 2
    with pytest.raises(ValueError, match="one value per pixel"):
        pcl.EdgeAwarePlaneComparator(normals, distance_map[:10])


def test_ground_plane_comparator_keeps_the_floor(scene):
    cloud, normals, _, floor = scene
    seg = pcl.OrganizedConnectedComponentSegmentation(
        pcl.GroundPlaneComparator(normals, math.radians(3.0),
                                  math.radians(10.0)))
    seg.set_InputCloud(cloud)
    labels, components = seg.segment()
    biggest = max(components, key=len)
    assert len(biggest) > 0.4 * floor.sum()   # the border and the seam
    # have no normals at smoothing size 5; the wall faces the camera, so
    # every wall pixel fails the ground test and joins nothing
    for c in components:
        if len(c) > 1:
            assert np.all(floor.flat[c])
    with pytest.raises(ValueError):
        pcl.GroundPlaneComparator(normals, expected_ground_normal=(0, 0, 0))


def test_rgb_plane_comparator_splits_colours(scene):
    cloud, normals, _, _ = scene
    xyz, _ = _wall_and_floor()
    rgba = np.full((H, W, 4), 255, np.uint8)
    rgba[:, :W // 2, :3] = (255, 0, 0)
    rgba[:, W // 2:, :3] = (0, 0, 255)
    coloured = pcl.PointCloud_PointXYZRGBA()
    coloured.from_organized_rgba_array(xyz, rgba)
    # distance_threshold is euclidean here; pixels are ~3 cm apart
    comparator = pcl.RGBPlaneCoefficientComparator(
        normals, 30.0, math.radians(3.0), 0.1, False)
    seg = pcl.OrganizedConnectedComponentSegmentation(comparator)
    seg.set_InputCloud(coloured)
    _, components = seg.segment()
    assert len(_big(components, 200)) == 4   # wall / floor x red / blue
    seg.set_InputCloud(cloud)
    with pytest.raises(ValueError, match="PointXYZRGBA"):
        seg.segment()


def test_segmentation_checks_its_inputs(scene):
    cloud, normals, _, _ = scene
    seg = pcl.OrganizedConnectedComponentSegmentation()
    with pytest.raises(RuntimeError, match="input cloud"):
        seg.segment()
    with pytest.raises(ValueError, match="organized"):
        seg.set_InputCloud(pcl.PointCloud(np.zeros((5, 3), np.float32)))
    seg.set_InputCloud(cloud)
    with pytest.raises(RuntimeError, match="comparator"):
        seg.segment()
    small = _organized(np.ones((4, 4, 3), np.float32))
    seg.set_Comparator(pcl.PlaneCoefficientComparator(normals))
    seg.set_InputCloud(small)
    with pytest.raises(ValueError, match="normals"):
        seg.segment()
    with pytest.raises(ValueError, match="radians"):
        pcl.PlaneCoefficientComparator(normals, 10.0)


def test_region_boundary_traces_the_outline():
    labels = np.zeros((10, 12), np.int64)
    labels[2:6, 3:8] = 5
    last = 5 * 12 + 7                 # last pixel of the region
    boundary = pcl.OrganizedConnectedComponentSegmentation \
        .find_LabeledRegionBoundary(labels, last)
    rows, cols = np.divmod(np.asarray(boundary), 12)
    assert set(labels[rows, cols].tolist()) == {5}
    # the 4 x 5 block's outline: 14 pixels, interior excluded
    assert len(set(boundary)) == 14
    assert pcl.OrganizedConnectedComponentSegmentation \
        .find_LabeledRegionBoundary(np.zeros((5, 5), np.int64), 12) == []


# --- multi-plane refinement -------------------------------------------------------


def test_segment_and_refine_reports_regions(scene):
    cloud, normals, _, floor = scene
    omps = pcl.OrganizedMultiPlaneSegmentation(cloud)
    omps.set_InputNormals(normals)
    omps.set_MinInliers(200)
    omps.set_AngularThreshold(math.radians(3.0))
    omps.set_DistanceThreshold(0.02)
    planes, inliers = omps.segment()
    result = omps.segment_and_refine(0.01)
    assert len(result["coefficients"]) == len(planes) == 2
    assert result["labels"].shape == (H, W)
    assert result["centroids"].shape == (2, 3)
    assert result["covariances"].shape == (2, 3, 3)
    assert all(len(b) > 0 for b in result["boundaries"])
    # refinement only ever adds pixels
    for before, after in zip(inliers, result["inliers"]):
        assert set(before) <= set(after)
    # plane i is labelled labels.flat[inliers[i][0]] throughout
    for one in result["inliers"]:
        assert len(set(result["labels"].flat[one].tolist())) == 1
    floor_plane = int(np.argmax([abs(c[1]) for c in result["coefficients"]]))
    assert result["centroids"][floor_plane][1] == pytest.approx(0.5, abs=0.02)


# --- CPC ----------------------------------------------------------------------------


def _two_touching_boxes():
    """Two boxes side by side, touching along a concave seam."""
    rng = np.random.RandomState(8)
    pts = []
    for x0 in (0.0, 1.0):
        faces = rng.rand(6, 900, 2)
        for axis in range(3):
            for side in (0.0, 1.0):
                f = faces[axis * 2 + int(side)]
                p = np.empty((900, 3))
                others = [a for a in range(3) if a != axis]
                p[:, axis] = side
                p[:, others[0]] = f[:, 0]
                p[:, others[1]] = f[:, 1]
                p[:, 0] += x0
                pts.append(p)
    xyz = np.vstack(pts).astype(np.float32)
    rgba = np.full((len(xyz), 4), 255, np.uint8)
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_rgba_array(xyz, rgba)
    return cloud, xyz


def test_cpc_matches_lccp_shape():
    cloud, xyz = _two_touching_boxes()
    sv = cloud.make_SupervoxelClustering(0.05, 0.2)
    sv.set_UseSingleCameraTransform(False)
    with pytest.raises(RuntimeError, match="extract"):
        sv.cpc()
    labels, _, _, _ = sv.extract()
    point_labels, mapping = sv.cpc(max_cuts=5, cutting_min_segments=3,
                                   cutting_min_score=0.1)
    assert point_labels.shape == (cloud.size,)
    assert set(mapping) == set(labels.tolist())
    assert len(set(point_labels.tolist()) - {0}) >= 1


def test_cpc_on_xyz_supervoxels():
    _, xyz = _two_touching_boxes()
    sv = pcl.PointCloud(xyz).make_SupervoxelClustering(0.05, 0.2)
    sv.set_UseSingleCameraTransform(False)
    sv.extract()
    point_labels, mapping = sv.cpc(max_cuts=2)
    assert point_labels.shape == (len(xyz),)
    assert mapping


# --- labelled euclidean clusters ------------------------------------------------------


def test_labeled_clusters_split_by_label_and_space():
    rng = np.random.RandomState(1)
    blobs = [rng.rand(100, 3) * 0.1 + offset
             for offset in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0])]
    xyz = np.vstack(blobs).astype(np.float32)
    labels = np.repeat([3, 3, 9, 9], 100)
    # a fifth group: same place as blob 0, different label
    xyz = np.vstack([xyz, (rng.rand(50, 3) * 0.1).astype(np.float32)])
    labels = np.concatenate([labels, np.full(50, 4)])
    ext = pcl.LabeledEuclideanClusterExtraction(pcl.PointCloud(xyz), labels)
    ext.set_ClusterTolerance(0.05)
    ext.set_MinClusterSize(10)
    clusters = ext.Extract()
    assert sorted(clusters) == [3, 4, 9]
    assert [len(c) for c in clusters[3]] == [100, 100]
    assert [len(c) for c in clusters[9]] == [100, 100]
    assert [len(c) for c in clusters[4]] == [50]
    for label, groups in clusters.items():
        for group in groups:
            assert set(labels[group].tolist()) == {label}


def test_labeled_clusters_check_inputs():
    ext = pcl.LabeledEuclideanClusterExtraction(
        pcl.PointCloud(np.zeros((5, 3), np.float32)))
    with pytest.raises(RuntimeError, match="set_Labels"):
        ext.Extract()
    with pytest.raises(ValueError):
        ext.set_Labels([-1, 0, 0, 0, 0])
    ext.set_Labels([0, 0, 1, 1, 1])
    with pytest.raises(RuntimeError, match="Tolerance"):
        ext.Extract()


# --- GrabCut --------------------------------------------------------------------------


def test_grabcut_separates_an_object_by_colour():
    rng = np.random.RandomState(4)
    h, w = 40, 50
    v, u = np.mgrid[0:h, 0:w]
    xyz = np.dstack([u * 0.01, v * 0.01, np.ones((h, w))]).astype(np.float32)
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., 3] = 255
    background = np.array([40, 60, 200]) + rng.randint(-20, 20, (h, w, 3))
    obj = np.array([220, 50, 40]) + rng.randint(-20, 20, (h, w, 3))
    inside = (v >= 12) & (v < 28) & (u >= 15) & (u < 35)
    rgba[..., :3] = np.where(inside[..., None], obj, background).clip(0, 255)
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz, rgba)
    region = np.flatnonzero((v >= 6) & (v < 34) & (u >= 8) & (u < 42))
    grabcut = pcl.GrabCut()
    assert grabcut.get_K() == 5 and grabcut.get_NumberOfNeighbours() == 9
    grabcut.set_InputCloud(cloud)
    with pytest.raises(RuntimeError, match="ForegroundRegion"):
        grabcut.extract()
    grabcut.set_ForegroundRegion(region)
    foreground = np.asarray(grabcut.extract())
    truth = np.flatnonzero(inside)
    overlap = len(np.intersect1d(foreground, truth))
    assert overlap / len(truth) > 0.95            # recall
    # the initial foreground model is fitted on the whole region, two
    # thirds of it background, so some noisy background pixels stay in
    assert overlap / len(foreground) > 0.6        # precision
    # nothing outside the region can be foreground
    assert set(foreground.tolist()) <= set(region.tolist())


# --- unary classifier -----------------------------------------------------------------


def _plane_and_sphere(rng, n=1500):
    plane = np.column_stack([rng.rand(n) * 0.6, rng.rand(n) * 0.6,
                             np.zeros(n)])
    d = rng.randn(n, 3)
    sphere = d / np.linalg.norm(d, axis=1, keepdims=True) * 0.12 + [1.0, 0.3, 0]
    return np.vstack([plane, sphere]).astype(np.float32), n


def test_unary_classifier_tells_plane_from_sphere():
    rng = np.random.RandomState(6)
    xyz, n = _plane_and_sphere(rng)
    labels = np.repeat([0, 1], n)
    classifier = pcl.UnaryClassifier()
    classifier.set_ClusterSize(4)
    classifier.set_NormalRadiusSearch(0.05)
    classifier.set_FPFHRadiusSearch(0.08)
    trained = classifier.train_with_labels(pcl.PointCloud(xyz), labels)
    assert sorted(trained) == [0, 1]
    assert all(t.shape == (4, 33) for t in trained.values())
    # PCL's k-means leaves NaN for an empty cluster; the wrapper repairs it
    assert all(np.all(np.isfinite(t)) for t in trained.values())
    whole = classifier.train(pcl.PointCloud(xyz))
    assert whole.shape == (4, 33)

    classifier.set_TrainedFeatures([trained[0], trained[1]])
    classifier.set_FeatureThreshold(1e9)
    test_xyz, _ = _plane_and_sphere(np.random.RandomState(7))
    assigned = classifier.segment(pcl.PointCloud(test_xyz))
    assert assigned.shape == (len(test_xyz),)
    assert np.mean(assigned[:n] == 0) > 0.8
    assert np.mean(assigned[n:] == 1) > 0.8
    # a threshold nothing passes: every point unmatched
    classifier.set_FeatureThreshold(0.0)
    assert np.all(classifier.segment(pcl.PointCloud(test_xyz)) == -1)


def test_unary_classifier_checks_inputs():
    classifier = pcl.UnaryClassifier()
    cloud = pcl.PointCloud(np.random.rand(20, 3).astype(np.float32))
    with pytest.raises(RuntimeError, match="ClusterSize"):
        classifier.train(cloud)
    classifier.set_ClusterSize(4)
    with pytest.raises(ValueError, match="at least"):
        classifier.train_with_labels(cloud, [0] * 18 + [1, 1])
    with pytest.raises(RuntimeError, match="TrainedFeatures"):
        classifier.segment(cloud)
    with pytest.raises(ValueError, match="same number"):
        classifier.set_TrainedFeatures([np.zeros((4, 33)), np.zeros((3, 33))])
    with pytest.raises(ValueError, match="finite"):
        classifier.set_TrainedFeatures([np.full((4, 33), np.nan)])


def test_random_walker_follows_the_weights():
    # two triangles joined by one weak edge, one seed in each
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    weights = [1, 1, 1, 1, 1, 1, 0.01]
    seeds = [5, 0, 0, 0, 0, 9]
    colours, potentials, columns = pcl.random_walker(edges, weights, seeds)
    assert colours.tolist() == [5, 5, 5, 9, 9, 9]
    assert sorted(columns) == [5, 9]
    assert potentials.shape == (6, 2)
    assert np.allclose(potentials.sum(axis=1), 1.0, atol=1e-5)
    five = columns.index(5)
    assert potentials[0, five] == pytest.approx(1.0)
    assert potentials[2, five] > 0.9


def test_random_walker_leaves_isolated_vertices_unseeded():
    colours, potentials, _ = pcl.random_walker([(0, 1)], [1.0], [3, 0, 0])
    assert colours.tolist() == [3, 3, 0]
    with pytest.raises(ValueError, match="seeded"):
        pcl.random_walker([(0, 1)], [1.0], [0, 0])
    with pytest.raises(ValueError, match="positive"):
        pcl.random_walker([(0, 1)], [0.0], [1, 0])
    with pytest.raises(ValueError, match="vertices"):
        pcl.random_walker([(0, 5)], [1.0], [1, 0])
