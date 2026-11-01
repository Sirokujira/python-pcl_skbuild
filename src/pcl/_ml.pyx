# distutils: language = c++
# cython: language_level=3
"""pcl/ml: k-means, the libsvm wrapper (pcl::SVMTrain / SVMClassify)
and the fully connected CRF (pcl::DenseCrf).

PCL's decision trees, forests and ferns are frameworks to be specialised
with C++ feature handlers and are not wrapped (see the README).
Everything goes through pcl/compat/ml_args.h.
"""

from libcpp cimport bool as cbool
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.compat.ml_args cimport (
    FullyConnectedCrf, SupportVectorMachine, kmeans as c_kmeans)


cdef _rows(data, str what="data"):
    import numpy as np
    arr = np.ascontiguousarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("%s must be a non-empty (n, d) array" % what)
    if not np.isfinite(arr).all():
        raise ValueError("%s must be finite" % what)
    return arr


def kmeans(data, unsigned int k):
    """``(centroids, labels)`` from pcl::Kmeans on the rows of `data`.

    PCL starts from a round-robin assignment (row i in cluster i % k) and
    iterates Lloyd steps until no row moves, so the result is
    deterministic and depends on the row order. A cluster that ends up
    empty has a NaN centroid (PCL divides by its zero size)."""
    import numpy as np
    arr = _rows(data)
    if not 1 <= k <= arr.shape[0]:
        raise ValueError("k must be in 1..%d" % arr.shape[0])
    cdef vector[float] flat = arr.ravel().tolist()
    cdef vector[float] centroids
    cdef vector[int] labels
    cdef unsigned int n = arr.shape[0]
    cdef unsigned int d = arr.shape[1]
    with nogil:
        c_kmeans(flat, n, d, k, centroids, labels)
    return (np.asarray(centroids, dtype=np.float32).reshape(k, d),
            np.asarray(labels, dtype=np.int32))


_SVM_TYPES = {"c_svc": 0, "nu_svc": 1, "one_class": 2, "epsilon_svr": 3,
              "nu_svr": 4}
_KERNELS = {"linear": 0, "poly": 1, "rbf": 2, "sigmoid": 3}


cdef class SVM:
    """libsvm through pcl::SVMTrain and pcl::SVMClassify.

        svm = pcl.SVM(kernel="rbf", C=10)
        svm.add_TrainingSamples(features, labels)   # (n, d), (n,)
        svm.train()
        predicted = svm.classify(other_features)

    PCL scales every feature by its largest absolute training value, and
    a `gamma` of 0 becomes 1 / (number of features + 1). With
    `probability`, `classify(..., probability=True)` also returns one
    probability per class (in the order of `labels`). libsvm rejecting
    the parameters is a ValueError here; PCL calls exit(1).
    """

    cdef SupportVectorMachine* me
    cdef int dimension

    def __cinit__(self, str svm_type="c_svc", str kernel="rbf",
                  int degree=3, double gamma=0.0, double coef0=0.0,
                  double nu=0.5, double C=1.0, double eps=1e-3,
                  double p=0.1, bint shrinking=False,
                  bint probability=False):
        if svm_type not in _SVM_TYPES:
            raise ValueError("svm_type must be one of %s"
                             % ", ".join(_SVM_TYPES))
        if kernel not in _KERNELS:
            raise ValueError("kernel must be one of %s" % ", ".join(_KERNELS))
        self.me = new SupportVectorMachine()
        self.me.setParameters(_SVM_TYPES[svm_type], _KERNELS[kernel], degree,
                              gamma, coef0, nu, C, eps, p, shrinking,
                              probability)
        self.dimension = 0

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def add_TrainingSamples(self, features, labels):
        import numpy as np
        arr = _rows(features, "features")
        lab = np.asarray(labels, dtype=np.float64).ravel()
        if lab.shape != (arr.shape[0],) or not np.isfinite(lab).all():
            raise ValueError("one finite label per row")
        if self.dimension and arr.shape[1] != self.dimension:
            raise ValueError("every sample needs %d features" % self.dimension)
        self.dimension = arr.shape[1]
        cdef vector[float] flat = arr.ravel().tolist()
        cdef vector[double] vl = lab.tolist()
        self.me.addSamples(flat, arr.shape[0], arr.shape[1], vl)

    def train(self):
        cdef string error
        with nogil:
            error = self.me.train()
        if error.size():
            raise ValueError(error.decode())

    @property
    def labels(self):
        """The class labels, in the order probabilities are returned."""
        if not self.me.ready():
            return []
        return list(self.me.labels())

    def classify(self, features, bint probability=False):
        import numpy as np
        if not self.me.ready():
            raise RuntimeError("train() or load_Model() is required first")
        arr = _rows(features, "features")
        if self.dimension and arr.shape[1] != self.dimension:
            raise ValueError("every sample needs %d features" % self.dimension)
        cdef vector[float] flat = arr.ravel().tolist()
        cdef vector[double] out
        cdef int width = 0
        cdef unsigned int n = arr.shape[0]
        cdef unsigned int d = arr.shape[1]
        with nogil:
            self.me.classify(flat, n, d, probability, out, width)
        table = np.asarray(out, dtype=np.float64).reshape(n, width)
        if probability:
            if width < 2:
                raise RuntimeError("this model was not trained with "
                                   "probability=True")
            return table[:, 0], table[:, 1:]
        return table[:, 0]

    def save_Model(self, path):
        import os
        if not self.me.save(os.fsencode(os.fspath(path))):
            raise RuntimeError("only a model trained here can be saved")

    def load_Model(self, path):
        import os
        if not os.path.exists(path):
            raise IOError("no such file: %s" % path)
        if not self.me.load(os.fsencode(os.fspath(path))):
            raise IOError("could not read an SVM model from %s" % path)


cdef class DenseCrf:
    """A fully connected CRF solved by mean field (pcl::DenseCrf,
    Kraehenbuehl & Koltun).

        crf = pcl.DenseCrf(n_nodes, n_labels)
        crf.set_Positions(voxel_coordinates)     # (n, 3) int
        crf.set_UnaryEnergy(-np.log(probabilities))   # (n, n_labels)
        crf.add_PairwiseGaussian(3, 3, 3, w=3)
        labels = crf.map_inference(5)

    Positions (and colours, for the bilateral term) are integers -- voxel
    or pixel coordinates; each pairwise term divides them by its own
    standard deviations. Pairwise terms read the positions when added,
    so set those first. PCL prints a line per iteration to std::cout;
    the shim mutes it for the call.
    """

    cdef FullyConnectedCrf* me
    cdef int n
    cdef int m
    cdef bint has_positions
    cdef bint has_colours
    cdef bint has_unary

    def __cinit__(self, int n, int labels):
        if n < 1 or labels < 2:
            raise ValueError("need n >= 1 nodes and >= 2 labels")
        self.me = new FullyConnectedCrf(n, labels)
        self.n = n
        self.m = labels
        self.has_positions = self.has_colours = self.has_unary = False

    def __dealloc__(self):
        del self.me
        self.me = NULL

    cdef vector[int] _triples(self, values, str what) except *:
        import numpy as np
        arr = np.asarray(values)
        if arr.shape != (self.n, 3):
            raise ValueError("%s must be (%d, 3)" % (what, self.n))
        return np.ascontiguousarray(arr, dtype=np.int32).ravel().tolist()

    def set_Positions(self, xyz):
        self.me.setPositions(self._triples(xyz, "positions"))
        self.has_positions = True

    def set_Colours(self, rgb):
        self.me.setColours(self._triples(rgb, "colours"))
        self.has_colours = True

    def set_UnaryEnergy(self, energy):
        """(n, labels) energies -- the negative log probability of each
        label at each node."""
        import numpy as np
        arr = np.asarray(energy, dtype=np.float32)
        if arr.shape != (self.n, self.m) or not np.isfinite(arr).all():
            raise ValueError("energy must be a finite (%d, %d) array"
                             % (self.n, self.m))
        self.me.setUnaryEnergy(arr.ravel().tolist())
        self.has_unary = True

    def add_PairwiseGaussian(self, float sx, float sy, float sz, float w):
        if not self.has_positions:
            raise RuntimeError("set_Positions() is required first")
        if not (sx > 0 and sy > 0 and sz > 0):
            raise ValueError("standard deviations must be > 0")
        self.me.addPairwiseGaussian(sx, sy, sz, w)

    def add_PairwiseBilateral(self, float sx, float sy, float sz, float sr,
                              float sg, float sb, float w):
        if not (self.has_positions and self.has_colours):
            raise RuntimeError("set_Positions() and set_Colours() are "
                               "required first")
        if not (sx > 0 and sy > 0 and sz > 0 and sr > 0 and sg > 0 and sb > 0):
            raise ValueError("standard deviations must be > 0")
        self.me.addPairwiseBilateral(sx, sy, sz, sr, sg, sb, w)

    def map_inference(self, int iterations=5, float relax=1.0):
        """The most likely label of each node after `iterations` of mean
        field."""
        import numpy as np
        if not self.has_unary:
            raise RuntimeError("set_UnaryEnergy() is required first")
        cdef vector[int] labels
        with nogil:
            self.me.mapInference(iterations, relax, labels)
        return np.asarray(labels, dtype=np.int32)

    def inference(self, int iterations=5, float relax=1.0):
        """(n, labels) marginals after `iterations` of mean field."""
        import numpy as np
        if not self.has_unary:
            raise RuntimeError("set_UnaryEnergy() is required first")
        cdef vector[float] probabilities
        with nogil:
            self.me.inference(iterations, relax, probabilities)
        return np.asarray(probabilities, dtype=np.float32).reshape(self.n,
                                                                   self.m)
