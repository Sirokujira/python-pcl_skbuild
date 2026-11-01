// pcl/ml behind flat arrays: Kmeans, the libsvm wrapper, and DenseCrf.
//
// What it works around, each measured against 1.14:
//
// - SVMTrain::trainClassifier() calls exit(1) when libsvm rejects the
//   parameters -- the interpreter would vanish. SupportVectorMachine
//   runs svm_check_parameter on the same problem first and reports the
//   message instead.
// - SVMClassify::setInputTrainingSet appends to what it held, and every
//   call mallocs the rows of a libsvm problem nobody frees (ASan: 60
//   allocations for one train + classify); the classifier is reset and
//   its rows freed after each classification. The trainer's rows must
//   stay: libsvm's model points its support vectors into them.
// - SVMClassify borrows the trainer's model, so the trainer is declared
//   first and outlives it.
// - Kmeans keeps the point -> cluster assignment protected; Clusterer
//   reads it. An empty cluster's centroid is 0 / 0 (NaN); that is left
//   as PCL computes it.
// - DenseCrf::mapInference writes result[i] without sizing `result`;
//   the shim sizes it. Both inference calls print a line per iteration
//   to std::cout, which the shim mutes for the call (C++ streams only).
// - Neither pcl::SVM nor PCL's bundled libsvm is marked PCL_EXPORTS
//   (1.11 through master), so on Windows, where a DLL exports only what
//   is marked, no module outside pcl_ml can link them (MSVC: 12
//   unresolved externals). SupportVectorMachine's constructor throws
//   there; Kmeans and DenseCrf are exported and work everywhere.
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <pcl/ml/densecrf.h>
#include <pcl/ml/kmeans.h>
#ifndef _WIN32
#include <pcl/ml/svm_wrapper.h>
#else
#include <stdexcept>
#endif

namespace pclcompat {

// --- Kmeans -----------------------------------------------------------------

class Clusterer : public pcl::Kmeans {
public:
    using pcl::Kmeans::Kmeans;
    const std::vector<unsigned int>& assignment() const { return points_to_clusters_; }
};

/// pcl::Kmeans on `n` rows of `dim` values (row-major): `k` centroids
/// (row-major) and each row's cluster.
inline void kmeans(const std::vector<float>& data, unsigned int n,
                   unsigned int dim, unsigned int k,
                   std::vector<float>& centroids, std::vector<int>& labels) {
    Clusterer clusterer(n, dim);
    clusterer.setClusterSize(k);
    std::vector<std::vector<float>> rows(n, std::vector<float>(dim));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int d = 0; d < dim; ++d) {
            rows[i][d] = data[i * dim + d];
        }
    }
    clusterer.setInputData(rows);
    clusterer.kMeans();
    centroids.clear();
    for (const auto& c : clusterer.get_centroids()) {
        centroids.insert(centroids.end(), c.begin(), c.end());
    }
    labels.assign(clusterer.assignment().begin(), clusterer.assignment().end());
}

// --- SVM --------------------------------------------------------------------------

#ifndef _WIN32

class CheckedTrainer : public pcl::SVMTrain {
public:
    /// Empty on success, else why (libsvm's own message when it rejects
    /// the parameters).
    std::string checkedTrain() {
        if (training_set_.empty()) {
            return "no training samples";
        }
        scaleFactors(training_set_, scaling_);
        adaptInputToLibSVM(training_set_, prob_);
        if (param_.gamma == 0 && scaling_.max > 0) {
            param_.gamma = 1.0 / scaling_.max;
        }
        const char* error = svm_check_parameter(&prob_, &param_);
        // trainClassifier() builds both again without freeing these
        releaseProblem();
        std::free(scaling_.obj);
        scaling_.obj = nullptr;
        scaling_.max = 0;
        if (error != nullptr) {
            return error;
        }
        return trainClassifier() ? "" : "libsvm could not train a model";
    }

    bool trained() const { return model_.l > 0; }

    // The rows the model's support vectors point into; ~SVM frees only
    // the arrays holding them, and the model never frees borrowed rows.
    ~CheckedTrainer() {
        for (int i = 0; i < prob_.l; ++i) {
            std::free(prob_.x[i]);
        }
    }

private:
    void releaseProblem() {
        for (int i = 0; i < prob_.l; ++i) {
            std::free(prob_.x[i]);
        }
        std::free(prob_.x);
        std::free(prob_.y);
        prob_.x = nullptr;
        prob_.y = nullptr;
        prob_.l = 0;
    }
};

class CheckedClassifier : public pcl::SVMClassify {
public:
    bool loaded() const { return model_.l > 0; }

    void releaseProblem() {
        for (int i = 0; i < prob_.l; ++i) {
            std::free(prob_.x[i]);
        }
        std::free(prob_.x);
        std::free(prob_.y);
        prob_.x = nullptr;
        prob_.y = nullptr;
        prob_.l = 0;
    }
};

/// pcl::SVMTrain + pcl::SVMClassify. Samples are dense rows; libsvm's
/// feature indices start at 1.
class SupportVectorMachine {
public:
    void setParameters(int svm_type, int kernel_type, int degree, double gamma,
                       double coef0, double nu, double cost, double eps, double p,
                       bool shrinking, bool probability) {
        pcl::SVMParam param;
        param.svm_type = svm_type;
        param.kernel_type = kernel_type;
        param.degree = degree;
        param.gamma = gamma;
        param.coef0 = coef0;
        param.nu = nu;
        param.C = cost;
        param.eps = eps;
        param.p = p;
        param.shrinking = shrinking ? 1 : 0;
        param.probability = probability ? 1 : 0;
        trainer_.setParameters(param);
    }

    void addSamples(const std::vector<float>& rows, unsigned int n, unsigned int dim,
                    const std::vector<double>& labels) {
        trainer_.setInputTrainingSet(toData(rows, n, dim, &labels));
    }

    std::string train() {
        const std::string error = trainer_.checkedTrain();
        if (error.empty()) {
            classifier_.setClassifierModel(trainer_.getClassifierModel());
        }
        return error;
    }

    bool ready() const { return classifier_.loaded(); }

    bool save(const std::string& path) {
        if (!trainer_.trained()) {
            return false;
        }
        trainer_.saveClassifierModel(path.c_str());
        return true;
    }

    bool load(const std::string& path) {
        return classifier_.loadClassifierModel(path.c_str());
    }

    /// One row per sample: the predicted label, then (with probability
    /// estimates on a model trained for them) one probability per class.
    void classify(const std::vector<float>& rows, unsigned int n, unsigned int dim,
                  bool probability, std::vector<double>& out, int& width) {
        classifier_.resetTrainingSet();
        classifier_.setProbabilityEstimates(probability);
        classifier_.setInputTrainingSet(toData(rows, n, dim, nullptr));
        classifier_.classification();
        std::vector<std::vector<double>> result;
        classifier_.getClassificationResult(result);
        classifier_.releaseProblem();
        classifier_.resetTrainingSet();
        out.clear();
        width = result.empty() ? 0 : static_cast<int>(result[0].size());
        for (const auto& row : result) {
            out.insert(out.end(), row.begin(), row.end());
        }
    }

    std::vector<int> labels() {
        std::vector<int> l;
        classifier_.getLabel(l);
        return l;
    }

private:
    static std::vector<pcl::SVMData> toData(const std::vector<float>& rows,
                                            unsigned int n, unsigned int dim,
                                            const std::vector<double>* labels) {
        std::vector<pcl::SVMData> data(n);
        for (unsigned int i = 0; i < n; ++i) {
            if (labels != nullptr) {
                data[i].label = (*labels)[i];
            }
            for (unsigned int d = 0; d < dim; ++d) {
                pcl::SVMDataPoint point;
                point.idx = static_cast<int>(d) + 1;
                point.value = rows[i * dim + d];
                data[i].SV.push_back(point);
            }
        }
        return data;
    }

    CheckedTrainer trainer_;        // first: the classifier borrows its model
    CheckedClassifier classifier_;
};

#else

class SupportVectorMachine {
public:
    SupportVectorMachine() {
        throw std::runtime_error(
            "pcl::SVM cannot be linked from outside PCL on Windows (neither "
            "pcl::SVM nor PCL's bundled libsvm is exported from pcl_ml)");
    }
    void setParameters(int, int, int, double, double, double, double, double, double,
                       bool, bool) {}
    void addSamples(const std::vector<float>&, unsigned int, unsigned int,
                    const std::vector<double>&) {}
    std::string train() { return "unavailable on Windows"; }
    bool ready() const { return false; }
    bool save(const std::string&) { return false; }
    bool load(const std::string&) { return false; }
    void classify(const std::vector<float>&, unsigned int, unsigned int, bool,
                  std::vector<double>& out, int& width) {
        out.clear();
        width = 0;
    }
    std::vector<int> labels() { return {}; }
};

#endif

// --- DenseCrf ------------------------------------------------------------------

/// pcl::DenseCrf over `n` nodes and `labels` labels: integer coordinates
/// (n x 3), optional colours (n x 3), unary energies (n x labels, the
/// negative log probability of each label).
class FullyConnectedCrf {
public:
    FullyConnectedCrf(int n, int labels) : crf_(n, labels), n_(n), m_(labels) {}

    void setPositions(const std::vector<int>& xyz) {
        std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> v(n_);
        for (int i = 0; i < n_; ++i) {
            v[i] = Eigen::Vector3i(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
        }
        crf_.setDataVector(v);
    }

    void setColours(const std::vector<int>& rgb) {
        std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> v(n_);
        for (int i = 0; i < n_; ++i) {
            v[i] = Eigen::Vector3i(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
        }
        crf_.setColorVector(v);
    }

    void setUnaryEnergy(const std::vector<float>& energy) { crf_.setUnaryEnergy(energy); }

    void addPairwiseGaussian(float sx, float sy, float sz, float w) {
        crf_.addPairwiseGaussian(sx, sy, sz, w);
    }

    void addPairwiseBilateral(float sx, float sy, float sz, float sr, float sg,
                              float sb, float w) {
        crf_.addPairwiseBilateral(sx, sy, sz, sr, sg, sb, w);
    }

    void mapInference(int iterations, float relax, std::vector<int>& labels) {
        labels.assign(n_, 0);
        Quiet quiet;
        crf_.mapInference(iterations, labels, relax);
    }

    void inference(int iterations, float relax, std::vector<float>& probabilities) {
        probabilities.assign(static_cast<std::size_t>(n_) * m_, 0.f);
        Quiet quiet;
        crf_.inference(iterations, probabilities, relax);
    }

private:
    struct Quiet {
        Quiet() : saved(std::cout.rdbuf(nullptr)) {}
        ~Quiet() {
            std::cout.rdbuf(saved);
            std::cout.clear();
        }
        std::streambuf* saved;
    };

    pcl::DenseCrf crf_;
    int n_;
    int m_;
};

}  // namespace pclcompat
