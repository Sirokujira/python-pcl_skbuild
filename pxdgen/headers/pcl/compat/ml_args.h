// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/ml_args.h (reached as "pcl/compat/ml_args.h" because
// CMake adds src/ to the includes). Rows are flat, row-major.
#pragma once

#include <string>
#include <vector>

namespace pclcompat {

void kmeans(const std::vector<float>& data, unsigned int n, unsigned int dim,
            unsigned int k, std::vector<float>& centroids, std::vector<int>& labels);

class SupportVectorMachine {
public:
    SupportVectorMachine();
    void setParameters(int svm_type, int kernel_type, int degree, double gamma,
                       double coef0, double nu, double cost, double eps, double p,
                       bool shrinking, bool probability);
    void addSamples(const std::vector<float>& rows, unsigned int n, unsigned int dim,
                    const std::vector<double>& labels);
    std::string train();
    bool ready() const;
    bool save(const std::string& path);
    bool load(const std::string& path);
    void classify(const std::vector<float>& rows, unsigned int n, unsigned int dim,
                  bool probability, std::vector<double>& out, int& width);
    std::vector<int> labels();
};

class FullyConnectedCrf {
public:
    FullyConnectedCrf(int n, int labels);
    void setPositions(const std::vector<int>& xyz);
    void setColours(const std::vector<int>& rgb);
    void setUnaryEnergy(const std::vector<float>& energy);
    void addPairwiseGaussian(float sx, float sy, float sz, float w);
    void addPairwiseBilateral(float sx, float sy, float sz, float sr, float sg,
                              float sb, float w);
    void mapInference(int iterations, float relax, std::vector<int>& labels);
    void inference(int iterations, float relax, std::vector<float>& probabilities);
};

}  // namespace pclcompat
