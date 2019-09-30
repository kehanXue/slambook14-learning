//
// Created by kehan on 2019/9/30.
//

#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    // TODO
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }


    void oplusImpl(const double* update) override
    {
        _estimate += Eigen::Vector3d(update);
    }


    bool read(istream &in) override
    {

    }


    bool write(ostream &out) const override
    {

    }
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    explicit CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x)
    {

    }


    void computeError() override
    {
        const auto* v = dynamic_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d &abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }


    bool read(istream &in) override
    {

    }


    bool write(ostream &out) const override
    {

    }


public:
    double _x;
};


int main(int argc, char** argv)
{
    double a = 1.;
    double b = 2.;
    double c = 1.;

    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[3] = {0, 0, 0};

    vector<double> x_data, y_data;

    cout << "Generating data: " << endl;
    for (int i = 0; i < N; ++i)
    {
        double x = i / 100.;
        x_data.emplace_back(x);
        y_data.emplace_back(
                exp(a * x * x + b * x + c) + rng.gaussian(w_sigma)
        );
        cout << x_data.at(i) << " " << y_data.at(i) << endl;
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;
    Block::LinearSolverType* linear_solver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    auto* solver_ptr = new Block(linear_solver);

    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);


    auto* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    for (int i = 0; i < N; ++i)
    {
        auto* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data.at(i));
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }


    cout << "Start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Solve time cost = " << time_used.count() << "s" << endl;

    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "Estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}


