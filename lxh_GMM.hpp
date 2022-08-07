/*
三元高斯混合模型的实现
部分参考opencv源码：https://github.com/opencv/opencv/blob/e628fd7bce2b5d64c36b5bdc55a37c0ae78bc907/modules/imgproc/src/grabcut.cpp
*/
#pragma once
#include "lxh_macro.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <fstream>
using namespace std;

namespace lxh {

#ifdef LXH_OUTPUT
fstream fGMMLog("./output/lxh_GMM.log", ios::out);
#endif

class GMM3D {
public:
    static const int componentsCount = 5; // 默认使用5个高斯分量

    GMM3D(cv::Mat& _modelParams);
    double prob(const cv::Vec3d color) const;
    double probCi(int ci, const cv::Vec3d color) const;
    int whichComponent(const cv::Vec3d color) const;

    void initLearning();
    void addSample(int ci, const cv::Vec3d color);
    void endLearning();

private:
    void calcInverseCovAndDeterm(int ci, double singularFix);
    cv::Mat modelParams; // 模型参数，包括coefs, mean, cov
    double* coefs; // 指向第一个比例系数的指针
    double* mean; // 指向第一个均值的指针
    double* cov; // 指向第一个协方差矩阵的指针

    double inverseCovs[componentsCount][3][3]; // 协方差矩阵的逆矩阵
    double covDeterms[componentsCount]; // 协方差矩阵的行列式

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

/*
构造函数，使用一个1x13的Mat初始化GMM的参数
一个高斯分量都有13个参数：1个比例系数，3个均值，3*3个协方差
*/
GMM3D::GMM3D(cv::Mat& _modelParams)
{
    const int modelSize = 13; //component weight:1 + mean:3 + covariance:9
    if (_modelParams.empty()) {
        _modelParams.create(1, modelSize * componentsCount, CV_64FC1);
        _modelParams.setTo(cv::Scalar(0));
    } else if ((_modelParams.type() != CV_64FC1) || (_modelParams.rows != 1) || (_modelParams.cols != modelSize * componentsCount))
        CV_Error(cv::Error::StsBadArg, "_modelParams must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");

    modelParams = _modelParams;

    coefs = modelParams.ptr<double>(0); 
    mean = coefs + componentsCount;
    cov = mean + 3 * componentsCount;

    for (int ci = 0; ci < componentsCount; ci++){
        if (coefs[ci] > 0)
            calcInverseCovAndDeterm(ci, 0.0);
    }
    totalSampleCount = 0;
}

/*
求协方差矩阵的行列式和逆矩阵，由于只是3x3的矩阵，这里使用伴随矩阵求逆：A^{−1}= A^∗ / |A| 
*/
void GMM3D::calcInverseCovAndDeterm(int ci, const double singularFix)
{   
    if (coefs[ci] > 0) {
        double* c = cov + 9 * ci;
        double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        if (dtrm <= 1e-6 && singularFix > 0) {
            // Adds the white noise to avoid singular covariance matrix.
            c[0] += singularFix;
            c[4] += singularFix;
            c[8] += singularFix; // 观察下一行，加0，4，8的话可以保证原来为0的dtrm大于0
            dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        }
        covDeterms[ci] = dtrm;

        CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
        double inv_dtrm = 1.0 / dtrm;
        inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
        inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
        inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
        inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
        inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
        inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
        inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
        inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
        inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;
    }
}

/////////////////// EM算法中的Expectation部分 ///////////////////

/*
计算像素点属于该高斯混合模型的概率
*/
double GMM3D::prob(const cv::Vec3d color) const
{
    double res = 0;
    for (int ci = 0; ci < componentsCount; ci++)
        res += coefs[ci] * probCi(ci, color);
    return res;
}

/*
计算像素点属于第ci个高斯分量的概率
*/
double GMM3D::probCi(int ci, const cv::Vec3d color) const
{
    double res = 0;
    if (coefs[ci] > 0) {
        CV_Assert(covDeterms[ci] > std::numeric_limits<double>::epsilon());
        cv::Vec3d diff = color;
        double* m = mean + 3 * ci;
        diff[0] -= m[0];
        diff[1] -= m[1];
        diff[2] -= m[2];
        double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
            + diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
            + diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
        res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
    }
    return res;
}

/*
判断像素点最有可能属于哪一个高斯分量
*/
int GMM3D::whichComponent(const cv::Vec3d color) const
{
    int k = 0;
    double max = 0;

    for (int ci = 0; ci < componentsCount; ci++) {
        double p = probCi(ci, color);
        if (p > max) {
            k = ci;
            max = p;
        }
    }
    return k;
}

/////////////////// EM算法中的Maximization部分 ///////////////////

/*
清空之前累计的累加和累乘，准备重新开始计算GMM的参数
*/
void GMM3D::initLearning()
{
    for (int ci = 0; ci < componentsCount; ci++) {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

/*
为指定的高斯分量增加样本数
*/
void GMM3D::addSample(int ci, const cv::Vec3d color)
{
    sums[ci][0] += color[0];
    sums[ci][1] += color[1];
    sums[ci][2] += color[2];
    prods[ci][0][0] += color[0] * color[0];
    prods[ci][0][1] += color[0] * color[1];
    prods[ci][0][2] += color[0] * color[2];
    prods[ci][1][0] += color[1] * color[0];
    prods[ci][1][1] += color[1] * color[1];
    prods[ci][1][2] += color[1] * color[2];
    prods[ci][2][0] += color[2] * color[0];
    prods[ci][2][1] += color[2] * color[1];
    prods[ci][2][2] += color[2] * color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

/*
使用之前累计的累加和累乘，重新计算GMM的参数
*/
void GMM3D::endLearning()
{
    CV_Assert(totalSampleCount > 0);
    for (int ci = 0; ci < componentsCount; ci++) {
        int n = sampleCounts[ci];
        if (n == 0)
            coefs[ci] = 0;
        else {
            double inv_n = 1.0 / n;
            coefs[ci] = (double)n / totalSampleCount;

            double* m = mean + 3 * ci;
            m[0] = sums[ci][0] * inv_n;
            m[1] = sums[ci][1] * inv_n;
            m[2] = sums[ci][2] * inv_n;
            // COV(X,Y) = E(X-EX)(Y-EY) = E(XY) - EXEY
            double* c = cov + 9 * ci;
            c[0] = prods[ci][0][0] * inv_n - m[0] * m[0];
            c[1] = prods[ci][0][1] * inv_n - m[0] * m[1];
            c[2] = prods[ci][0][2] * inv_n - m[0] * m[2];
            c[3] = prods[ci][1][0] * inv_n - m[1] * m[0];
            c[4] = prods[ci][1][1] * inv_n - m[1] * m[1];
            c[5] = prods[ci][1][2] * inv_n - m[1] * m[2];
            c[6] = prods[ci][2][0] * inv_n - m[2] * m[0];
            c[7] = prods[ci][2][1] * inv_n - m[2] * m[1];
            c[8] = prods[ci][2][2] * inv_n - m[2] * m[2];

            calcInverseCovAndDeterm(ci, 0.01);
        }
    }
    #ifdef LXH_OUTPUT
    fGMMLog << "---------------------------------------------------------------" << endl;
    fGMMLog << "coefs: " <<endl;
    for(int ci = 0; ci < componentsCount; ++ci){
        fGMMLog << coefs[ci] << endl;
    }
    fGMMLog << "mean: " << endl;
    for(int ci = 0; ci < componentsCount; ++ci){
        double* m = mean + 3 * ci;
        fGMMLog << m[0] << " " << m[1] << " " << m[2] << endl;
    }
    fGMMLog << "cov: " << endl;
    for(int ci = 0; ci < componentsCount; ++ci){
        double* c = cov + 9 * ci;
        for(int i = 0; i < 8; ++i)
            fGMMLog << c[i] << " ";
        fGMMLog << c[8] << endl;
    }
    #endif
}

}