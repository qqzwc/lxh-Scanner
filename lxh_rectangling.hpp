#pragma once
#include "lxh_globalWarping.hpp"
#include "lxh_localWarping.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

lxh::Timer timerR;

namespace lxh {
class Rectangling {
private:
    cv::Mat originImg; // 原图像
    cv::Mat img; // 原图像经过缩小后图像
    cv::Mat mask; // 用于判断图像边界的掩码
    cv::Mat warppedImg; // local step扭曲得到的图像
    vector<vector<CoordInt>> displacementField; // 位移场
    vector<vector<CoordDouble>> localMesh; // mesh存放grid-mesh所有顶点位置
    vector<vector<CoordDouble>> globalMesh;
    MeshInfo* meshInfo;
    double imgScaleFactor;
    double lambdaL;
    double lambdaB;
    double sx_avg;
    double sy_avg;
    /*
    生成输入图像的掩码
    */
    void makeMask(const cv::Mat& _img)
    {
        for (int row = 0; row < mask.rows; ++row) {
            for (int col = 0; col < mask.cols; ++col) {
                if (mask.at<uchar>(row, col))
                    mask.at<uchar>(row, col) = 255;
            }
        }
        // cv::imshow("Bin mask", mask);

        // 泛洪填充获得缺失像素区域
        cv::Mat tmpMask = cv::Mat::zeros(mask.rows + 2, mask.cols + 2, mask.type());
        mask.copyTo(tmpMask(cv::Range(1, mask.rows + 1), cv::Range(1, mask.cols + 1)));
        cv::floodFill(tmpMask, cv::Point(0, 0), cv::Scalar(125)); // 填成灰色以便于分辨（之前不是黑就是白）
        // cv::imshow("Tmp mask", tmpMask);
        tmpMask(cv::Range(1, mask.rows + 1), cv::Range(1, mask.cols + 1)).copyTo(mask);
        mask = ~(mask != 125); // 白色部分(255)为缺失像素点
        // 作膨胀腐蚀操作,加了以后能去掉一些噪点
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat dilate_out;
        cv::dilate(mask, dilate_out, element);
        cv::dilate(dilate_out, dilate_out, element);
        cv::dilate(dilate_out, dilate_out, element);
        erode(dilate_out, mask, element);

        cv::imshow("Final mask", mask);
        cv::waitKey(0);

    }
    /*
    局部步
    */
    void localWarping()
    {
        cv::Mat tmpMask; // tmpMask用于可视化中间结果，原mask不改变
        mask.copyTo(tmpMask);
        warppedImg = cv::Mat::zeros(img.size(), CV_8UC3);
        img.copyTo(warppedImg);
        displacementField = localWarping::runLocalWarping(warppedImg, tmpMask);
        // 传的是warppedImg的引用，故不需要再计算warppedImg
        
    }
    /*
    过渡阶段，包括生成grid-mesh以及利用向量场扭曲回原图像
    */
    // 根据meshInfo生成网格
    void makeGridMesh()
    {
        for (int row = 0; row < meshInfo->vertexNumRow; ++row) {
            vector<CoordDouble> meshrow;
            for (int col = 0; col < meshInfo->vertexNumCol; ++col) {
                meshrow.push_back(CoordDouble(row * meshInfo->quadHeight, col * meshInfo->quadWidth));
            }
            localMesh.push_back(meshrow);
        }
    }
    // 根据位移场扭曲格网
    void warpGridMesh()
    {
        // floor 不大于x的最大整数 TODO 这里都是正数，直接int就行吧
        for (int row = 0; row < meshInfo->vertexNumRow; ++row) {
            for (int col = 0; col < meshInfo->vertexNumCol; ++col) {
                CoordDouble& tmpMesh = localMesh[row][col];
                CoordInt displacement = displacementField[int(floor(tmpMesh.row))][int(floor(tmpMesh.col))];
                tmpMesh.row += displacement.row;
                tmpMesh.col += displacement.col;
            }
        }
    }

    void putMeshAndWarpingBack()
    {
        makeGridMesh();
        warpGridMesh();
#ifdef LXH_DEBUG_RECT
        cv::Mat tmpImg;
        img.copyTo(tmpImg);
        drawMesh(tmpImg, localMesh, meshInfo, "localMesh103.jpg");
#endif
    }
    /*
    全局步
    */
    void globalWarping(int iterCount)
    {
        double Nq = meshInfo->meshQuadRow * meshInfo->meshQuadCol; // quad的数量
        /*
        Prepare shape preservation energy
        存储 (A_q(A_q^TA_q)^{-1}A_q^T - I)(仅对角上的8*8子矩阵) 以及 Vq标记的超大矩阵
        */
        SpareseMatrixD_Row shapeEnergyMat = globalWarping::getShapeEnergyMat(localMesh, meshInfo);
        SpareseMatrixD_Row Q = globalWarping::getVertexIndexQ(meshInfo);
        /*
        Prepare line preservation energy
        */
        int Nl = 0; // 线条数量
        vector<pair<int, double>> ID_theta; //论文中theta被quantize成了M=50份，因此id的范围是0-49
        vector<double> rotateThetas; // 存放所有的旋转角度
        vector<pair<MatrixXd, MatrixXd>> inverseBilinearWeights; // 存放逆双线性插值权重
        vector<vector<vector<LineD>>> LineSeg = globalWarping::initLineSegment(img, mask, meshInfo, localMesh, ID_theta, rotateThetas, inverseBilinearWeights, Nl);
        /*
        Prepare boundary constraint energy B中对V的标记的矩阵（对）
        */
        pair<SpareseMatrixD_Row, VectorXd> M_b = globalWarping::getBoundaryMat(meshInfo);
        SpareseMatrixD_Row M = M_b.first;
        VectorXd b = M_b.second;
        SpareseMatrixD_Row lineEnergyMat, shape, boundary, line;
        // 进行交替迭代优化
        for (int iter = 1; iter <= iterCount; ++iter) {
            lineEnergyMat = globalWarping::getLineEnergyMat(img, mask, rotateThetas, LineSeg, inverseBilinearWeights, meshInfo);
            /*
            求解线性方程组更新顶点V
            Q(8 * numQuadRow * numQuadCol, 2 * numVertexRow * numVertexCol)
            shapeEnergyMat(8 * numQuadRow * numQuadCol, 8 * numQuadRow * numQuadCol)
            LineEnergyMat(2 * 线段数量, 8 * meshQuadRow * meshQuadCol)
            M(vertexnum * 2, vertexnum * 2)
            b(vertexnum * 2)
            这玩意也是一个超大的矩阵，对角上存储（2，8）—— C*T，稀疏存储
            */
            SpareseMatrixD_Row shape = 1 / Nq * (shapeEnergyMat * Q); 
            SpareseMatrixD_Row line = lambdaL / Nl * (lineEnergyMat * Q);
            SpareseMatrixD_Row boundary = lambdaB * M;

            SpareseMatrixD_Row K = rowStack(shape, line);
            SpareseMatrixD_Row K2 = rowStack(K, boundary);

            VectorXd bA = VectorXd::Zero(K2.rows());
            bA.tail(b.size()) = lambdaB * b;

            SparseMatrixD K2_trans = K2.transpose();
            SparseMatrixD A = K2_trans * K2;
            VectorXd b = K2_trans * bA;
            VectorXd x;
            CSolve* p_A = new CSolve(A);
            x = p_A->solve(b);

            globalMesh = vector2Mesh(x, meshInfo); // 更新顶点V

            // 更新角度theta
            int tmpLineNum = -1;
            VectorXd thetaGroup = VectorXd::Zero(50);
            VectorXd thetaGroupCnt = VectorXd::Zero(50);
            for (int row = 0; row < meshInfo->meshQuadRow; ++row) {
                for (int col = 0; col < meshInfo->meshQuadCol; ++col) {
                    vector<LineD> linesegInquad = LineSeg[row][col];
                    if (linesegInquad.size() == 0)
                        continue;
                    else {
                        VectorXd Vq = globalWarping::getVq(row, col, globalMesh);
                        for (int k = 0; k < linesegInquad.size(); k++) {
                            ++tmpLineNum;
                            pair<MatrixXd, MatrixXd> Bstartend = inverseBilinearWeights[tmpLineNum];
                            MatrixXd startWeightMat = Bstartend.first;
                            MatrixXd endWeightMat = Bstartend.second;
                            Vector2d newStart = startWeightMat * Vq;
                            Vector2d newEnd = endWeightMat * Vq;

                            double theta = atan((newStart(1) - newEnd(1)) / (newStart(0) - newEnd(0)));
                            double dTheta = theta - ID_theta[tmpLineNum].second;
                            if (isnan(ID_theta[tmpLineNum].second) || isnan(dTheta))
                                continue;
                            if (dTheta > (PI / 2))
                                dTheta -= PI;
                            if (dTheta < (-PI / 2))
                                dTheta += PI;
                            thetaGroup(ID_theta[tmpLineNum].first) += dTheta;
                            thetaGroupCnt(ID_theta[tmpLineNum].first) += 1;
                        }
                    }
                }
            }
            // 对每个bin计算平均旋转角
            for (int i = 0; i < thetaGroup.size(); i++)
                thetaGroup(i) /= thetaGroupCnt(i);
            // 为每个线条赋予新的平均旋转角
            for (int i = 0; i < rotateThetas.size(); i++)
                rotateThetas[i] = thetaGroup[ID_theta[i].first];
        }
    }
    /*
    减少拉伸现象
    */
    void stretchingReduction()
    {
        computeScaling(sx_avg, sy_avg, localMesh, globalMesh, meshInfo); // 计算缩放因子
        meshInfo->reConfig(meshInfo->rows / sy_avg, meshInfo->cols / sx_avg); // 修改目标网格框大小
        reMesh(globalMesh, 1 / sx_avg, 1 / sy_avg, meshInfo);
    }
    /*
    后处理
    */
    void postProcessing()
    {
        reMesh(localMesh, 1.0 / imgScaleFactor, 1.0 / imgScaleFactor, meshInfo); // 因为开始缩小了，获得的网格也是小的，需要放大
        reMesh(globalMesh, 1.0 / imgScaleFactor, 1.0 / imgScaleFactor, meshInfo);
#ifdef LXH_DEBUG_RECT
        drawMesh(originImg, localMesh, meshInfo, "localMesh224.jpg");
        drawMesh(originImg, globalMesh, meshInfo, "globalMesh225.jpg");
#endif
    }

public:
    Rectangling(const cv::Mat& _img, const cv::Mat& _binMask, double _imgScaleFactor, double lambdaL = 100, double lambdaB = INF, int rowMeshNum = 20, int colMeshNum = 20)
    {
        if (_img.empty())
            CV_Error(cv::Error::StsBadArg, "image is empty");
        if (_img.type() != CV_8UC3)
            CV_Error(cv::Error::StsBadArg, "image must have CV_8UC3 type");

        _img.copyTo(originImg);
        _binMask.copyTo(mask);
        imgScaleFactor = _imgScaleFactor;
        cv::resize(originImg, img, cv::Size(0, 0), imgScaleFactor, imgScaleFactor);
        cv::resize(mask, mask, cv::Size(0, 0), imgScaleFactor, imgScaleFactor);
        cout << img.size() << endl;
        makeMask(img);
        meshInfo = new MeshInfo(img.rows, img.cols, rowMeshNum, colMeshNum); //论文中所述400个顶点
        this->lambdaL = lambdaL;
        this->lambdaB = lambdaB;
        sx_avg = 1, sy_avg = 1; 
    }
    ~Rectangling()
    {
        delete meshInfo;
    }
    pair<double, double> runRectangling(int iterCount = 10, bool fixStretch = false)
    {
        timerR.start();
        localWarping();
        timerR.end("Local Warping");
        timerR.start();
        putMeshAndWarpingBack();
        timerR.end("Put Mesh and Warping Back");
        timerR.start();
        globalWarping(iterCount);
        timerR.end("Global Warping " + to_string(iterCount) + " iters");
#ifdef LXH_DEBUG_RECT
        drawMesh(img, localMesh, meshInfo, "localMesh278.jpg");
        drawMesh(img, globalMesh, meshInfo, "globalMesh279.jpg");
#endif
        if (fixStretch) { // 缓解拉伸现象
            timerR.start();
            stretchingReduction();
            timerR.end("Stretching Reduction");
            // timerR.start(); 好像不重新跑效果也差不多
            // globalWarping(10); 
            // timerR.end("Global Warping 10 iters");
        }
        timerR.start();
        postProcessing();
        timerR.end("Post Processing");
        return make_pair(sx_avg, sy_avg);
    }
    vector<vector<CoordDouble>> getLocalMesh()
    {
        return localMesh;
    }
    vector<vector<CoordDouble>> getGlobalMesh()
    {
        return globalMesh;
    }
};

}