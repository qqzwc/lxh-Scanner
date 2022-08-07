#pragma once
#include "lxh_utils.hpp"
#include <fstream>

#ifdef LXH_OUT
fstream fDebugL("./debugLocal.log", ios::out);
#endif
namespace lxh {

lxh::Timer timerL; // 计时器
namespace localWarping {
    enum BoundDirection { // 最长缺失像素边界的位置
        BOUND_TOP,
        BOUND_BOTTOM,
        BOUND_LEFT,
        BOUND_RIGHT
    };

    //通过mask来判断是否透明，就是判断是否缺失吧，寻找seam carving的子图用的判断
    bool isMissing(const cv::Mat& mask, int row, int col)
    {
        return mask.at<uchar>(row, col) != 0; // mask的白色部分就是缺失像素
    }

    /*
    找到最长的缺失像素边界段（longest Boundary Segment），返回起点和终点（就记录两个int值就可以，因为有方向了）
    这里假设mask的形状就是目标矩形框的形状
    */
    pair<int, int> findLongestBoundary(const cv::Mat& mask, BoundDirection& direction)
    {
        int rows = mask.rows;
        int cols = mask.cols;
        int sIndex = -1; // 起点位置
        int maxLength = 0;
        // left
        int tmpIndex = -1, tmpLength = -1;
        bool isCounting = false;
        for (int row = 0; row < rows; ++row) {
            if (isMissing(mask, row, 0)) { // 该点缺失，开始计数或者继续上一计数
                if (isCounting) {
                    ++tmpLength;
                } else {
                    tmpIndex = row;
                    tmpLength = 1;
                    isCounting = true;
                }
            } else { // 终止计数并判断当前计数结果是否超过已知最大值
                if (tmpLength > maxLength) {
                    maxLength = tmpLength;
                    sIndex = tmpIndex;
                    direction = BOUND_LEFT;
                }
                tmpIndex = -1, tmpLength = -1, isCounting = false;
            }
        }
        if (tmpLength > maxLength) { // 最后检查一把
            maxLength = tmpLength;
            sIndex = tmpIndex;
            direction = BOUND_LEFT;
        }
        // right
        tmpIndex = -1, tmpLength = -1, isCounting = false;
        for (int row = 0; row < rows; ++row) {
            if (isMissing(mask, row, cols - 1)) { // 该点缺失，开始计数或者继续上一计数
                if (isCounting) {
                    ++tmpLength;
                } else {
                    tmpIndex = row;
                    tmpLength = 1;
                    isCounting = true;
                }
            } else { // 终止计数并判断当前计数结果是否超过已知最大值
                if (tmpLength > maxLength) {
                    maxLength = tmpLength;
                    sIndex = tmpIndex;
                    direction = BOUND_RIGHT;
                }
                tmpIndex = -1, tmpLength = -1, isCounting = false;
            }
        }
        if (tmpLength > maxLength) { // 最后检查一把
            maxLength = tmpLength;
            sIndex = tmpIndex;
            direction = BOUND_RIGHT;
        }
        // top
        tmpIndex = -1, tmpLength = -1, isCounting = false;
        for (int col = 0; col < cols; col++) {
            if (isMissing(mask, 0, col)) { // 该点缺失，开始计数或者继续上一计数
                if (isCounting) {
                    ++tmpLength;
                } else {
                    tmpIndex = col;
                    tmpLength = 1;
                    isCounting = true;
                }
            } else { // 终止计数并判断当前计数结果是否超过已知最大值
                if (tmpLength > maxLength) {
                    maxLength = tmpLength;
                    sIndex = tmpIndex;
                    direction = BOUND_TOP;
                }
                tmpIndex = -1, tmpLength = -1, isCounting = false;
            }
        }
        if (tmpLength > maxLength) { // 最后检查一把
            maxLength = tmpLength;
            sIndex = tmpIndex;
            direction = BOUND_TOP;
        }
        // bottom
        tmpIndex = 0, tmpLength = -1, isCounting = false;
        isCounting = false;
        for (int col = 0; col < cols; col++) {
            if (isMissing(mask, rows - 1, col)) { // 该点缺失，开始计数或者继续上一计数
                if (isCounting) {
                    ++tmpLength;
                } else {
                    tmpIndex = col;
                    tmpLength = 1;
                    isCounting = true;
                }
            } else { // 终止计数并判断当前计数结果是否超过已知最大值
                if (tmpLength > maxLength) {
                    maxLength = tmpLength;
                    sIndex = tmpIndex;
                    direction = BOUND_BOTTOM;
                }
                tmpIndex = -1, tmpLength = -1, isCounting = false;
            }
        }
        if (tmpLength > maxLength) { // 最后检查一把
            maxLength = tmpLength;
            sIndex = tmpIndex;
            direction = BOUND_BOTTOM;
        }
        // cout << maxLength << endl;

        if (maxLength == 0)
            return make_pair(0, 0);
        return make_pair(sIndex, maxLength);
    }

#ifdef LXH_DEBUG_LOCAL
    /*
    将最长边界标红
    */
    void showLongestBoundary(cv::Mat src, pair<int, int> start_length, BoundDirection direction)
    {
        cv::Mat tmpsrc;
        src.copyTo(tmpsrc);
        int rows = src.rows;
        int cols = src.cols;
        int begin = start_length.first;
        int end = start_length.first + start_length.second;
        switch (direction) {
        case BOUND_LEFT:
            for (int row = begin; row < end; row++)
                tmpsrc.at<colorPixel>(row, 0) = colorPixel(0, 0, 255);
            break;
        case BOUND_RIGHT:
            for (int row = begin; row < end; row++)
                tmpsrc.at<colorPixel>(row, cols - 1) = colorPixel(0, 0, 255);
            break;
        case BOUND_TOP:
            for (int col = begin; col < end; col++)
                tmpsrc.at<colorPixel>(0, col) = colorPixel(0, 0, 255);
            break;
        case BOUND_BOTTOM:
            for (int col = begin; col < end; col++)
                tmpsrc.at<colorPixel>(rows - 1, col) = colorPixel(0, 0, 255);
            break;
        default:
            break;
        }

        cv::namedWindow("Local Warping", cv::WINDOW_AUTOSIZE);
        cv::imshow("Local Warping", tmpsrc);
        cv::waitKey(0);
    }

    /*
        对seam执行插入操作
        */
    inline cv::Mat insertSeam(cv::Mat& src, cv::Mat& seamImg, cv::Mat& mask, int* seam, BoundDirection direction, pair<int, int> start_length)
    {
        // 减少代码量的一个trick,如果方向是top或者bottom就转置图像一下,当成left或者right处理,相当于只考虑vertical seam
        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) { // seam也是根据情况转置过的
            cv::transpose(src, src);
            cv::transpose(seamImg, seamImg);
            cv::transpose(mask, mask);
        }

        cv::Mat newImg;
        src.copyTo(newImg);

        int begin = start_length.first; //最长border所在的local row范围
        int end = begin + start_length.second;

        int rows = src.rows;
        int cols = src.cols;

        for (int row = begin; row < end; row++) {
            int localRow = row - begin;
            if (direction == BOUND_LEFT || direction == BOUND_TOP) // top转置后后等价于left,都是往左移
                for (int col = 0; col < seam[localRow]; col++) {
                    newImg.at<colorPixel>(row, col) = src.at<colorPixel>(row, col + 1);
                    seamImg.at<colorPixel>(row, col) = seamImg.at<colorPixel>(row, col + 1);
                    mask.at<uchar>(row, col) = mask.at<uchar>(row, col + 1);
                }
            else
                for (int col = cols - 1; col > seam[localRow]; col--) {
                    newImg.at<colorPixel>(row, col) = src.at<colorPixel>(row, col - 1);
                    seamImg.at<colorPixel>(row, col) = seamImg.at<colorPixel>(row, col - 1);
                    mask.at<uchar>(row, col) = mask.at<uchar>(row, col - 1);
                }

            //把seam上的值处理了（两边平均），边界情况单独处理
            mask.at<uchar>(row, seam[localRow]) = 0;
            if (seam[localRow] == 0)
                newImg.at<colorPixel>(row, seam[localRow]) = src.at<colorPixel>(row, seam[localRow] + 1); // 使用右邻居填充边界
            else {
                if (seam[localRow] == cols - 1)
                    newImg.at<colorPixel>(row, seam[localRow]) = src.at<colorPixel>(row, seam[localRow] - 1); // 使用左邻居填充边界
                else {
                    colorPixel pixel1 = src.at<colorPixel>(row, seam[localRow] + 1);
                    colorPixel pixel2 = src.at<colorPixel>(row, seam[localRow] - 1);
                    newImg.at<colorPixel>(row, seam[localRow]) = 0.5 * pixel1 + 0.5 * pixel2; // 使用左右邻居均值填充边界
                }
            }
            seamImg.at<colorPixel>(row, seam[localRow]) = colorPixel(224, 236, 251); // 设置seam的颜色和原文差不多，bgr这是
        }

        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) { //如果是转置过的要转置回来
            cv::transpose(newImg, newImg);
            cv::transpose(seamImg, seamImg);
            cv::transpose(mask, mask);
        }

        // cv::namedWindow("insert_seam", cv::WINDOW_AUTOSIZE);
        // cv::imshow("insert_seam", newImg);
        // cv::waitKey(0);

        return newImg;
    }
#else

    /*
        对seam执行插入操作
        */
    inline cv::Mat insertSeam(cv::Mat& src, cv::Mat& mask, int* seam, BoundDirection direction, pair<int, int> start_length)
    {
        // 减少代码量的一个trick,如果方向是top或者bottom就转置图像一下,当成left或者right处理,相当于只考虑vertical seam
        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) { // seam也是根据情况转置过的
            cv::transpose(src, src);
            cv::transpose(mask, mask);
        }

        cv::Mat newImg;
        src.copyTo(newImg);

        int begin = start_length.first; //最长border所在的local row范围
        int end = begin + start_length.second;

        int rows = src.rows;
        int cols = src.cols;

        for (int row = begin; row < end; row++) {
            int localRow = row - begin;
            if (direction == BOUND_LEFT || direction == BOUND_TOP) // top转置后后等价于left,都是往左移
                for (int col = 0; col < seam[localRow]; col++) {
                    newImg.at<colorPixel>(row, col) = src.at<colorPixel>(row, col + 1);
                    mask.at<uchar>(row, col) = mask.at<uchar>(row, col + 1);
                }
            else
                for (int col = cols - 1; col > seam[localRow]; col--) {
                    newImg.at<colorPixel>(row, col) = src.at<colorPixel>(row, col - 1);
                    mask.at<uchar>(row, col) = mask.at<uchar>(row, col - 1);
                }

            //把seam上的值处理了（两边平均），边界情况单独处理
            mask.at<uchar>(row, seam[localRow]) = 0;
            if (seam[localRow] == 0)
                newImg.at<colorPixel>(row, seam[localRow]) = src.at<colorPixel>(row, seam[localRow] + 1); // 使用右邻居填充边界
            else {
                if (seam[localRow] == cols - 1)
                    newImg.at<colorPixel>(row, seam[localRow]) = src.at<colorPixel>(row, seam[localRow] - 1); // 使用左邻居填充边界
                else {
                    colorPixel pixel1 = src.at<colorPixel>(row, seam[localRow] + 1);
                    colorPixel pixel2 = src.at<colorPixel>(row, seam[localRow] - 1);
                    newImg.at<colorPixel>(row, seam[localRow]) = 0.5 * pixel1 + 0.5 * pixel2; // 使用左右邻居均值填充边界
                }
            }
        }

        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) { //如果是转置过的要转置回来
            cv::transpose(newImg, newImg);
            cv::transpose(mask, mask);
        }

        return newImg;
    }

#endif

    /*
    用sobel算子在灰度图像上计算梯度
    */
    inline cv::Mat SobelImg(cv::Mat gray)
    {
        cv::Mat grad_x, grad_y, dst;
        cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
        return dst;
    }

    // 从代码复用的角度确实不应该写两个，但是合在一起写的话因为多余的一点运算对速度有一些影响
    inline int* getOriginSeam(cv::Mat src, cv::Mat mask, BoundDirection direction, pair<int, int> start_length, cv::Mat& penalty)
    {
        // 水平方向只需将图片转置就可以用处理垂直方法处理, 这里传的形参不是引用是值,所以不会改变外面的src和mask,因此本函数末尾不需要转置回去
        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) {
            cv::transpose(src, src);
            cv::transpose(mask, mask);
            cv::transpose(penalty, penalty);
        }

        // 寻找竖直的seam
        int rows = src.rows;
        int cols = src.cols;

        int rowStart = start_length.first;
        int rowEnd = rowStart + start_length.second;

        int seamLength = rowEnd - rowStart;
        // 垂直seam
        int colStart = 0;
        int colEnd = cols;

        cv::Mat subImg = src(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd));
        cv::Mat subMask = mask(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd)); // Range 左闭右开
        // 像素点本身的能量，原始seam算法使用，对于Improved算法来说为0
        cv::Mat localEnergy(seamLength, cols, CV_32FC1, cv::Scalar(0));

        cv::Mat gray; // 两种能量都是在灰度图上计算的
        cv::cvtColor(subImg, gray, cv::COLOR_BGR2GRAY);
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, localEnergy, CV_32F);

        cv::Mat dpEnergy; //这个指的是seam的累计能量
        localEnergy.copyTo(dpEnergy); // improved情况下初始化为0

//非图中部分能量置为无穷大
#ifdef LXH_PARALLEL
#pragma omp parallel for
#endif
        for (int row = 0; row < seamLength; ++row) {
            for (int col = 0; col < cols; ++col) {
                dpEnergy.at<float>(row, col) += penalty.at<float>(rowStart + row, colStart + col);
            }
        }

        for (int row = 0; row < seamLength; row++) {
            for (int col = 0; col < cols; col++) {
                if ((int)subMask.at<uchar>(row, col) == 255) {
                    dpEnergy.at<float>(row, col) = INF;
                }
            }
        }

        // DP计算最优seam
        for (int row = 1; row < seamLength; row++) {
            for (int col = 0; col < cols; col++) {
                if (col == 0) {
                    dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col), dpEnergy.at<float>(row - 1, col + 1));
                } else if (col == (cols - 1)) {
                    dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col - 1), dpEnergy.at<float>(row - 1, col));
                } else {
                    dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col), min(dpEnergy.at<float>(row - 1, col - 1), dpEnergy.at<float>(row - 1, col + 1)));
                }
            }
        }

        // saveMat("dpEnergy.txt", dpEnergy);
        //找最小seam能量
        float minEnergy = dpEnergy.at<float>(seamLength - 1, 0);
        int minLoc = 0;
        for (int col = 1; col < cols; ++col) {
            if (dpEnergy.at<float>(seamLength - 1, col) < minEnergy) {
                minEnergy = dpEnergy.at<float>(seamLength - 1, col);
                minLoc = col;
            }
        }
        int* seam = new int[seamLength];
        seam[seamLength - 1] = minLoc;

        // 回溯获得整条seam
        for (int row = seamLength - 1; row > 0; --row) {
            int col = seam[row];
            if (col == 0) {
                if ((dpEnergy.at<float>(row - 1, col + 1)) < (dpEnergy.at<float>(row - 1, col)))
                    seam[row - 1] = col + 1;
                else
                    seam[row - 1] = col;
            } else if (col == (cols - 1)) {
                if ((dpEnergy.at<float>(row - 1, col - 1)) < (dpEnergy.at<float>(row - 1, col)))
                    seam[row - 1] = col - 1;
                else
                    seam[row - 1] = col;
            } else {
                float leftVal = dpEnergy.at<float>(row - 1, col - 1);
                float upVal = dpEnergy.at<float>(row - 1, col);
                float rightVal = dpEnergy.at<float>(row - 1, col + 1);
                if (leftVal < upVal) {
                    if (upVal < rightVal) {
                        seam[row - 1] = col - 1;
                    } else {
                        if (leftVal < rightVal) {
                            seam[row - 1] = col - 1;
                        } else {
                            seam[row - 1] = col + 1;
                        }
                    }
                } else { // upVal <= leftVal
                    if (leftVal < rightVal) {
                        seam[row - 1] = col;
                    } else {
                        if (upVal < rightVal) {
                            seam[row - 1] = col;
                        } else {
                            seam[row - 1] = col + 1;
                        }
                    }
                }
            }
        }

        for (int row = seamLength - 1; row > 0; --row) { // 在seam上面添加惩罚
            int col = seam[row];
            penalty.at<float>(rowStart + row, colStart + col) = INF / 10.0;
        }

        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) {
            cv::transpose(penalty, penalty);
        }

        // TODO 这里算的seam是相对值，如果以后有不是 colStart = 0 colEnd = cols的情况需要加上偏移
        return seam;
    }

    inline int* getImprovedSeam(cv::Mat src, cv::Mat mask, BoundDirection direction, pair<int, int> start_length, cv::Mat& penalty)
    {
        // 水平方向只需将图片转置就可以用处理垂直方法处理, 这里传的形参不是引用是值,所以不会改变外面的src和mask,因此本函数末尾不需要转置回去
        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) {
            cv::transpose(src, src);
            cv::transpose(mask, mask);
            cv::transpose(penalty, penalty);
        }

        // 寻找竖直的seam
        int rows = src.rows;
        int cols = src.cols;

        int rowStart = start_length.first;
        int rowEnd = rowStart + start_length.second;

        int seamLength = rowEnd - rowStart;
        // 垂直seam
        int colStart = 0;
        int colEnd = cols;

        cv::Mat subImg = src(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd));
        cv::Mat subMask = mask(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd)); // Range 左闭右开

        // Forward Energy，improved算法使用
        cv::Mat cU, cL, cR;

        cv::Mat gray; // 两种能量都是在灰度图上计算的
        cv::cvtColor(subImg, gray, cv::COLOR_BGR2GRAY);

        cv::Mat filterU = (cv::Mat_<int>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
        cv::Mat filterL = (cv::Mat_<int>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 0);
        cv::Mat filterR = (cv::Mat_<int>(3, 3) << 0, 1, 0, 0, 0, -1, 0, 0, 0);

        filter2D(gray, cU, CV_32F, filterU);
        filter2D(gray, cL, CV_32F, filterL);
        filter2D(gray, cR, CV_32F, filterR);
        cU = abs(cU);
        cL = abs(cL) + cU;
        cR = abs(cR) + cU;

        cv::Mat dpEnergy(seamLength, cols, CV_32FC1, cv::Scalar(0));

//非图中部分能量置为无穷大
#ifdef LXH_PARALLEL
#pragma omp parallel for
#endif
        for (int row = 0; row < seamLength; ++row) {
            for (int col = 0; col < cols; ++col) {
                dpEnergy.at<float>(row, col) += penalty.at<float>(rowStart + row, colStart + col);
            }
        }

        for (int row = 0; row < seamLength; row++) {
            for (int col = 0; col < cols; col++) {
                if ((int)subMask.at<uchar>(row, col) == 255) {
                    dpEnergy.at<float>(row, col) = INF;
                }
            }
        }

        // DP计算最优seam
        for (int row = 1; row < seamLength; row++) {
            for (int col = 0; col < cols; col++) {
                if (col == 0) {
                    dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col), dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col));
                } else if (col == (cols - 1)) {
                    dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col), dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col));
                } else {
                    dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col), min(dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col), dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col)));
                }
            }
        }

        // saveMat("dpEnergy.txt", dpEnergy);
        //找最小seam能量
        float minEnergy = dpEnergy.at<float>(seamLength - 1, 0);
        int minLoc = 0;
        for (int col = 1; col < cols; ++col) {
            if (dpEnergy.at<float>(seamLength - 1, col) < minEnergy) {
                minEnergy = dpEnergy.at<float>(seamLength - 1, col);
                minLoc = col;
            }
        }
        int* seam = new int[seamLength];
        seam[seamLength - 1] = minLoc;

        // 回溯获得整条seam
        for (int row = seamLength - 1; row > 0; --row) {
            int col = seam[row];
            if (col == 0) {
                if ((dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col)) < (dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col)))
                    seam[row - 1] = col + 1;
                else
                    seam[row - 1] = col;
            } else if (col == (cols - 1)) {
                if ((dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col)) < (dpEnergy.at<float>(row - 1, col) + +cU.at<float>(row, col)))
                    seam[row - 1] = col - 1;
                else
                    seam[row - 1] = col;
            } else {
                float leftVal = dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col);
                float upVal = dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col);
                float rightVal = dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col);
                if (leftVal < upVal) {
                    if (upVal < rightVal) {
                        seam[row - 1] = col - 1;
                    } else {
                        if (leftVal < rightVal) {
                            seam[row - 1] = col - 1;
                        } else {
                            seam[row - 1] = col + 1;
                        }
                    }
                } else { // upVal <= leftVal
                    if (leftVal < rightVal) {
                        seam[row - 1] = col;
                    } else {
                        if (upVal < rightVal) {
                            seam[row - 1] = col;
                        } else {
                            seam[row - 1] = col + 1;
                        }
                    }
                }
            }
        }

        for (int row = seamLength - 1; row > 0; --row) { // 在seam上面添加惩罚
            int col = seam[row];
            penalty.at<float>(rowStart + row, colStart + col) += INF / 10.0;
        }

        if (direction == BOUND_TOP || direction == BOUND_BOTTOM) {
            cv::transpose(penalty, penalty);
        }

        // TODO 这里算的seam是相对值，如果以后有不是 colStart = 0 colEnd = cols的情况需要加上偏移
        return seam;
    }
    /*
    原始seam carving算法以及Improved carving实现（TODO，待完善）,返回一个cols值数组，不用这个版本，因为分开写速度更快 TODO
    */
    //     inline int* getSeam(cv::Mat src, cv::Mat mask, BoundDirection direction, pair<int, int> start_length, bool useImprovedSeam)
    //     {
    //         // 水平方向只需将图片转置就可以用处理垂直方法处理, 这里传的形参不是引用是值,所以不会改变外面的src和mask,因此本函数末尾不需要转置回去
    //         if (direction == BOUND_TOP || direction == BOUND_BOTTOM) {
    //             cv::transpose(src, src);
    //             cv::transpose(mask, mask);
    //         }

    //         // 寻找竖直的seam
    //         int rows = src.rows;
    //         int cols = src.cols;

    //         int rowStart = start_length.first;
    //         int rowEnd = rowStart + start_length.second;

    //         int seamLength = rowEnd - rowStart;
    //         // 垂直seam
    //         int colStart = 0;
    //         int colEnd = cols;

    //         cv::Mat subImg = src(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd));
    //         cv::Mat subMask = mask(cv::Range(rowStart, rowEnd), cv::Range(colStart, colEnd)); // Range 左闭右开
    //         // 像素点本身的能量，原始seam算法使用，对于Improved算法来说为0
    //         cv::Mat localEnergy(seamLength, cols, CV_32FC1, cv::Scalar(0));

    //         // Forward Energy，improved算法使用，对于原始算法来说为0
    //         cv::Mat cU(seamLength, cols, CV_32FC1, cv::Scalar(0));
    //         cv::Mat cL(seamLength, cols, CV_32FC1, cv::Scalar(0));
    //         cv::Mat cR(seamLength, cols, CV_32FC1, cv::Scalar(0));

    //         cv::Mat gray; // 两种能量都是在灰度图上计算的
    //         cv::cvtColor(subImg, gray, cv::COLOR_BGR2GRAY);

    //         if (useImprovedSeam) {
    //             cv::Mat U = (cv::Mat_<int>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
    //             cv::Mat L_ = (cv::Mat_<int>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 0);
    //             cv::Mat R_ = (cv::Mat_<int>(3, 3) << 0, 1, 0, 0, 0, -1, 0, 0, 0);

    //             filter2D(gray, cU, CV_32F, U);
    //             filter2D(gray, cL, CV_32F, U);
    //             filter2D(gray, cR, CV_32F, U);
    //             cU = abs(cU);
    //             cL = abs(cL) + cU;
    //             cR = abs(cR) + cU;
    //         } else {
    //             cv::Mat grad_x, grad_y;
    //             cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    //             cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    //             cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, localEnergy, CV_32F);
    //         }
    //         cv::Mat dpEnergy; //这个指的是seam的累计能量
    //         localEnergy.copyTo(dpEnergy); // improved情况下初始化为0

    // //非图中部分能量置为无穷大
    // #ifdef LXH_PARALLEL
    // #pragma omp parallel for
    // #endif
    //         for (int row = 0; row < seamLength; row++) {
    //             for (int col = 0; col < cols; col++) {
    //                 if ((int)subMask.at<uchar>(row, col) == 255) {
    //                     dpEnergy.at<float>(row, col) = INF;
    //                 }
    //             }
    //         }

    //         // DP计算最优seam
    //         for (int row = 1; row < seamLength; row++) {
    //             for (int col = 0; col < cols; col++) {
    //                 if (col == 0) {
    //                     dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col), dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col));
    //                 } else if (col == (cols - 1)) {
    //                     dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col), dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col));
    //                 } else {
    //                     dpEnergy.at<float>(row, col) += min(dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col), min(dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col), dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col)));
    //                 }
    //             }
    //         }

    //         // saveMat("dpEnergy.txt", dpEnergy);
    //         //找最小seam能量
    //         float minEnergy = dpEnergy.at<float>(seamLength - 1, 0);
    //         int minLoc = 0;
    //         for (int col = 1; col < cols; ++col) {
    //             if (dpEnergy.at<float>(seamLength - 1, col) < minEnergy) {
    //                 minEnergy = dpEnergy.at<float>(seamLength - 1, col);
    //                 minLoc = col;
    //             }
    //         }
    //         int* seam = new int[seamLength];
    //         seam[seamLength - 1] = minLoc;

    //         // 回溯获得整条seam
    //         for (int row = seamLength - 1; row > 0; --row) {
    //             int col = seam[row];
    //             if (col == 0) {
    //                 if ((dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col)) < (dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col)))
    //                     seam[row - 1] = col + 1;
    //                 else
    //                     seam[row - 1] = col;
    //             } else if (col == (cols - 1)) {
    //                 if ((dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col)) < (dpEnergy.at<float>(row - 1, col) + +cU.at<float>(row, col)))
    //                     seam[row - 1] = col - 1;
    //                 else
    //                     seam[row - 1] = col;
    //             } else {
    //                 float leftVal = dpEnergy.at<float>(row - 1, col - 1) + cL.at<float>(row, col);
    //                 float upVal = dpEnergy.at<float>(row - 1, col) + cU.at<float>(row, col);
    //                 float rightVal = dpEnergy.at<float>(row - 1, col + 1) + cR.at<float>(row, col);
    //                 if (leftVal < upVal) {
    //                     if (upVal < rightVal) {
    //                         seam[row - 1] = col - 1;
    //                     } else {
    //                         if (leftVal < rightVal) {
    //                             seam[row - 1] = col - 1;
    //                         } else {
    //                             seam[row - 1] = col + 1;
    //                         }
    //                     }
    //                 } else { // upVal <= leftVal
    //                     if (leftVal < rightVal) {
    //                         seam[row - 1] = col;
    //                     } else {
    //                         if (upVal < rightVal) {
    //                             seam[row - 1] = col;
    //                         } else {
    //                             seam[row - 1] = col + 1;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         // TODO 这里算的seam是相对值，如果以后有不是 colStart = 0 colEnd = cols的情况需要加上偏移
    //         return seam;
    //     }

    /*
      执行local warping，将原图扭曲为矩形，并记录位移场
    */
    vector<vector<CoordInt>> runLocalWarping(cv::Mat& warpImg, cv::Mat& mask, bool useImprovedSeam = false)
    {
#ifdef LXH_DEBUG_LOCAL
        cv::Mat seamImg; // 记录Seam的图片
        warpImg.copyTo(seamImg); // 将待扭曲的图先复制一份,之后用seam盖上去
#endif
        int rows = warpImg.rows;
        int cols = warpImg.cols;
        vector<vector<CoordInt>> displacementField(rows, vector<CoordInt>(cols, CoordInt(0, 0))); // 这个用来存储最终位移矩阵
        vector<vector<CoordInt>> newDisplacementField(rows, vector<CoordInt>(cols, CoordInt(0, 0))); // 这个用作临时更新

        BoundDirection direction;
        cv::Mat penalty = cv::Mat::zeros(rows, cols, CV_32FC1);

        while (true) {
            // timerL.start();
            
            pair<int, int> start_length = findLongestBoundary(mask, direction); //每次都选择最长的边界
// timerL.end("Find longest Boundary");
// cv::imshow("Mask", mask);
#ifdef LXH_DEBUG_LOCAL
            showLongestBoundary(seamImg, start_length, direction);
#endif
            if (start_length.second == 0) { // 找不到缺失像素了
#ifdef LXH_DEBUG_LOCAL
                cv::imwrite("./results/localWarpingImg.png", warpImg); //处理完之后保存这一步的结果
                cv::imwrite("./results/seamImg.png", seamImg);
#endif
                return displacementField;
            } else {
                // timerL.start();
                int* seam;
                if (useImprovedSeam)
                    seam = getImprovedSeam(warpImg, mask, direction, start_length, penalty);
                else
                    seam = getOriginSeam(warpImg, mask, direction, start_length, penalty);
// timerL.end("Get Seam");
// timerL.start();
#ifdef LXH_DEBUG_LOCAL
                warpImg = insertSeam(warpImg, seamImg, mask, seam, direction, start_length);
#else
                warpImg = insertSeam(warpImg, mask, seam, direction, start_length);
#endif
                // timerL.end("Insert Seam");
                // timerL.start();
                //更新位移场矩阵
                bool isVertical = (direction == BOUND_LEFT || direction == BOUND_RIGHT);
                bool move2Right = (direction == BOUND_RIGHT || direction == BOUND_BOTTOM); // 向右移
                int begin = start_length.first;
                int end = begin + start_length.second;
#ifdef LXH_PARALLEL
#pragma omp parallel for
#endif
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        int dCol = 0, dRow = 0; // 本次位移量
                        if (isVertical && row >= begin && row < end) {
                            int localRow = row - begin;
                            if (move2Right) {
                                if (col > seam[localRow])
                                    dCol = -1;
                            } else {
                                if (col < seam[localRow])
                                    dCol = 1;
                            }
                        } else {
                            if (!isVertical && col >= begin && col < end) {
                                int localCol = col - begin;
                                if (move2Right) {
                                    if (row > seam[localCol])
                                        dRow = -1;
                                } else {
                                    if (row < seam[localCol])
                                        dRow = 1;
                                }
                            }
                        }
                        int tmpDisplaceRow = row + dRow; // 计算即将到当前位置的像素点（新像素）位置
                        int tmpDisplaceCol = col + dCol; // 并得到他的位移场
                        CoordInt displacementOfTarget = displacementField[tmpDisplaceRow][tmpDisplaceCol];
                        // 进而获得新像素原来的位置
                        int rowInOrigin = tmpDisplaceRow + displacementOfTarget.row;
                        int colInOrigin = tmpDisplaceCol + displacementOfTarget.col;
                        CoordInt& newDisplacement = newDisplacementField[row][col];
                        // 存储新像素从当前位置回到原来的位置需要的位移场
                        newDisplacement.row = rowInOrigin - row;
                        newDisplacement.col = colInOrigin - col;
                    }
                }

#ifdef LXH_PARALLEL
#pragma omp parallel for
#endif
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        CoordInt& displacement = displacementField[row][col];
                        CoordInt newDisplacement = newDisplacementField[row][col];
                        displacement.row = newDisplacement.row;
                        displacement.col = newDisplacement.col;
                    }
                }
                // timerL.end("Other");
                if (seam != nullptr)
                    delete[] seam; // 释放空间
            }
        }
        throw "Seam carving 异常终止";
    }

}
}