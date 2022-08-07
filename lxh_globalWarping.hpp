#pragma once
#include "lsd.cpp"
#include "lsd.h"
#include "lxh_utils.hpp"
namespace lxh {
namespace globalWarping {

    /*
    获取顶点坐标向量
    */
    VectorXd getVq(int row, int col, vector<vector<CoordDouble>> mesh)
    {
        // x0,y0,x1,y1...（左上 右上 左下 右下）,row为y轴，col为x轴
        VectorXd Vq = VectorXd::Zero(8);
        CoordDouble p0 = mesh[row][col]; //左上
        CoordDouble p1 = mesh[row][col + 1]; //右上
        CoordDouble p2 = mesh[row + 1][col]; //左下
        CoordDouble p3 = mesh[row + 1][col + 1]; //右下
        Vq << p0.col, p0.row, p1.col, p1.row, p2.col, p2.row, p3.col, p3.row;
        return Vq;
    }

    ////////////////////////////// Shape Preservation Energy //////////////////////////////

    /*
    计算Es(Q)中的存储(Aq...... - I)全部的超大矩阵 V的顺序都是（左上 右上 左下 右下）
    */
    SpareseMatrixD_Row getShapeEnergyMat(vector<vector<CoordDouble>> mesh, MeshInfo* meshInfo)
    {
        int numVertexRow = meshInfo->vertexNumRow;
        int numVertexCol = meshInfo->vertexNumCol;
        int numQuadRow = meshInfo->meshQuadRow;
        int numQuadCol = meshInfo->meshQuadCol;
        // row为y轴，col为x轴 Vq=[x0,y0,x1,y1...]，矩阵太大了所以用了稀疏矩阵压缩
        SpareseMatrixD_Row shapeEnergyMat(8 * numQuadRow * numQuadCol, 8 * numQuadRow * numQuadCol);
        for (int row = 0; row < numQuadRow; row++) {
            for (int col = 0; col < numQuadCol; col++) {
                CoordDouble p0 = mesh[row][col]; //左上
                CoordDouble p1 = mesh[row][col + 1]; //右上
                CoordDouble p2 = mesh[row + 1][col]; //左下
                CoordDouble p3 = mesh[row + 1][col + 1]; //右下
                MatrixXd Aq(8, 4);
                Aq << p0.col, -p0.row, 1, 0,
                    p0.row, p0.col, 0, 1,
                    p1.col, -p1.row, 1, 0,
                    p1.row, p1.col, 0, 1,
                    p2.col, -p2.row, 1, 0,
                    p2.row, p2.col, 0, 1,
                    p3.col, -p3.row, 1, 0,
                    p3.row, p3.col, 0, 1;

                MatrixXd Aq_trans = Aq.transpose(); // Aq^T
                MatrixXd Aq_trans_mul_Aq_inverse = (Aq_trans * Aq).inverse(); //(Aq^TAq)^-1
                MatrixXd I = MatrixXd::Identity(8, 8); //单位阵I
                MatrixXd coeff = (Aq * (Aq_trans_mul_Aq_inverse)*Aq_trans - I);

                int left_top_x = (row * numQuadCol + col) * 8;
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        shapeEnergyMat.insert(left_top_x + i, left_top_x + j) = coeff(i, j);
                    }
                }
            }
        }
        shapeEnergyMat.makeCompressed(); //压缩，压缩有必要吗
        return shapeEnergyMat;
    }

    /*
    计算Es(V)中的Vq标记的超大独热码矩阵Q的顺序都是（左上 右上 左下 右下）
    */
    SpareseMatrixD_Row getVertexIndexQ(MeshInfo* meshInfo)
    {
        int numVertexRow = meshInfo->vertexNumRow;
        int numVertexCol = meshInfo->vertexNumCol;
        int numQuadRow = meshInfo->meshQuadRow;
        int numQuadCol = meshInfo->meshQuadCol;
        // 8指的是每个quad有8个坐标，2指的是每个顶点2个坐标；标记每个quad中的顶点，
        SpareseMatrixD_Row Q(8 * numQuadRow * numQuadCol, 2 * numVertexRow * numVertexCol); //标记Vq！
        for (int row = 0; row < numQuadRow; row++) {
            for (int col = 0; col < numQuadCol; col++) {
                int quadID = 8 * (row * numQuadCol + col); //第几个quad
                int topLeftVertexID = 2 * (row * numVertexCol + col);
                Q.insert(quadID, topLeftVertexID) = 1;         // 左上点的x
                Q.insert(quadID + 1, topLeftVertexID + 1) = 1; // 左上点的y
                Q.insert(quadID + 2, topLeftVertexID + 2) = 1; // 右上点的x
                Q.insert(quadID + 3, topLeftVertexID + 3) = 1; // 右上点的y
                Q.insert(quadID + 4, topLeftVertexID + 2 * numVertexCol) = 1;     // 左下点的x
                Q.insert(quadID + 5, topLeftVertexID + 2 * numVertexCol + 1) = 1; // 左下点的y
                Q.insert(quadID + 6, topLeftVertexID + 2 * numVertexCol + 2) = 1; // 右下点的x
                Q.insert(quadID + 7, topLeftVertexID + 2 * numVertexCol + 3) = 1; // 右下点的y
            }
        }

        Q.makeCompressed();
        return Q;
    }

    ////////////////////////////// Line Preservation Energy //////////////////////////////

    /*
    判断这个线是否有效（沿着原图边缘是个必须处理掉的情况）
    */
    bool checkLine(cv::Mat mask, LineD line)
    {
        int row1 = round(line.row1), row2 = round(line.row2),
            col1 = round(line.col1), col2 = round(line.col2);
        //端点不在内部且和图像没有交集不算（用中点判断）
        if (mask.at<uchar>(row1, col1) != 0 && mask.at<uchar>(row2, col2) != 0 && mask.at<uchar>((row1+row2)/2, (col1+col2)/2) != 0)
            return false;
        //边界上不算
        if ((col1 == mask.cols - 1 && col2 == mask.cols - 1) || (col1 == 0 && col2 == 0))
            return false;
        if ((row1 == mask.rows - 1 && row2 == mask.rows - 1) || (row1 == 0 && row2 == 0))
            return false;
        return true;
        // //单点边界的时候
        // if (row1 == 0 || row1 == mask.rows - 1 || col1 == 0 || col1 == mask.cols - 1) {
        //     try {
        //         if (mask.at<uchar>(row2 + 1, col2) == 255 || mask.at<uchar>(row2 - 1, col2) == 255
        //             || mask.at<uchar>(line.row2, line.col2 + 1) == 255 || mask.at<uchar>(line.row2, line.col2 - 1) == 255)
        //             return false;
        //     } catch (std::exception) {
        //     }
        //     return true;
        // }
        // if (row2 == 0 || row2 == mask.rows - 1 || col2 == 0 || col2 == mask.cols - 1) {
        //     try {
        //         if (mask.at<uchar>(row1 + 1, col1) == 255 || mask.at<uchar>(row1 - 1, col1) == 255
        //             || mask.at<uchar>(row1, col1 + 1) == 255 || mask.at<uchar>(row1, col1 - 1) == 255)
        //             return false;
        //     } catch (std::exception) {
        //     }
        //     return true;
        // }

        // //一般情况
        // try {
        //     if (mask.at<uchar>(row1 + 1, col1) == 255 || mask.at<uchar>(row1 - 1, col1) == 255
        //         || mask.at<uchar>(row1, col1 + 1) == 255 || mask.at<uchar>(row1, col1 - 1) == 255)
        //         return false;
        //     else {
        //         if (mask.at<uchar>(row2 + 1, col2) == 255 || mask.at<uchar>(row2 - 1, col2) == 255
        //             || mask.at<uchar>(line.row2, line.col2 + 1) == 255 || mask.at<uchar>(line.row2, line.col2 - 1) == 255)
        //             return false;
        //         else
        //             return true;
        //     }
        // } catch (std::exception) {
        //     throw "line 的判断异常";
        // }
    }

    /*
    利用lsd的代码进行line的检测
    */
    vector<LineD> lsdDetect(const cv::Mat src, cv::Mat mask)
    {
#ifdef LXH_DEBUG_GLOBAL
        cv::Mat temp;
        src.copyTo(temp);
#endif

        vector<LineD> lines;
        cv::Mat greyImg;
        cv::cvtColor(src, greyImg, cv::COLOR_BGR2GRAY); // 转成灰度图后再进行线条检测

        // 转换成普通数组以满足lsd接口
        double* image = new double[greyImg.rows * greyImg.cols];
        for (int row = 0; row < greyImg.rows; row++)
            for (int col = 0; col < greyImg.cols; col++)
                image[row * greyImg.cols + col] = greyImg.at<uchar>(row, col);

        int lineNum = 0;
        double* out = lsd(&lineNum, image, greyImg.cols, greyImg.rows);
        // x1, y1, x2, y2, width, p, -log_nfa. 我也不知道NFA是个啥......
        for (int i = 0; i < lineNum; i++) {
            LineD line(out[i * 7 + 1], out[i * 7 + 0], out[i * 7 + 3], out[i * 7 + 2]);
            if (checkLine(mask, line)) {
                lines.push_back(line);
#ifdef LXH_DEBUG_GLOBAL
                drawLine(temp, line);
#endif
            }
        }
        
#ifdef LXH_DEBUG_GLOBAL
        /*cv::namedWindow("Border", cv::WINDOW_AUTOSIZE);
        cv::imshow("Border", temp);
        cv::waitKey(0);*/
        cv::imwrite("line_detect.png", temp);
#endif
        delete[] out;
        delete[] image;
        return lines;
    }

    /*
    判断quad的某一边界是否与line有交点
    */
    bool hasIntersection(const LineD& lineSegment, double slope, double intersect, CoordDouble& intersectPoint)
    {
        /*
        row为y轴，col为x轴，斜率k=(y2-y1)/(x2-x1) 截距b=y1-k*x1=y2-k*x2 从而y=k*x+b
        */

        double lineSegmentSlope = INF;
        if (lineSegment.col1 != lineSegment.col2)
            lineSegmentSlope = (lineSegment.row2 - lineSegment.row1) / (lineSegment.col2 - lineSegment.col1); // k
        double lineSegmentIntersect = lineSegment.row1 - lineSegmentSlope * lineSegment.col1; // b

        // calculate intersection
        if (lineSegmentSlope == slope) {
            //如果重叠的话！会被另外的情况下捕捉到与另外相交两边的两交点
            if (lineSegmentIntersect == intersect) {
                return false;
            } else
                return false; //平行无交点
        }

        // y=k1*x+b1 y=k2*x+b2 → 交点：x0=(b2-b1)/(k1-k2),y0=k1*x+b1
        double intersectX = (intersect - lineSegmentIntersect) / (lineSegmentSlope - slope);
        double intersectY = lineSegmentSlope * intersectX + lineSegmentIntersect;

        // 检查交点是否在线段内，用行坐标或者列坐标一个就够
        if ((intersectY <= lineSegment.row1 && intersectY >= lineSegment.row2) || (intersectY <= lineSegment.row2 && intersectY >= lineSegment.row1)) {
            intersectPoint.col = intersectX;
            intersectPoint.row = intersectY;
            return true;
        } else
            return false;
    }

    /*
    计算线段和quad的交点（一个quad用四个坐标表示）
    */
    vector<CoordDouble> getIntersectionsWithQuad(LineD lineSegment, CoordDouble topLeft,
        CoordDouble topRight, CoordDouble bottomLeft, CoordDouble bottomRight)
    {
        /*
        这里可以这样理解：row为y轴，col为x轴，斜率k=(y2-y1)/(x2-x1) 截距b=y1-k*x1=y2-k*x2 从而y=k*x+b
        下述代码 **Slope=k,**Intersect=b
        */
        vector<CoordDouble> intersections;

        // left
        double leftSlope = INF;
        if (topLeft.col != bottomLeft.col)
            leftSlope = (topLeft.row - bottomLeft.row) / (topLeft.col - bottomLeft.col);

        double leftIntersect = topLeft.row - leftSlope * topLeft.col;
        // check
        CoordDouble leftIntersectPoint;
        if (hasIntersection(lineSegment, leftSlope, leftIntersect, leftIntersectPoint))
            if (leftIntersectPoint.row >= topLeft.row && leftIntersectPoint.row <= bottomLeft.row)
                intersections.push_back(leftIntersectPoint);

        // right
        double rightSlope = INF;
        if (topRight.col != bottomRight.col)
            rightSlope = (topRight.row - bottomRight.row) / (topRight.col - bottomRight.col);
        double rightIntersect = topRight.row - rightSlope * topRight.col;
        // check
        CoordDouble rightIntersectPoint;
        if (hasIntersection(lineSegment, rightSlope, rightIntersect, rightIntersectPoint))
            if (rightIntersectPoint.row >= topRight.row && rightIntersectPoint.row <= bottomRight.row)
                intersections.push_back(rightIntersectPoint);

        // top
        double topSlope = INF;
        if (topLeft.col != topRight.col)
            topSlope = (topRight.row - topLeft.row) / (topRight.col - topLeft.col);
        double topIntersect = topLeft.row - topSlope * topLeft.col;
        // check
        CoordDouble topIntersectPoint;
        if (hasIntersection(lineSegment, topSlope, topIntersect, topIntersectPoint))
            if (topIntersectPoint.col >= topLeft.col && topIntersectPoint.col <= topRight.col)
                intersections.push_back(topIntersectPoint);

        // bottom
        double bottomSlope = INF;
        if (bottomLeft.col != bottomRight.col)
            bottomSlope = (bottomRight.row - bottomLeft.row) / (bottomRight.col - bottomLeft.col);
        double bottomIntersect = bottomLeft.row - bottomSlope * bottomLeft.col;
        // check
        CoordDouble bottomIntersectPoint;
        if (hasIntersection(lineSegment, bottomSlope, bottomIntersect, bottomIntersectPoint))
            if (bottomIntersectPoint.col >= bottomLeft.col && bottomIntersectPoint.col <= bottomRight.col)
                intersections.push_back(bottomIntersectPoint);

        return intersections;
    }

    /*
    判断点是否在quad中，这个地方较复杂的原因是quad是个多边形而不是矩形
    */
    bool isInQuad(CoordDouble point, CoordDouble topLeft, CoordDouble topRight,
        CoordDouble bottomLeft, CoordDouble bottomRight)
    {
        // the point must be to the right of the left line, below the top line, above the bottom line,
        // and to the left of the right line
        // must be right of leftt line
        // 如果左上和左下处在一列的话就可以直接判断
        if (topLeft.col == bottomLeft.col) {
            if (point.col < topLeft.col)
                return false;
        } else {
            double leftSlope = (topLeft.col - bottomLeft.col) / (topLeft.row - bottomLeft.row); // k=(y2-y1)/(x2-x1)
            double yOnLineX = leftSlope * (point.row - bottomLeft.row) + bottomLeft.col; // y=k*(x-x1)+y1
            if (point.col < yOnLineX)
                return false;
        }
        // must be left of right line
        if (topRight.col == bottomRight.col) {
            if (point.col > topRight.col)
                return false;
        } else {
            double rightSlope = (topRight.col - bottomRight.col) / (topRight.row - bottomRight.row); // k=(y2-y1)/(x2-x1)
            double yOnLineX = rightSlope * (point.row - bottomRight.row) + bottomRight.col; // y=k*(x-x1)+y1
            if (point.col > yOnLineX)
                return false;
        }
        // must be below top line
        if (topLeft.row == topRight.row) {
            if (point.row < topRight.row)
                return false;
        } else {
            double topSlope = (topRight.col - topLeft.col) / (topRight.row - topLeft.row); // k=(y2-y1)/(x2-x1)
            double xOnLineY = 1 / topSlope * (point.col - topLeft.col) + topLeft.row; // x=1/k*(y-y1)+x1
            if (point.row < xOnLineY)
                return false;
        }
        // must be above bottom line
        if (bottomLeft.row == bottomRight.row) {
            if (point.row > bottomRight.row)
                return false;
        } else {

            double bottomSlope = (bottomRight.col - bottomLeft.col) / (bottomRight.row - bottomLeft.row); // k=(y2-y1)/(x2-x1)
            double xOnLineY = 1 / bottomSlope * (point.col - bottomLeft.col) + bottomLeft.row; // x=1/k*(y-y1)+x1
            if (point.row > xOnLineY)
                return false;
        }
        // if all four constraints are satisfied, the point must be in the quad
        return true;
    }

    /*
    将每个line切割分在quad中
    */
    vector<vector<vector<LineD>>> segmentLineInQuad(vector<LineD> lines, vector<vector<CoordDouble>> mesh, MeshInfo* meshInfo)
    {
        int rowQuadNum = meshInfo->meshQuadRow;
        int colQuadNum = meshInfo->meshQuadCol;
        vector<vector<vector<LineD>>> quadLineSeg;

        for (int row = 0; row < rowQuadNum; row++) {
            vector<vector<LineD>> vecRow;
            for (int col = 0; col < colQuadNum; col++) { // 每个quad分配一个LineD数组（线段数组）
                CoordDouble leftTop = mesh[row][col];
                CoordDouble rightTop = mesh[row][col + 1];
                CoordDouble leftBottom = mesh[row + 1][col];
                CoordDouble rightBottom = mesh[row + 1][col + 1];

                vector<LineD> lineInQuad;
                for (int i = 0; i < lines.size(); i++) {
                    LineD line = lines[i];
                    CoordDouble p1(line.row1, line.col1);
                    CoordDouble p2(line.row2, line.col2);
                    bool p1InQuad = isInQuad(p1, leftTop, rightTop, leftBottom, rightBottom);
                    bool p2InQuad = isInQuad(p2, leftTop, rightTop, leftBottom, rightBottom);
                    if (p1InQuad && p2InQuad) // 整条线都在该quad里面，直接放进去
                        lineInQuad.push_back(line);
                    else if (p1InQuad) { // p1在，p2不在，需要截取交点
                        vector<CoordDouble> intersections = getIntersectionsWithQuad(line, leftTop, rightTop, leftBottom, rightBottom);
                        // assert(intersections.size() == 1);
                        if (intersections.size() != 0) { // 理论上只会有一个交点
                            LineD cutLine(p1, intersections[0]);
                            lineInQuad.push_back(cutLine);
                        }
                    } else if (p2InQuad) { // 同上
                        vector<CoordDouble> intersections = getIntersectionsWithQuad(line, leftTop, rightTop, leftBottom, rightBottom);
                        // assert(intersections.size() == 1);
                        if (intersections.size() != 0) {
                            LineD cutLine(p2, intersections[0]);
                            lineInQuad.push_back(cutLine);
                        }
                    } else { // 两个端点都不在quad里面，就截取两端交点
                        vector<CoordDouble> intersections = getIntersectionsWithQuad(line, leftTop, rightTop, leftBottom, rightBottom);
                        if (intersections.size() == 2) {
                            LineD cutLine(intersections[0], intersections[1]);
                            lineInQuad.push_back(cutLine);
                        }
                    }
                }
                vecRow.push_back(lineInQuad);
            }
            quadLineSeg.push_back(vecRow);
        }

        return quadLineSeg;
    }

    /*
    计算逆双线性插值计算出来的权重的对应矩阵T（2，8）维度, P = T * Vq，该矩阵用于乘以Vq=[x0,y0,...,y3,y4] V的顺序都是（左上 右上 左下 右下）
    */
    MatrixXd inverseBilinearWeights2Matrix(pair<double, double> w)
    {
        double u = w.first, v = w.second;
        MatrixXd mat(2, 8);
        // a-左上 b-右上 d-左下 c-右下
        double a_w = 1 - u - v + u * v;
        double b_w = u - u * v;
        double d_w = v - u * v;
        double c_w = u * v;
        mat << a_w, 0, b_w, 0, d_w, 0, c_w, 0,
            0, a_w, 0, b_w, 0, d_w, 0, c_w;
        return mat;
    }

    /*
    计算两二维坐标的叉积，即：a.x*b.y-a.y-b.x
    */
    double cross2d(CoordDouble a, CoordDouble b) { return a.col * b.row - a.row * b.col; }

    /*
    计算逆双线性插值的u,v, 参考https://iquilezles.org/articles/ibilinear/
    */
    pair<double, double> getInverseBilinearWeights(CoordDouble point, CoordInt upperLeftIndices, const vector<vector<CoordDouble>>& mesh)
    {
        //
        // row为y轴，col为x轴
        //通过mesh和左上顶点的索引计算四个顶点坐标(顺序！)
        CoordDouble a = mesh[upperLeftIndices.row][upperLeftIndices.col]; // topLeft
        CoordDouble b = mesh[upperLeftIndices.row][upperLeftIndices.col + 1]; // topRight
        CoordDouble d = mesh[upperLeftIndices.row + 1][upperLeftIndices.col]; // bottomLeft
        CoordDouble c = mesh[upperLeftIndices.row + 1][upperLeftIndices.col + 1]; // bottomRight

        // E、F、G、H
        CoordDouble e = b - a;
        CoordDouble f = d - a;
        CoordDouble g = a - b + c - d;
        CoordDouble h = point - a;

        // k2,k1,k0
        double k2 = cross2d(g, f);
        double k1 = cross2d(e, f) + cross2d(h, g);
        double k0 = cross2d(h, e);

        double u, v;

        if ((int)k2 == 0) {
            v = -k0 / k1;
            u = (h.col - f.col * v) / (e.col + g.col * v);
        } else {
            double w = k1 * k1 - 4.0 * k0 * k2;
            assert(w >= 0.0); //如果point在quad内部，必然不会小于0
            w = sqrt(w);

            double v1 = (-k1 - w) / (2.0 * k2);
            double u1 = (h.col - f.col * v1) / (e.col + g.col * v1);

            double v2 = (-k1 + w) / (2.0 * k2);
            double u2 = (h.col - f.col * v2) / (e.col + g.col * v2);

            u = u1;
            v = v1;

            if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0) {
                u = u2;
                v = v2;
            }
        }
        return pair<double, double>(u, v);
    }

    /*
    初始化计算line的vector数组（行、列、线）及线段角度，以及逆双线性插值权重。
    */
    vector<vector<vector<LineD>>> initLineSegment(const cv::Mat src, const cv::Mat mask, MeshInfo* meshInfo,
        vector<vector<CoordDouble>> mesh, vector<pair<int, double>>& id_theta, vector<double>& rotateThetas,
        vector<pair<MatrixXd, MatrixXd>>& inverseBilinearWeights, int& lineNum)
    {
        double thetaPerbin = PI / 49; //即分成了50份从-PI/2到PI/2
        lineNum = 0; // 注意这个lineNum是切割后的lineNum
        // Step1: Detect line except border
        vector<LineD> lines = lsdDetect(src, mask);

        // Step2: Segment line in each quad
        vector<vector<vector<LineD>>> lineSeg = segmentLineInQuad(lines, mesh, meshInfo);

        int numQuadRow = meshInfo->meshQuadRow;
        int numQuadCol = meshInfo->meshQuadCol;
        for (int row = 0; row < numQuadRow; row++) {
            for (int col = 0; col < numQuadCol; col++) {
                CoordInt topLeft(row, col); // quad num的
                for (int k = 0; k < lineSeg[row][col].size(); k++) {
                    LineD line = lineSeg[row][col][k]; // 这里算方向向量是start point - end point
                    double theta = atan((line.row1 - line.row2) / (line.col1 - line.col2)); //-PI/2~PI/2 计算线段角度
                    int binID = (int)round((theta + PI / 2) / thetaPerbin); //四舍五入计算属于哪一个Bin..0~49
                    assert(binID < 50);
                    id_theta.push_back(make_pair(binID, theta));
                    rotateThetas.push_back(0); // 初始的时候偏移角度为0

                    CoordDouble startPoint(line.row1, line.col1);
                    CoordDouble endPoint(line.row2, line.col2);
                    pair<double, double> startWeight = getInverseBilinearWeights(startPoint, topLeft, mesh); // s t2n t t1n
                    MatrixXd startWeightMat = inverseBilinearWeights2Matrix(startWeight);
                    pair<double, double> endWeight = getInverseBilinearWeights(endPoint, topLeft, mesh);
                    MatrixXd endWeightMat = inverseBilinearWeights2Matrix(endWeight);
                    inverseBilinearWeights.push_back(make_pair(startWeightMat, endWeightMat));

                    ++lineNum;
                }
            }
        }
        return lineSeg;
    }


    /*
    以对角的形式存储CT 之后直接乘V就可以得到结果
    */
    SpareseMatrixD_Row blockDiag(const SpareseMatrixD_Row& origin, const MatrixXd& addin, int quadID, MeshInfo* meshInfo)
    {
        int cols_total = 8 * meshInfo->meshQuadRow * meshInfo->meshQuadCol;
        SpareseMatrixD_Row res(origin.rows() + addin.rows(), cols_total);
        res.topRows(origin.rows()) = origin;

        int lefttop_row = origin.rows();
        int lefttop_col = 8 * quadID;
        for (int row = 0; row < addin.rows(); row++) {
            for (int col = 0; col < addin.cols(); col++) {
                res.insert(lefttop_row + row, lefttop_col + col) = addin(row, col);
            }
        }
        res.makeCompressed();
        return res;
    }

    //获取计算线的能量的矩阵 对角上存储（2，8）—— C*T，稀疏存储
    SpareseMatrixD_Row getLineEnergyMat(const cv::Mat src, cv::Mat mask, vector<double> rotateThetas, 
    vector<vector<vector<LineD>>> lineSeg, const vector<pair<MatrixXd, MatrixXd>>& inverseBilinearWeights,
        MeshInfo* meshInfo)
    {

        int rowQuadNum = meshInfo->meshQuadRow;
        int colQuadNum = meshInfo->meshQuadCol;
        int lineID = 0; // 计数的

        SpareseMatrixD_Row lineEnergyMat;
        for (int row = 0; row < rowQuadNum; row++) {
            for (int col = 0; col < colQuadNum; col++) {
                vector<LineD> lineSegInQuad = lineSeg[row][col];
                int quadID = row * colQuadNum + col;
                if (lineSegInQuad.size() == 0) { // quad里面没有线段
                    continue;
                } else {
                    CoordInt topLeft(row, col); // quad num的
                    MatrixXd CT_rowStack(0, 8);
                    for (int k = 0; k < lineSegInQuad.size(); k++) {
                        LineD line = lineSegInQuad[k];
                        MatrixXd startWeightMat = inverseBilinearWeights[lineID].first;
                        MatrixXd endWeightMat = inverseBilinearWeights[lineID].second;
                        double theta = rotateThetas[lineID];

                        Matrix2d R; //旋转矩阵R
                        R << cos(theta), -sin(theta),
                            sin(theta), cos(theta);
                        MatrixXd ehat(2, 1);
                        ehat << line.col1 - line.col2, line.row1 - line.row2; //[col,row]
                        Matrix2d I = Matrix2d::Identity();
                        MatrixXd C = R * ehat * ((ehat.transpose() * ehat).inverse()) * (ehat.transpose()) * (R.transpose()) - I;
                        MatrixXd CT = C * (startWeightMat - endWeightMat); // CT之后乘以Vq就是了
                        CT_rowStack = rowStack(CT_rowStack, CT); // CT (2,8)

                        ++lineID;
                    }
                    lineEnergyMat = blockDiag(lineEnergyMat, CT_rowStack, quadID, meshInfo);
                }
            }
        }
        return lineEnergyMat;
    }



    ////////////////////////////// Boundary Constraint Energy //////////////////////////////

    /*
    计算EB(Q)中V的标记的矩阵（pair） V的顺序都是（左上 右上 左下 右下）
    */
    pair<SpareseMatrixD_Row, VectorXd> getBoundaryMat(MeshInfo* meshInfo)
    {

        int h = meshInfo->rows - 1;
        int w = meshInfo->cols - 1;
        int numVertexRow = meshInfo->vertexNumRow;
        int numVertexCol = meshInfo->vertexNumCol;
        int vertexnum = numVertexRow * numVertexCol;

        // 顺序：[x0,y0,x1,y1...] row为y轴，col为x轴
        VectorXd M = VectorXd::Zero(vertexnum * 2); // M
        VectorXd b = VectorXd::Zero(vertexnum * 2); // b
        for (int i = 0; i < vertexnum * 2; i += numVertexCol * 2) { // 
            M(i) = 1;
            b(i) = 0;
        } // x
        for (int i = numVertexCol * 2 - 2; i < vertexnum * 2; i += numVertexCol * 2) { // right
            M(i) = 1;
            b(i) = w;
        } // y

        for (int i = 1; i < 2 * numVertexCol; i += 2) { // top
            M(i) = 1;
            b(i) = 0;
        }

        for (int i = 2 * vertexnum - 2 * numVertexCol + 1; i < vertexnum * 2; i += 2) { // bottom
            M(i) = 1;
            b(i) = h;
        }

        // 把M换成稀疏矩阵对角来存储和后续计算
        SpareseMatrixD_Row diag(M.size(), M.size());
        for (int i = 0; i < M.size(); i++) {
            diag.insert(i, i) = M(i);
        }
        diag.makeCompressed();
        return make_pair(diag, b);
    };

}
}