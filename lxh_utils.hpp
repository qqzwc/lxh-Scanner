
#pragma once
#include "GL/glut.h"
#include "lxh_macro.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "lxh_timer.hpp"
using namespace std;


namespace lxh {
typedef cv::Vec3b colorPixel;
/*
• Matrix 矩阵
• Vector 向量
• i -> int
• f -> float
• d -> double
• cf -> complex float
• cd -> complex double
• X for dynamic size
• 2 固定大小2×2
• 3 固定大小3×3
• 4 固定大小4×4
*/
typedef Eigen::SparseMatrix<double> SparseMatrixD; // 默认列优先
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpareseMatrixD_Row; //行优先
typedef Eigen::Vector2d Vector2d;
typedef Eigen::Vector2i Vector2i;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Matrix2d Matrix2d;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::SimplicialLDLT<SparseMatrixD> CSolve;

/*
Grid-Mesh的基本信息
*/
class MeshInfo {
public:
    int rows; // 对应图像行数
    int cols; // 对应图像列数
    int vertexNumRow; // 网格行顶点数
    int vertexNumCol; // 网格列顶点数
    int meshQuadRow; // quad的行数
    int meshQuadCol; // quad的列数
    double quadHeight; // 单个quad的高度
    double quadWidth; // 单个quad的宽度
    MeshInfo(int rows, int cols, int vertexNumRow, int vertexNumCol)
    {
        this->rows = rows;
        this->cols = cols;
        this->vertexNumRow = vertexNumRow;
        this->vertexNumCol = vertexNumCol;
        this->meshQuadRow = vertexNumRow - 1;
        this->meshQuadCol = vertexNumCol - 1;
        this->quadHeight = double(rows - 1) / meshQuadRow; //高度算的是quad的高度加一个边框宽度
        this->quadWidth = double(cols - 1) / meshQuadCol; // 宽度算的是quad的宽度加一个边框宽度
    }
    void reConfig(int rows, int cols)
    {
        this->rows = rows;
        this->cols = cols;
        this->quadHeight = double(rows - 1) / meshQuadRow; //高度算的是quad的高度加一个边框宽度
        this->quadWidth = double(cols - 1) / meshQuadCol; // 宽度算的是quad的宽度加一个边框宽度
    }
};

//整型坐标
struct CoordInt {
    int row;
    int col;

    bool operator==(const CoordInt& rhs) const
    {
        return (row == rhs.row && col == rhs.col);
    }
    bool operator<(const CoordInt& rhs) const
    {
        // this operator is used to determine equality, so it must use both x and y
        if (row < rhs.row) {
            return true;
        }
        if (row > rhs.row) {
            return false;
        }
        return col < rhs.col;
    }
    CoordInt()
    {
        row = 0;
        col = 0;
    };
    CoordInt(int setRow, int setCol)
    {
        row = setRow;
        col = setCol;
    };
};

//双精度浮点坐标
struct CoordDouble {
    double row;
    double col;

    bool operator==(const CoordDouble& rhs) const
    {
        return (row == rhs.row && col == rhs.col);
    }
    bool operator<(const CoordDouble& rhs) const
    {
        // this operator is used to determine equality, so it must use both x and y
        if (row < rhs.row) {
            return true;
        }
        if (row > rhs.row) {
            return false;
        }
        return col < rhs.col;
    }

    CoordDouble operator+(const CoordDouble& b)
    {
        CoordDouble temp;
        temp.row = row + b.row;
        temp.col = col + b.col;
        return temp;
    }
    CoordDouble operator-(const CoordDouble& b)
    {
        CoordDouble temp;
        temp.row = row - b.row;
        temp.col = col - b.col;
        return temp;
    }

    friend ostream& operator<<(ostream& stream, const CoordDouble& p)
    {
        stream << "(" << p.col << "," << p.row << ")";
        return stream;
    }
    CoordDouble()
    {
        row = 0;
        col = 0;
    };
    CoordDouble(double setRow, double setCol)
    {
        row = setRow;
        col = setCol;
    };
};

// 线段
struct LineD {
    double row1, col1;
    double row2, col2;
    LineD(double row1, double col1, double row2, double col2)
    {
        this->row1 = row1;
        this->row2 = row2;
        this->col1 = col1;
        this->col2 = col2;
    }
    LineD()
    {
        row1 = 0;
        col1 = 0;
        row2 = 0;
        col2 = 0;
    }
    LineD(CoordDouble p1, CoordDouble p2)
    {
        row1 = p1.row;
        row2 = p2.row;
        col1 = p1.col;
        col2 = p2.col;
    }
};

//按行堆叠两个矩阵
SpareseMatrixD_Row rowStack(SparseMatrixD origin, SpareseMatrixD_Row diag)
{
    SpareseMatrixD_Row res(origin.rows() + diag.rows(), origin.cols());
    res.topRows(origin.rows()) = origin;
    res.bottomRows(diag.rows()) = diag;
    return res;
}
SpareseMatrixD_Row rowStack(SpareseMatrixD_Row origin, SpareseMatrixD_Row diag)
{
    SpareseMatrixD_Row res(origin.rows() + diag.rows(), origin.cols());
    res.topRows(origin.rows()) = origin;
    res.bottomRows(diag.rows()) = diag;
    return res;
}
MatrixXd rowStack(MatrixXd mat1, MatrixXd mat2)
{
    MatrixXd res(mat1.rows() + mat2.rows(), mat1.cols());
    res.topRows(mat1.rows()) = mat1;
    res.bottomRows(mat2.rows()) = mat2;
    return res;
}

//把解转为mesh
vector<vector<CoordDouble>> vector2Mesh(VectorXd x, MeshInfo* meshInfo)
{
    int numMeshRow = meshInfo->vertexNumRow;
    int numMeshCol = meshInfo->vertexNumCol;
    vector<vector<CoordDouble>> mesh;
    // row为y轴，col为x轴 Vq=[y0,x0,y1,x1...]
    for (int row = 0; row < numMeshRow; row++) {
        vector<CoordDouble> meshRow;
        for (int col = 0; col < numMeshCol; col++) {
            int xid = (row * numMeshCol + col) * 2;
            CoordDouble coord;
            coord.row = x(xid + 1);
            coord.col = x(xid);
            meshRow.push_back(coord);
        }
        mesh.push_back(meshRow);
    }
    return mesh;
}

/*
根据缩放因子调整Mesh大小
*/
void reMesh(vector<vector<CoordDouble>>& mesh, double fx, double fy, MeshInfo* meshInfo)
{
    int numMeshRow = meshInfo->vertexNumRow;
    int numMeshCol = meshInfo->vertexNumCol;
    for (int row = 0; row < numMeshRow; row++) {
        for (int col = 0; col < numMeshCol; col++) {
            CoordDouble& coord = mesh[row][col];
            coord.row = coord.row * fy;
            coord.col = coord.col * fx;
        }
    }
};

/*
计算论文中最后提到的缩放因子
*/
void computeScaling(double& sx_avg, double& sy_avg, const vector<vector<CoordDouble>> mesh,
    const vector<vector<CoordDouble>> outputmesh, MeshInfo* meshInfo)
{
    // row为y轴，col为x轴
    int numQuadRow = meshInfo->meshQuadRow;
    int numQuadCol = meshInfo->meshQuadRow;
    double sx = 0, sy = 0;
    for (int row = 0; row < numQuadRow; row++) {
        for (int col = 0; col < numQuadCol; col++) {
            CoordDouble p0 = mesh[row][col]; //左上
            CoordDouble p1 = mesh[row][col + 1]; //右上
            CoordDouble p2 = mesh[row + 1][col]; //左下
            CoordDouble p3 = mesh[row + 1][col + 1]; //右下

            CoordDouble p0_out = outputmesh[row][col]; //左上
            CoordDouble p1_out = outputmesh[row][col + 1]; //右上
            CoordDouble p2_out = outputmesh[row + 1][col]; //左下
            CoordDouble p3_out = outputmesh[row + 1][col + 1]; //右下

            cv::Mat A = (cv::Mat_<double>(1, 4) << p0.row, p1.row, p2.row, p3.row);
            cv::Mat B = (cv::Mat_<double>(1, 4) << p0_out.row, p1_out.row, p2_out.row, p3_out.row);
            double maxVal, minVal;
            double maxValOut, minValOut;
            cv::minMaxIdx(A, &minVal, &maxVal);
            cv::minMaxIdx(B, &minValOut, &maxValOut);
            sy += (maxValOut - minValOut) / (maxVal - minVal);

            cv::Mat C = (cv::Mat_<double>(1, 4) << p0.col, p1.col, p2.col, p3.col);
            cv::Mat D = (cv::Mat_<double>(1, 4) << p0_out.col, p1_out.col, p2_out.col, p3_out.col);
            cv::minMaxIdx(C, &maxVal, &minVal);
            cv::minMaxIdx(D, &maxValOut, &minValOut);
            sx += (maxValOut - minValOut) / (maxVal - minVal);
        }
    }
    sx_avg = sx / (numQuadRow * numQuadCol);
    sy_avg = sy / (numQuadRow * numQuadCol);
}




#if defined(LXH_OUT) || defined(LXH_DEBUG_RECT) ||  defined(LXH_DEBUG_LOCAL) ||  defined(LXH_DEBUG_GLOBAL)
/*
绘图函数
*/

//画线1 直接int
void drawLine(cv::Mat img, CoordDouble coordstart, CoordDouble coordend)
{
    cv::Point start((int)coordstart.col, (int)coordstart.row);
    cv::Point end((int)coordend.col, (int)coordend.row);
    int thickness = 1;
    int lineType = cv::LINE_AA; //貌似是反锯齿
    cv::line(img, start, end, cv::Scalar(0, 255, 0), thickness, lineType);
}
//画线2 直接int
void drawLine(cv::Mat img, LineD line)
{
    cv::Point start((int)line.col1, (int)line.row1);
    cv::Point end((int)line.col2, (int)line.row2);
    int thickness = 1;
    int lineType = cv::LINE_AA; //貌似是反锯齿
    cv::line(img, start, end, cv::Scalar(0, 255, 0), thickness, lineType);
}
// 在src上画mesh
void drawMesh(cv::Mat _src, const vector<vector<CoordDouble>> mesh, MeshInfo* meshInfo, string filename)
{
    cv::Mat src;
    _src.copyTo(src);
    int vertexNumRow = meshInfo->vertexNumRow;
    int vertexNumCol = meshInfo->vertexNumCol;

    for (int row = 0; row < vertexNumRow; row++) {
        for (int col = 0; col < vertexNumCol; col++) {
            CoordDouble now = mesh[row][col];
            if (row == vertexNumRow - 1 && col < vertexNumCol - 1) {
                CoordDouble right = mesh[row][col + 1];
                drawLine(src, now, right);
            } else if (row < vertexNumRow - 1 && col == vertexNumCol - 1) {
                CoordDouble down = mesh[row + 1][col];
                drawLine(src, now, down);
            } else if (row == vertexNumRow - 1 && col == vertexNumCol - 1)
                ;
            else { // row < vertexNumRow - 1 && col < vertexNumCol - 1
                CoordDouble right = mesh[row][col + 1];
                drawLine(src, now, right);
                CoordDouble down = mesh[row + 1][col];
                drawLine(src, now, down);
            }
        }
    }
    // cv::imshow(filename, src);
    // cv::waitKey(0);
    cv::imwrite(filename, src);
}

// void saveMat(const char filename[], const cv::Mat& mat)
// {
//     fstream matWriter(filename, ios::out);
//     if (!matWriter.fail()) {
//         cout << "start writing " << filename << endl;
//         for (int i = 0; i < mat.rows; i++) {
//             for (int j = 0; j < mat.cols; j++) {
//                 matWriter << mat.at<float>(i, j) << "\t";
//             }
//             matWriter << std::endl;
//         }
//         cout << "finish writing " << filename << endl;
//     } else
//         cout << "can not open" << endl;
//     matWriter.close();
// }
#endif


}