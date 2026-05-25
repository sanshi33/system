#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace stitch {

// 角度（弧度）[-pi, pi]，以 center 为原点
double computeAngle(const cv::Point2d& pt, const cv::Point2d& center);

// 局部曲率估计
double computeLocalCurvature(const std::vector<cv::Point2d>& pts, int idx, int window = 5);

// 切线角变化量
double computeTangentChange(const std::vector<cv::Point2d>& pts, int idx, int window = 5);

// 轮廓排序（用于插值匹配）
void sortContourByX(std::vector<cv::Point2d>& pts);
void sortContourByY(std::vector<cv::Point2d>& pts);

// 线性插值：已按 X/Y 排序的点集上，插值另一维度
bool getInterpolatedY(const std::vector<cv::Point2d>& x_sorted, double x, double& outY);
bool getInterpolatedX(const std::vector<cv::Point2d>& y_sorted, double y, double& outX);

// Catmull-Rom 三次样条插值，用于替代线性插值建立轮廓对应关系
// p0, p1, p2, p3 为连续四个点，t∈[0,1] 为 p1-p2 段上的参数
// 若 out_tangent 非空，同时输出该点的切向量
cv::Point2d catmullRomInterpolate(const cv::Point2d& p0, const cv::Point2d& p1,
                                   const cv::Point2d& p2, const cv::Point2d& p3,
                                   double t, cv::Point2d* out_tangent = nullptr);

// 以 center 为中心旋转点集
void rotatePoints(const std::vector<cv::Point2d>& in,
                  std::vector<cv::Point2d>& out,
                  double angle_deg,
                  cv::Point2d center);

} // namespace stitch
