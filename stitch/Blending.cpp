#include "Blending.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace stitch {
using namespace cv;
using namespace std;

void initCanvasAndPlaceFirst(const Mat &first_img, Mat &canvas, Mat &M_global)
{
    int single_w = first_img.cols;
    int single_h = first_img.rows;
    canvas = Mat(single_h * 6, single_w * 6, CV_8UC3, Scalar(255, 255, 255));

    M_global = Mat::eye(3, 3, CV_64F);
    M_global.at<double>(0, 2) = canvas.cols / 2.0 - single_w / 2.0;
    M_global.at<double>(1, 2) = canvas.rows / 2.0 - single_h / 2.0;

    warpAffine(first_img, canvas, M_global.rowRange(0, 2), canvas.size(),
               INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
}

void applyTransformAndBlend(const Mat &img_next, Mat &canvas, Mat &M_global,
                            const Point2d &center, const TransformResult &res)
{
    Mat M_rel = Mat::eye(3, 3, CV_64F);
    if (res.hasCustomRelativeMatrix &&
        !res.customRelativeMatrix.empty() &&
        res.customRelativeMatrix.rows == 3 &&
        res.customRelativeMatrix.cols == 3) {
        M_rel = res.customRelativeMatrix.clone();
    } else {
        Mat M_rot_2x3 = getRotationMatrix2D(center, res.da, 1.0);
        M_rot_2x3.copyTo(M_rel(Rect(0, 0, 3, 2)));
        M_rel.at<double>(0, 2) += res.dx;
        M_rel.at<double>(1, 2) += res.dy;
    }

    M_global = M_global * M_rel;
    Mat M_draw = M_global.rowRange(0, 2);

    Mat warped_next;
    warpAffine(img_next, warped_next, M_draw, canvas.size(),
               INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
    cv::min(canvas, warped_next, canvas);
}

Rect cropCanvasAuto(Mat &canvas)
{
    Mat gray;
    cvtColor(canvas, gray, COLOR_BGR2GRAY);
    Mat non_white_mask;
    threshold(gray, non_white_mask, 250, 255, THRESH_BINARY_INV);

    vector<Point> non_zero_pts;
    findNonZero(non_white_mask, non_zero_pts);

    if (!non_zero_pts.empty())
    {
        Rect bbox = boundingRect(non_zero_pts);
        int padding = 20;
        bbox.x = std::max(0, bbox.x - padding);
        bbox.y = std::max(0, bbox.y - padding);
        bbox.width = std::min(canvas.cols - bbox.x, bbox.width + 2 * padding);
        bbox.height = std::min(canvas.rows - bbox.y, bbox.height + 2 * padding);
        canvas = canvas(bbox);
        return bbox;
    }

    return Rect();
}

} // namespace stitch
