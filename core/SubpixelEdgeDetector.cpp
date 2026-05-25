#include "SubpixelEdgeDetector.h"

#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace {

using SubpixelEdge::SubpixelPoint;

constexpr double kPi = 3.14159265358979323846;

/**
 * @brief ERF 拟合内部基类。
 *
 * 这部分不暴露到头文件中，只在实现文件里使用。
 * 这样可以把：
 * - 梯度计算
 * - 双线性插值
 * - Canny 点提取
 * - ERF 具体拟合过程
 *
 * 都收口在 `core` 层内部，避免污染外部接口。
 */
class SubpixelBase {
public:
    virtual ~SubpixelBase() = default;
    virtual std::vector<SubpixelPoint> detect(const cv::Mat& image) = 0;

    void setCannyThresholds(double low, double high)
    {
        cannyLow_ = low;
        cannyHigh_ = high;
    }

    void setWindowSize(int size)
    {
        windowSize_ = std::max(3, size | 1);
    }

    void setPreSmoothSigma(double sigma)
    {
        preSmoothSigma_ = std::max(0.0, sigma);
    }

    void setUseScharr(bool useScharr)
    {
        useScharr_ = useScharr;
    }

protected:
    static double bilinear64(const cv::Mat& image, double x, double y)
    {
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        if (x0 < 0 || y0 < 0 || x1 >= image.cols || y1 >= image.rows) {
            return 0.0;
        }

        const double fx = x - x0;
        const double fy = y - y0;

        const double v00 = image.at<double>(y0, x0);
        const double v01 = image.at<double>(y0, x1);
        const double v10 = image.at<double>(y1, x0);
        const double v11 = image.at<double>(y1, x1);

        return v00 * (1.0 - fx) * (1.0 - fy) +
               v01 * fx * (1.0 - fy) +
               v10 * (1.0 - fx) * fy +
               v11 * fx * fy;
    }

    static double bilinearU8(const cv::Mat& image, double x, double y)
    {
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        if (x0 < 0 || y0 < 0 || x1 >= image.cols || y1 >= image.rows) {
            return 0.0;
        }

        const double fx = x - x0;
        const double fy = y - y0;

        const double v00 = static_cast<double>(image.at<uchar>(y0, x0));
        const double v01 = static_cast<double>(image.at<uchar>(y0, x1));
        const double v10 = static_cast<double>(image.at<uchar>(y1, x0));
        const double v11 = static_cast<double>(image.at<uchar>(y1, x1));

        return v00 * (1.0 - fx) * (1.0 - fy) +
               v01 * fx * (1.0 - fy) +
               v10 * (1.0 - fx) * fy +
               v11 * fx * fy;
    }

    static void safeNormalize(double gx, double gy, double& nx, double& ny)
    {
        const double magnitude = std::sqrt(gx * gx + gy * gy);
        if (magnitude < 1e-12) {
            nx = 0.0;
            ny = 0.0;
            return;
        }

        nx = gx / magnitude;
        ny = gy / magnitude;
    }

    void computeGradients(const cv::Mat& image,
                          cv::Mat& gradX,
                          cv::Mat& gradY,
                          cv::Mat& gradMag) const
    {
        CV_Assert(image.type() == CV_8UC1);

        cv::Mat image64;
        image.convertTo(image64, CV_64F);

        if (preSmoothSigma_ > 0.0) {
            cv::GaussianBlur(image64, image64, cv::Size(0, 0), preSmoothSigma_, preSmoothSigma_,
                             cv::BORDER_REPLICATE);
        }

        if (useScharr_) {
            cv::Scharr(image64, gradX, CV_64F, 1, 0);
            cv::Scharr(image64, gradY, CV_64F, 0, 1);
        } else {
            cv::Sobel(image64, gradX, CV_64F, 1, 0, 3);
            cv::Sobel(image64, gradY, CV_64F, 0, 1, 3);
        }

        cv::magnitude(gradX, gradY, gradMag);
    }

    std::vector<cv::Point> getEdgePoints(const cv::Mat& image) const
    {
        CV_Assert(image.type() == CV_8UC1);

        cv::Mat smoothed = image;
        if (preSmoothSigma_ > 0.0) {
            cv::GaussianBlur(image, smoothed, cv::Size(0, 0), preSmoothSigma_, preSmoothSigma_,
                             cv::BORDER_REPLICATE);
        }

        cv::Mat edgeMask;
        cv::Canny(smoothed, edgeMask, cannyLow_, cannyHigh_, 3, true);

        std::vector<cv::Point> points;
        points.reserve(static_cast<std::size_t>(edgeMask.total() / 20));

        for (int y = 0; y < edgeMask.rows; ++y) {
            const uchar* row = edgeMask.ptr<uchar>(y);
            for (int x = 0; x < edgeMask.cols; ++x) {
                if (row[x]) {
                    points.emplace_back(x, y);
                }
            }
        }

        return points;
    }

protected:
    double cannyLow_ = 50.0;
    double cannyHigh_ = 150.0;
    int windowSize_ = 5;
    double preSmoothSigma_ = 1.0;
    bool useScharr_ = true;
};

/**
 * @brief 基于 1D ERF + 线性背景模型的亚像素边缘拟合。
 *
 * 算法流程保持与旧实现一致：
 * 1. 在像素级 Canny 边缘点上取样
 * 2. 沿梯度法向构造 1D 灰度剖面
 * 3. 先用中值穿越和 25%/75% 宽度给出初值
 * 4. 使用带 Huber 权重的 LM 迭代拟合 ERF
 */
class ErfFitting final : public SubpixelBase {
public:
    std::vector<SubpixelPoint> detect(const cv::Mat& image) override
    {
        std::vector<SubpixelPoint> results;

        cv::Mat gradX;
        cv::Mat gradY;
        cv::Mat gradMag;
        computeGradients(image, gradX, gradY, gradMag);

        const auto edgePoints = getEdgePoints(image);
        const int halfWindow = std::max(3, windowSize_ / 2);

        results.reserve(edgePoints.size());

        for (const auto& point : edgePoints) {
            if (point.x < halfWindow || point.x >= image.cols - halfWindow ||
                point.y < halfWindow || point.y >= image.rows - halfWindow) {
                continue;
            }

            const double gx0 = gradX.at<double>(point.y, point.x);
            const double gy0 = gradY.at<double>(point.y, point.x);
            const double gradientMagnitude = gradMag.at<double>(point.y, point.x);

            if (!std::isfinite(gradientMagnitude) || gradientMagnitude < 1e-6) {
                continue;
            }

            double nx = 0.0;
            double ny = 0.0;
            safeNormalize(gx0, gy0, nx, ny);
            if (nx == 0.0 && ny == 0.0) {
                continue;
            }

            std::vector<double> t;
            std::vector<double> y;
            t.reserve(2 * halfWindow + 1);
            y.reserve(2 * halfWindow + 1);

            for (int offset = -halfWindow; offset <= halfWindow; ++offset) {
                const double px = point.x + offset * nx;
                const double py = point.y + offset * ny;
                t.push_back(static_cast<double>(offset));
                y.push_back(bilinearU8(image, px, py));
            }

            double x0 = 0.0;
            double sigma = 1.0;
            double confidence = 0.0;

            if (fitErfLm(t, y, x0, sigma, confidence)) {
                const double subX = point.x + x0 * nx;
                const double subY = point.y + x0 * ny;
                results.emplace_back(subX, subY, confidence, gradientMagnitude);
            }
        }

        return results;
    }

private:
    static double erfValue(double u)
    {
        return std::erf(u);
    }

    static bool estimateMidCrossing(const std::vector<double>& t,
                                    const std::vector<double>& y,
                                    double& x0)
    {
        const int count = static_cast<int>(t.size());
        if (count < 5) {
            return false;
        }

        const auto [yMinIt, yMaxIt] = std::minmax_element(y.begin(), y.end());
        const double yMin = *yMinIt;
        const double yMax = *yMaxIt;
        if (yMax - yMin < 5.0) {
            return false;
        }

        const double mid = 0.5 * (yMin + yMax);

        int bestIndex = 0;
        double bestDiff = std::abs(y[0] - mid);
        for (int i = 1; i < count; ++i) {
            const double diff = std::abs(y[i] - mid);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIndex = i;
            }
        }

        const auto interpolateCrossing = [&](int i0, int i1) -> bool {
            const double y0 = y[i0];
            const double y1 = y[i1];
            if ((y0 - mid) * (y1 - mid) > 0.0) {
                return false;
            }

            const double denominator = y1 - y0;
            if (std::abs(denominator) < 1e-9) {
                return false;
            }

            x0 = t[i0] + (mid - y0) / denominator * (t[i1] - t[i0]);
            return std::isfinite(x0);
        };

        if (bestIndex > 0 && interpolateCrossing(bestIndex - 1, bestIndex)) {
            return true;
        }
        if (bestIndex + 1 < count && interpolateCrossing(bestIndex, bestIndex + 1)) {
            return true;
        }

        x0 = t[bestIndex];
        return std::isfinite(x0);
    }

    static bool estimateSigma25_75(const std::vector<double>& t,
                                   const std::vector<double>& y,
                                   double& sigma)
    {
        const int count = static_cast<int>(t.size());
        if (count < 5) {
            return false;
        }

        const auto [yMinIt, yMaxIt] = std::minmax_element(y.begin(), y.end());
        const double yMin = *yMinIt;
        const double yMax = *yMaxIt;
        const double amplitude = yMax - yMin;
        if (amplitude < 5.0) {
            return false;
        }

        const double y25 = yMin + 0.25 * amplitude;
        const double y75 = yMin + 0.75 * amplitude;

        const auto findCross = [&](double target, double& outT) -> bool {
            for (int i = 0; i < count - 1; ++i) {
                const double y0 = y[i];
                const double y1 = y[i + 1];
                if ((y0 - target) * (y1 - target) <= 0.0 && std::abs(y1 - y0) > 1e-9) {
                    outT = t[i] + (target - y0) / (y1 - y0) * (t[i + 1] - t[i]);
                    return std::isfinite(outT);
                }
            }
            return false;
        };

        double t25 = 0.0;
        double t75 = 0.0;
        if (!findCross(y25, t25) || !findCross(y75, t75)) {
            return false;
        }

        const double delta = std::abs(t75 - t25);
        if (delta < 1e-6) {
            return false;
        }

        sigma = delta / 1.34898;
        return std::isfinite(sigma) && sigma > 0.2 && sigma < 10.0;
    }

    static bool fitErfLm(const std::vector<double>& t,
                         const std::vector<double>& y,
                         double& x0Out,
                         double& sigmaOut,
                         double& confidenceOut)
    {
        const int count = static_cast<int>(t.size());
        if (count < 7) {
            return false;
        }

        const auto [yMinIt, yMaxIt] = std::minmax_element(y.begin(), y.end());
        const double yMin = *yMinIt;
        const double yMax = *yMaxIt;
        const double amplitudeSpan = yMax - yMin;
        if (amplitudeSpan < 5.0) {
            return false;
        }

        double x0 = 0.0;
        if (!estimateMidCrossing(t, y, x0)) {
            return false;
        }

        double sigma = 1.0;
        if (!estimateSigma25_75(t, y, sigma)) {
            sigma = 1.2;
        }

        const double c1 = (y.back() - y.front()) / (t.back() - t.front());
        const double a0 = (y.back() > y.front()) ? amplitudeSpan : -amplitudeSpan;
        const double u0 = (0.0 - x0) / (std::sqrt(2.0) * sigma);
        const double f0 = 0.5 * (1.0 + erfValue(u0));
        const double c0 = y[count / 2] - c1 * 0.0 - a0 * f0;

        Eigen::Matrix<double, 5, 1> p;
        p << c0, c1, a0, x0, std::log(sigma);

        double lambda = 1e-3;
        double lastCost = std::numeric_limits<double>::infinity();

        const auto evaluateCostAndJacobian =
            [&](const Eigen::Matrix<double, 5, 1>& parameters, Eigen::VectorXd& r, Eigen::MatrixXd& J) -> double {
            r.resize(count);
            J.resize(count, 5);

            const double c0p = parameters(0);
            const double c1p = parameters(1);
            const double ap = parameters(2);
            const double x0p = parameters(3);
            const double sig = std::exp(parameters(4));

            const double invSqrt2Sigma = 1.0 / (std::sqrt(2.0) * sig);

            double cost = 0.0;
            for (int i = 0; i < count; ++i) {
                const double ti = t[i];
                const double yi = y[i];

                const double u = (ti - x0p) * invSqrt2Sigma;
                const double erfU = erfValue(u);
                const double f = 0.5 * (1.0 + erfU);
                const double model = c0p + c1p * ti + ap * f;
                const double residual = model - yi;

                const double huberThreshold = 2.5;
                const double absResidual = std::abs(residual);
                const double weight = (absResidual <= huberThreshold) ? 1.0 : (huberThreshold / absResidual);

                r(i) = std::sqrt(weight) * residual;
                cost += r(i) * r(i);

                const double dfDu = (1.0 / std::sqrt(kPi)) * std::exp(-u * u);
                J(i, 0) = std::sqrt(weight);
                J(i, 1) = std::sqrt(weight) * ti;
                J(i, 2) = std::sqrt(weight) * f;
                J(i, 3) = std::sqrt(weight) * (ap * dfDu * (-invSqrt2Sigma));
                J(i, 4) = std::sqrt(weight) * (ap * dfDu * (-u));
            }

            return cost;
        };

        Eigen::VectorXd residuals;
        Eigen::MatrixXd jacobian;

        for (int iter = 0; iter < 25; ++iter) {
            const double cost = evaluateCostAndJacobian(p, residuals, jacobian);
            if (!std::isfinite(cost)) {
                return false;
            }

            if (std::abs(lastCost - cost) < 1e-6) {
                break;
            }
            lastCost = cost;

            Eigen::Matrix<double, 5, 5> h = jacobian.transpose() * jacobian;
            Eigen::Matrix<double, 5, 1> g = jacobian.transpose() * residuals;

            Eigen::Matrix<double, 5, 5> hLm = h;
            for (int d = 0; d < 5; ++d) {
                hLm(d, d) *= (1.0 + lambda);
            }

            Eigen::Matrix<double, 5, 1> dp = -hLm.ldlt().solve(g);
            if (!dp.allFinite()) {
                lambda *= 10.0;
                continue;
            }

            const Eigen::Matrix<double, 5, 1> tryParameters = p + dp;
            const double trySigma = std::exp(tryParameters(4));
            if (!(trySigma > 0.15 && trySigma < 20.0)) {
                lambda *= 10.0;
                continue;
            }

            Eigen::VectorXd residualsTry;
            Eigen::MatrixXd jacobianTry;
            const double tryCost = evaluateCostAndJacobian(tryParameters, residualsTry, jacobianTry);

            if (tryCost < cost) {
                p = tryParameters;
                lambda = std::max(1e-6, lambda * 0.3);
            } else {
                lambda = std::min(1e6, lambda * 10.0);
            }

            if (dp.norm() < 1e-4) {
                break;
            }
        }

        const double x0Estimate = p(3);
        const double sigmaEstimate = std::exp(p(4));

        if (!std::isfinite(x0Estimate) || !std::isfinite(sigmaEstimate)) {
            return false;
        }
        if (std::abs(x0Estimate) > (count - 1) * 0.5 + 0.5) {
            return false;
        }
        if (std::abs(p(2)) < 3.0) {
            return false;
        }

        x0Out = x0Estimate;
        sigmaOut = sigmaEstimate;

        const double amplitudeAbs = std::abs(p(2));
        const double rms = std::sqrt(lastCost / std::max(1, count - 5));
        confidenceOut = (amplitudeAbs / std::max(1e-3, sigmaOut)) / std::max(1e-3, rms);
        return true;
    }
};

std::unique_ptr<SubpixelBase> makeAlgorithm()
{
    return std::make_unique<ErfFitting>();
}

} // namespace

namespace SubpixelEdge {

SubpixelPoint::SubpixelPoint(double xValue, double yValue, double confidenceValue, double gradientValue)
    : x(xValue), y(yValue), confidence(confidenceValue), gradient(gradientValue)
{
}

double SubpixelPoint::distanceTo(const SubpixelPoint& other) const
{
    const double dx = x - other.x;
    const double dy = y - other.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace SubpixelEdge

SubpixelEdgeDetector::SubpixelEdgeDetector(double pixelSize) noexcept
    : pixelSize_(pixelSize)
{
}

cv::Mat SubpixelEdgeDetector::toGray8U(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    if (gray.type() == CV_8UC1) {
        return gray.clone();
    }

    cv::Mat output;
    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(gray, &minValue, &maxValue);

    if (std::abs(maxValue - minValue) < 1e-12) {
        output = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(0));
    } else {
        gray.convertTo(output, CV_8U, 255.0 / (maxValue - minValue), -minValue * 255.0 / (maxValue - minValue));
    }

    return output;
}

void SubpixelEdgeDetector::setImage(const cv::Mat& image)
{
    gray_ = toGray8U(image);
    edges_.clear();
    points_.clear();

#if SUBPIXEL_EDGE_DETECTOR_ENABLE_DEBUG
    edge_.release();
#endif
}

void SubpixelEdgeDetector::setCannyThresholds(double low, double high) noexcept
{
    params_.canny_low = low;
    params_.canny_high = high;
}

void SubpixelEdgeDetector::detectCannyEdges(double low, double high, int aperture, bool l2)
{
    params_.canny_low = low;
    params_.canny_high = high;

#if SUBPIXEL_EDGE_DETECTOR_ENABLE_DEBUG
    if (gray_.empty()) {
        edge_.release();
        return;
    }

    cv::Mat blur;
    cv::GaussianBlur(gray_, blur, {5, 5}, 1.0);
    cv::Canny(blur, edge_, low, high, aperture, l2);
#else
    (void)aperture;
    (void)l2;
#endif
}

void SubpixelEdgeDetector::refineEdgesSubpixel(int windowSize, double presmoothSigma)
{
    edges_.clear();
    points_.clear();

    if (gray_.empty()) {
        return;
    }

    Params effectiveParams = params_;
    if (windowSize > 0) {
        effectiveParams.window_size = windowSize;
    }
    if (presmoothSigma > 0.0) {
        effectiveParams.presmooth_sigma = presmoothSigma;
    }

    effectiveParams.window_size = std::max(3, effectiveParams.window_size | 1);

    auto algorithm = makeAlgorithm();
    algorithm->setCannyThresholds(effectiveParams.canny_low, effectiveParams.canny_high);
    algorithm->setWindowSize(effectiveParams.window_size);
    algorithm->setPreSmoothSigma(effectiveParams.presmooth_sigma);
    algorithm->setUseScharr(effectiveParams.use_scharr);

    points_ = algorithm->detect(gray_);

    edges_.reserve(points_.size());
    for (const auto& point : points_) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
            edges_.emplace_back(point.x, point.y);
        }
    }
}

std::vector<cv::Point2d> SubpixelEdgeDetector::detect(const cv::Mat& image)
{
    setImage(image);
    refineEdgesSubpixel();
    return edges_;
}
