#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

cv::Mat toGreyscale(const cv::Mat& image);
double getAngle(const cv::RotatedRect& rr);
bool areParallel(double angle1, double angle2, double threshold);
bool hasParallelNeighbor(const vector<cv::RotatedRect>& rotatedRects, int index, double angleTolerance, double distanceTolerance);

int main() {
    string Dataset = "../Dataset/";
    string Results = "../Results/";

    // Define image iterator
    fs::directory_iterator iterpos(Dataset);

    int hBins = 512; // Hue
    int lBins = 256; // Lightness
    int channels[] = {0, 1}; // H and L channels
    int histSize[] = {hBins, lBins};
    float hRanges[] = {0, 180}; // Hue ranges
    float lRanges[] = {0, 256}; // Lightness ranges
    const float* ranges[] = {hRanges, lRanges};

    cv::Mat sample_crossings, hist;
    cv::cvtColor(cv::imread("../Pedestrian Crossing Samples_2.png"), sample_crossings, cv::COLOR_BGR2HLS);
    cv::calcHist(&sample_crossings, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    // Define minimum width and height for bounding boxes
    const int MIN_WIDTH = 13;
    const int MIN_HEIGHT = 15;
    // Define minimum rectangularity ratio (relaxed)
    const double MIN_RECTANGULARITY = 0.3;
    // Define angle and distance tolerances for parallel check
    const double ANGLE_TOLERANCE = 19.0; // degrees
    const double DISTANCE_TOLERANCE = 180.0; // pixels

    for (fs::directory_iterator finish; iterpos != finish; iterpos++) {
        // Convert image to cv2
        cv::Mat image = cv::imread(iterpos->path().string());
        cv::Mat SmoothImage;
        cv::GaussianBlur(image, SmoothImage, cv::Size(7, 7), 0);
        cv::Mat hlsimage;
        cvtColor(SmoothImage, hlsimage, cv::COLOR_BGR2HLS);

        cv::Mat backProj;
        int channels_target[] = {0, 1};
        cv::calcBackProject(&hlsimage, 1, channels_target, hist, backProj, ranges, 1);

        cv::Mat thresholded;
        cv::threshold(backProj, thresholded, 20, 255, cv::THRESH_BINARY);

        cv::Mat five_by_five_element(8, 8, CV_8U, cv::Scalar(1));
        cv::Mat three_element(1, 1, CV_8U, cv::Scalar(1));
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_OPEN, three_element);
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE, five_by_five_element);

        // Find contours in the thresholded binary image
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(thresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cout << "Number of contours found: " << contours.size() << endl;

        // Clone the original image for drawing contours
        cv::Mat contourImage = image.clone();

        // Store rotated rectangles for valid contours
        vector<cv::RotatedRect> rotatedRects;

        // First pass: collect all potentially valid contours
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::RotatedRect rr = cv::minAreaRect(contours[i]);
            
            // Calculate contour area and rotated rectangle area
            double contourArea = cv::contourArea(contours[i]);
            double rrArea = rr.size.width * rr.size.height;
            
            // Calculate rectangularity
            double rectangularity = contourArea / rrArea;

            // Check if the contour meets the minimum size criteria and relaxed rectangularity
            if (rr.size.width >= MIN_WIDTH && rr.size.height >= MIN_HEIGHT && rectangularity >= MIN_RECTANGULARITY) {
                rotatedRects.push_back(rr);
            }
        }

        // Second pass: draw contours with parallel neighbors
        for (size_t i = 0; i < rotatedRects.size(); ++i) {
            if (hasParallelNeighbor(rotatedRects, i, ANGLE_TOLERANCE, DISTANCE_TOLERANCE)) {
                // Draw the contour on the original image in red
                cv::Point2f rect_points[4];
                rotatedRects[i].points(rect_points);
                for (int j = 0; j < 4; j++) {
                    cv::line(contourImage, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0, 0, 255), 2);
                }
                cout << "Contour drawn with angle: " << getAngle(rotatedRects[i]) << " degrees" << endl;
            }
        }

        // Save both the thresholded image and the image with contours
        string thresholdedImagePath = Results + "thresholded_" + iterpos->path().filename().string();
        string contourImagePath = Results + "contours_" + iterpos->path().filename().string();
        cv::imwrite(thresholdedImagePath, thresholded);
        cv::imwrite(contourImagePath, contourImage);

        cout << "Applied to Image: " << iterpos->path().filename().string() << endl;
        cout << "Thresholded image saved as: " << thresholdedImagePath << endl;
        cout << "Contour image saved as: " << contourImagePath << endl;
    }

    return 0;
}

cv::Mat toGreyscale(const cv::Mat& image) {
    cv::Mat greyimage;
    cv::cvtColor(image, greyimage, cv::COLOR_BGR2GRAY);
    return greyimage;
}

double getAngle(const cv::RotatedRect& rr) {
    if (rr.size.width < rr.size.height) {
        return rr.angle + 180;
    } else {
        return rr.angle + 90;
    }
}

bool areParallel(double angle1, double angle2, double threshold) {
    double diff = fabs(angle1 - angle2);
    return (diff < threshold || fabs(diff - 180) < threshold);
}

bool hasParallelNeighbor(const vector<cv::RotatedRect>& rotatedRects, int index, double angleTolerance, double distanceTolerance) {
    cv::RotatedRect rr1 = rotatedRects[index];
    double angle1 = getAngle(rr1);

    for (size_t i = 0; i < rotatedRects.size(); ++i) {
        if (i == index) continue;

        cv::RotatedRect rr2 = rotatedRects[i];
        double angle2 = getAngle(rr2);

        // Check if angles are parallel
        if (areParallel(angle1, angle2, angleTolerance)) {
            // Check if rectangles are close enough
            double distance = cv::norm(rr1.center - rr2.center);
            if (distance < distanceTolerance) {
                return true;
            }
        }
    }
    return false;
}