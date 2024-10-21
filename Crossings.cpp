#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

cv::Mat toGreyscale(const cv::Mat& image);
bool isValidQuadrilateral(const vector<cv::Point>& contour, double minArea, double maxArea, double minAspectRatio, double maxAspectRatio);
bool hasValidNeighbor(const vector<vector<cv::Point>>& quadrilaterals, const vector<cv::Point>& current, double distanceTolerance, double angleTolerance);
double calculateAngle(const cv::Point& p1, const cv::Point& p2);
void drawGroundTruth(cv::Mat& image, int imageNumber, const int groundTruth[][9]);
void drawEncompassingQuadrilateral(cv::Mat& image, const vector<vector<cv::Point>>& validQuadrilaterals);

int main() {
    string Dataset = "../Dataset/";
    string Results = "../Results/";

    // Define image iterator
    fs::directory_iterator iterpos(Dataset);

    // Define area and aspect ratio constraints for quadrilaterals
    const double MIN_AREA = 150.0;
    const double MAX_AREA = 2500.0;
    const double MIN_ASPECT_RATIO = 0.2;
    const double MAX_ASPECT_RATIO = 20;
    // Define distance and angle tolerances for nearby check
    const double DISTANCE_TOLERANCE = 100.0; // pixels
    const double ANGLE_TOLERANCE = 45.0; // degrees

    // Ground truth data
    int pedestrian_crossing_ground_truth[][9] = {
        { 10,0,132,503,113,0,177,503,148},
        { 11,0,131,503,144,0,168,503,177},
        { 12,0,154,503,164,0,206,503,213},
        { 13,0,110,503,110,0,156,503,144},
        { 14,0,95,503,104,0,124,503,128},
        { 15,0,85,503,91,0,113,503,128},
        { 16,0,65,503,173,0,79,503,215},
        { 17,0,43,503,93,0,89,503,146},
        { 18,0,122,503,117,0,169,503,176}
    };

    for (const auto& entry : iterpos) {
        // Convert image to cv2
        cv::Mat image = cv::imread(entry.path().string());
        cv::Mat smoothImage;
        cv::GaussianBlur(image, smoothImage, cv::Size(5, 5), 0);
        
        cv::Mat binaryImage;
        cv::Mat grayImage;
        cv::cvtColor(smoothImage, grayImage, cv::COLOR_BGR2GRAY);
        int thresholdLim = 190;
        cv::threshold(grayImage, binaryImage, thresholdLim, 255, cv::THRESH_BINARY);

        // Morphological operations
        cv::Mat morph_three_Kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat morph_five_Kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_OPEN, morph_three_Kernel);
        cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, morph_five_Kernel);

        // Find contours in the thresholded binary image
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cout << "Number of contours found: " << contours.size() << endl;

        // Clone the original image for drawing contours
        cv::Mat contourImage = image.clone();

        // Store valid quadrilaterals
        vector<vector<cv::Point>> validQuadrilaterals;

        // First pass: collect all valid quadrilaterals
        for (const auto& contour : contours) {
            vector<cv::Point> approx;
            double epsilon = 0.04 * cv::arcLength(contour, true);
            cv::approxPolyDP(contour, approx, epsilon, true);

            // Allow polygons with 4 to 5 sides
            if (approx.size() >= 4 && approx.size() <= 5 && 
                isValidQuadrilateral(approx, MIN_AREA, MAX_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)) {
                validQuadrilaterals.push_back(approx);
            }
        }

        // Second pass: draw quadrilaterals with valid neighbors
        vector<vector<cv::Point>> crossingQuadrilaterals;
        for (const auto& quad : validQuadrilaterals) {
            if (hasValidNeighbor(validQuadrilaterals, quad, DISTANCE_TOLERANCE, ANGLE_TOLERANCE)) {
                // Draw the quadrilateral on the original image in red
                for (size_t j = 0; j < quad.size(); j++) {
                    cv::line(contourImage, quad[j], quad[(j+1)%quad.size()], cv::Scalar(0, 0, 255), 2);
                }
                crossingQuadrilaterals.push_back(quad);
            }
        }

        // Extract image number from filename
        string filename = entry.path().filename().string();
        int imageNumber = stoi(filename.substr(2, filename.find('.') - 2));

        // Draw ground truth
        drawGroundTruth(contourImage, imageNumber, pedestrian_crossing_ground_truth);

        // Draw encompassing quadrilateral
        drawEncompassingQuadrilateral(contourImage, crossingQuadrilaterals);

        // Save both the thresholded image and the image with contours
        string thresholdedImagePath = Results + "thresholded_" + filename;
        string contourImagePath = Results + "contours_" + filename;
        cv::imwrite(thresholdedImagePath, binaryImage);
        cv::imwrite(contourImagePath, contourImage);

        cout << "Applied to Image: " << filename << endl;
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

bool isValidQuadrilateral(const vector<cv::Point>& contour, double minArea, double maxArea, double minAspectRatio, double maxAspectRatio) {
    double area = cv::contourArea(contour);
    if (area < minArea || area > maxArea) {
        return false;
    }

    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
    return (aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio);
}

bool hasValidNeighbor(const vector<vector<cv::Point>>& quadrilaterals, const vector<cv::Point>& current, double distanceTolerance, double angleTolerance) {
    cv::Point currentCenter(0, 0);
    for (const auto& point : current) {
        currentCenter += point;
    }
    currentCenter *= (1.0 / current.size());

    for (const auto& quad : quadrilaterals) {
        if (quad == current) continue;

        cv::Point quadCenter(0, 0);
        for (const auto& point : quad) {
            quadCenter += point;
        }
        quadCenter *= (1.0 / quad.size());

        double distance = cv::norm(currentCenter - quadCenter);

        if (distance < distanceTolerance) {
            double angle = calculateAngle(currentCenter, quadCenter);
            if (std::abs(angle) <= angleTolerance || std::abs(angle - 180) <= angleTolerance) {
                return true;
            }
        }
    }
    return false;
}

double calculateAngle(const cv::Point& p1, const cv::Point& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double angle = std::atan2(dy, dx) * 180.0 / CV_PI;
    return angle;
}

void drawGroundTruth(cv::Mat& image, int imageNumber, const int groundTruth[][9]) {
    // Find the corresponding ground truth for the image
    for (int i = 0; i < 9; i++) {
        if (groundTruth[i][0] == imageNumber) {
            // Extract coordinates
            int x1 = groundTruth[i][1];
            int y1 = groundTruth[i][2];
            int x2 = groundTruth[i][3];
            int y2 = groundTruth[i][4];
            int x3 = groundTruth[i][5];
            int y3 = groundTruth[i][6];
            int x4 = groundTruth[i][7];
            int y4 = groundTruth[i][8];

            // Create a polygon from the coordinates
            cv::Point points[4] = {
                cv::Point(x1, y1),
                cv::Point(x2, y2),
                cv::Point(x4, y4),
                cv::Point(x3, y3)
            };

            // Create a copy of the image for the transparent overlay
            cv::Mat overlay = image.clone();

            // Draw the filled polygon on the overlay
            cv::fillConvexPoly(overlay, points, 4, cv::Scalar(128, 0, 128)); // Purple color

            // Blend the overlay with the original image
            double alpha = 0.4; // Transparency factor
            cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);

            // Draw the outline of the polygon
            for (int j = 0; j < 4; j++) {
                cv::line(image, points[j], points[(j+1)%4], cv::Scalar(128, 0, 128), 2);
            }

            break; // We found the correct ground truth, no need to continue the loop
        }
    }
}

void drawEncompassingQuadrilateral(cv::Mat& image, const vector<vector<cv::Point>>& validQuadrilaterals) {
    if (validQuadrilaterals.empty()) {
        return;  // No quadrilaterals to encompass
    }

    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // Find the leftmost, rightmost, topmost, and bottommost points
    int leftX = imageWidth;
    int rightX = 0;
    int topY = imageHeight;
    int bottomY = 0;
    cv::Point leftBottom, rightBottom, leftTop, rightTop;

    for (const auto& quad : validQuadrilaterals) {
        for (const auto& point : quad) {
            if (point.x < leftX) {
                leftX = point.x;
                if (point.y > leftBottom.y) {
                    leftBottom = point;
                }
            }
            if (point.x > rightX) {
                rightX = point.x;
                if (point.y > rightBottom.y) {
                    rightBottom = point;
                }
            }
            if (point.y < topY) {
                topY = point.y;
                if (point.x < leftTop.x || leftTop.x == 0) {
                    leftTop = point;
                }
                if (point.x > rightTop.x) {
                    rightTop = point;
                }
            }
            bottomY = std::max(bottomY, point.y);
        }
    }

    // Extend the bottom points to the image edges
    leftBottom.x = 0;
    rightBottom.x = imageWidth - 1;

    // Calculate the slope of the top line
    double topSlope = static_cast<double>(rightTop.y - leftTop.y) / (rightTop.x - leftTop.x);

    // Extend the top points to the image edges
    leftTop.x = 0;
    leftTop.y = static_cast<int>(rightTop.y - topSlope * (imageWidth - 1 - rightTop.x));
    rightTop.x = imageWidth - 1;
    rightTop.y = static_cast<int>(leftTop.y + topSlope * (imageWidth - 1));

    // Ensure the y-coordinates are within the image bounds
    leftTop.y = std::max(0, std::min(leftTop.y, imageHeight - 1));
    rightTop.y = std::max(0, std::min(rightTop.y, imageHeight - 1));

    // Create the polygon points
    vector<cv::Point> polygonPoints = {leftTop, rightTop, rightBottom, leftBottom};

    // Create a copy of the image for the transparent overlay
    cv::Mat overlay = image.clone();

    // Draw the filled polygon on the overlay
    cv::fillConvexPoly(overlay, polygonPoints, cv::Scalar(0, 255, 255)); // Yellow color

    // Blend the overlay with the original image
    double alpha = 0.3; // Transparency factor
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);

    // Draw the outline of the polygon
    for (int i = 0; i < 4; i++) {
        cv::line(image, polygonPoints[i], polygonPoints[(i+1)%4], cv::Scalar(0, 255, 255), 2);
    }
}