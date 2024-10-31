#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>  // Add this for ofstream
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <random>
#include <vector>
using namespace std;
namespace fs = std::filesystem;

// Function declarations
cv::Mat toGreyscale(const cv::Mat& image);
bool isValidQuadrilateral(const vector<cv::Point>& contour, double minArea, double maxArea, double minAspectRatio, double maxAspectRatio);
bool hasValidNeighbor(const vector<vector<cv::Point>>& quadrilaterals, const vector<cv::Point>& current, double distanceTolerance, double angleTolerance);
double calculateAngle(const cv::Point& p1, const cv::Point& p2);
void drawGroundTruth(cv::Mat& image, int imageNumber, const int groundTruth[][9]);
void drawEncompassingQuadrilateral(cv::Mat& image, const vector<vector<cv::Point>>& validQuadrilaterals, int imageNumber, const int groundTruth[][9]);
double calculateIoU(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask);
cv::Mat createMaskFromPoints(const vector<cv::Point>& points, const cv::Size& imageSize);
void evaluateDetection(cv::Mat& image, const vector<cv::Point>& predictedPoints, int imageNumber, const int groundTruth[][9]);
void logMetricsToCSV(const string& filename, double iou, double falsePositiveRate, 
                     double recall, double predictedAreaPercentage, double groundTruthAreaPercentage);
int main() {

    
    string Dataset = "../Dataset/";
    string Results = "../Results_Verification/";
    string Verification = "../Verification Dataset/";

    // Define image iterator
    fs::directory_iterator iterpos(Dataset);

    // Define area and aspect ratio constraints for quadrilaterals
    const double MIN_AREA = 100.0;
    const double MAX_AREA = 2870.0;
    const double MIN_ASPECT_RATIO = 0.2;
    const double MAX_ASPECT_RATIO = 10;
    // Define distance and angle tolerances for nearby check
    const double DISTANCE_TOLERANCE = 2.7; // pixels
    const double ANGLE_TOLERANCE = 45.0; // degrees

    // Ground truth data
    // int pedestrian_crossing_ground_truth[][9] = {
    //     { 10,0,132,503,113,0,177,503,148},
    //     { 11,0,131,503,144,0,168,503,177},
    //     { 12,0,154,503,164,0,206,503,213},
    //     { 13,0,110,503,110,0,156,503,144},
    //     { 14,0,95,503,104,0,124,503,128},
    //     { 15,0,85,503,91,0,113,503,128},
    //     { 16,0,65,503,173,0,79,503,215},
    //     { 17,0,43,503,93,0,89,503,146},
    //     { 18,0,122,503,117,0,169,503,176}
    // };

    int pedestrian_crossing_ground_truth[][9] = {
    { 10,0,132,503,113,0,177,503,148},
    { 11,0,131,503,144,0,168,503,177},
    { 12,0,154,503,164,0,206,503,213},
    { 13,0,110,503,110,0,156,503,144},
    { 14,0,95,503,104,0,124,503,128},
    { 15,0,85,503,91,0,113,503,128},
    { 16,0,65,503,173,0,79,503,215},
    { 17,0,43,503,93,0,89,503,146},
    { 18,0,122,503,117,0,169,503,176},
    { 20,0,157,503,131,0,223,503,184},
    { 21,0,140,503,136,0,190,503,183},
    { 22,0,114,503,97,0,140,503,123},
    { 23,0,133,503,122,0,198,503,186},
    { 24,0,107,503,93,0,146,503,118},
    { 25,0,58,503,164,0,71,503,204},
    { 26,0,71,503,131,0,106,503,199},
    { 27,0,138,503,151,0,179,503,193}
    };

    for (const auto& entry : iterpos) {
        // Convert image to cv2
        cv::Mat image = cv::imread(entry.path().string());
        cv::Mat smoothImage;
        cv::GaussianBlur(image, smoothImage, cv::Size(5, 5), 0);
        
        cv::Mat binaryImage, binaryImage0, binaryImage1;
        cv::Mat grayImage;
        cv::cvtColor(smoothImage, grayImage, cv::COLOR_BGR2GRAY);
        int thresholdLim = 190;
        cv::threshold(grayImage, binaryImage0, thresholdLim, 255, cv::THRESH_BINARY);

        // Morphological operations
        cv::Mat morph_three_Kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat morph_five_Kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::morphologyEx(binaryImage0, binaryImage1, cv::MORPH_OPEN, morph_three_Kernel);
        cv::morphologyEx(binaryImage1, binaryImage, cv::MORPH_CLOSE, morph_five_Kernel);

        // Find contours in the thresholded binary image
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // Create an image to display all initial contours
        
        

        cout << "Number of contours found: " << contours.size() << endl;

        // Clone the original image for drawing contours
        cv::Mat contourImage = image.clone();

        // Store valid quadrilaterals
        vector<vector<cv::Point>> validQuadrilaterals;

        // First pass: collect all valid quadrilaterals
        for (const auto& contour : contours) {
            vector<cv::Point> approx;
            double epsilon = 0.02 * cv::arcLength(contour, true);
            cv::approxPolyDP(contour, approx, epsilon, true);
            
            // Allow polygons with 4 to 5 sides
            if (approx.size() >= 3 && approx.size() <= 6 && 
                isValidQuadrilateral(approx, MIN_AREA, MAX_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)) {
                validQuadrilaterals.push_back(approx);
            }
        }
        cv::Mat allContoursImage;
        cv::cvtColor(binaryImage, allContoursImage, cv::COLOR_GRAY2BGR);
        cv::drawContours(allContoursImage, validQuadrilaterals, -1, cv::Scalar(0, 255, 0), 2);  // Draw all contours in green

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

        // Draw encompassing quadrilateral and evaluate
        drawEncompassingQuadrilateral(contourImage, crossingQuadrilaterals, imageNumber, pedestrian_crossing_ground_truth);

        // Save both the thresholded image and the image with contours
        string thresholdedImagePath = Results + "thresholded_" + filename;
        string binary0 = Results + "binary0_" + filename;
        string binary1 = Results + "binary1_" + filename;
        string contourImagePath = Results + "contours_" + filename;
        string smoothImagePath = Results + "smooth_" + filename;
        string greyPath = Results + "grey_" + filename;
        cv::imwrite(thresholdedImagePath, binaryImage);
        cv::imwrite(contourImagePath, contourImage);
        cv::imwrite(smoothImagePath, smoothImage);
        // cv::imwrite(greyPath, grayImage);
        // cv::imwrite(binary0, binaryImage0);
        // cv::imwrite(binary1, binaryImage1);
        string allContoursImagePath = Results + "all_contours_" + filename;
        cv::imwrite(allContoursImagePath, allContoursImage);
        cout << "All contours image saved as: " << allContoursImagePath << endl;

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

double calculateQuadRadius(const vector<cv::Point>& quad) {
    // Calculate the center first
    cv::Point center(0, 0);
    for (const auto& point : quad) {
        center += point;
    }
    center *= (1.0 / quad.size());
    
    // Find the maximum distance from center to any vertex (radius)
    double maxRadius = 0;
    for (const auto& point : quad) {
        double dist = cv::norm(point - center);
        maxRadius = std::max(maxRadius, dist);
    }
    return maxRadius;
}

bool hasValidNeighbor(const vector<vector<cv::Point>>& quadrilaterals, 
                     const vector<cv::Point>& current, 
                     double distanceToleranceFactor,  // This is now a multiplier
                     double angleTolerance) {
    // Calculate center of current quadrilateral
    cv::Point currentCenter(0, 0);
    for (const auto& point : current) {
        currentCenter += point;
    }
    currentCenter *= (1.0 / current.size());
    
    // Calculate radius of current quadrilateral
    double currentRadius = calculateQuadRadius(current);
    
    // Calculate maximum allowed distance based on radius
    double maxAllowedDistance = currentRadius * distanceToleranceFactor;

    for (const auto& quad : quadrilaterals) {
        if (quad == current) continue;

        // Calculate center of neighboring quadrilateral
        cv::Point quadCenter(0, 0);
        for (const auto& point : quad) {
            quadCenter += point;
        }
        quadCenter *= (1.0 / quad.size());

        double distance = cv::norm(currentCenter - quadCenter);
        
        // Check if the neighbor is within the relative distance
        if (distance < maxAllowedDistance) {
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
    for (int i = 0; i < 19; i++) {
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

            break;
        }
    }
}

void fitLineRANSAC(const vector<cv::Point2f>& points, cv::Vec4f& lineParams,
                   int iterations = 6000,          // Increased iterations
                   double threshold = 3.0) {       // Decreased threshold for tighter fit
    int n_points = points.size();
    if (n_points < 2) return;

    unsigned seed = 42;
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, n_points - 1);

    double best_error = std::numeric_limits<double>::max();
    cv::Vec4f best_line;
    int best_inliers = 0;

    // Constants for stricter validation
    const double MIN_POINT_DISTANCE = 5.0;    // Minimum distance between sampled points
    const double MIN_INLIER_RATIO = 0.3;       // At least 60% of points must be inliers
    const double MAX_ANGLE_FROM_HORIZONTAL = 30.0; // Maximum allowed angle from horizontal (degrees)
    const double MAX_AVERAGE_ERROR = 30.0;      // Maximum allowed average error for inliers

    for (int iter = 0; iter < iterations; iter++) {
        // 1. Sample two points
        int idx1 = distribution(generator);
        int idx2 = distribution(generator);
        if (idx1 == idx2) continue;

        cv::Point2f p1 = points[idx1];
        cv::Point2f p2 = points[idx2];

        // 2. Check if points are far enough apart
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < MIN_POINT_DISTANCE) continue;

        // 3. Check angle from horizontal
        double angle = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);
        if (angle > MAX_ANGLE_FROM_HORIZONTAL && angle < (180.0 - MAX_ANGLE_FROM_HORIZONTAL)) {
            continue;  // Skip if line is too vertical
        }

        // 4. Compute normalized direction vector
        cv::Point2f dir(dx / dist, dy / dist);

        // 5. Count inliers and compute error with stricter criteria
        int inliers = 0;
        double total_error = 0;
        vector<double> errors;  // Store errors for standard deviation calculation
        
        for (const auto& point : points) {
            // Compute distance from point to line
            double distance = std::abs((point.x - p1.x) * (-dir.y) + 
                                     (point.y - p1.y) * dir.x);
            
            if (distance < threshold) {
                inliers++;
                total_error += distance;
                errors.push_back(distance);
            }
        }

        // 6. Check if we have enough inliers
        double inlier_ratio = static_cast<double>(inliers) / n_points;
        if (inlier_ratio < MIN_INLIER_RATIO) continue;

        // 7. Calculate average error and standard deviation
        if (inliers > 0) {
            double avg_error = total_error / inliers;
            if (avg_error > MAX_AVERAGE_ERROR) continue;

            // Calculate standard deviation of errors
            double sq_sum = 0;
            for (double err : errors) {
                sq_sum += (err - avg_error) * (err - avg_error);
            }
            double std_dev = std::sqrt(sq_sum / inliers);

            // 8. Update best model if this is better
            bool is_better = false;
            if (inliers > best_inliers) {
                is_better = true;
            } else if (inliers == best_inliers && avg_error < best_error) {
                is_better = true;
            }

            if (is_better && std_dev < threshold) {  // Add standard deviation check
                best_inliers = inliers;
                best_error = avg_error;
                best_line[0] = dir.x;
                best_line[1] = dir.y;
                best_line[2] = p1.x;
                best_line[3] = p1.y;

                // Debug output for the best line so far
                std::cout << "New best line found:" << std::endl;
                std::cout << "  Inlier ratio: " << inlier_ratio << std::endl;
                std::cout << "  Average error: " << avg_error << std::endl;
                std::cout << "  Standard deviation: " << std_dev << std::endl;
                std::cout << "  Angle from horizontal: " << angle << " degrees" << std::endl;
            }
        }
    }

    // Only accept the result if we found a good enough line
    if (best_inliers > 0) {
        lineParams = best_line;
    } else {
        std::cout << "Warning: No acceptable line found!" << std::endl;
        // Set a default horizontal line in the middle of the points
        double avg_y = 0;
        for (const auto& pt : points) {
            avg_y += pt.y;
        }
        avg_y /= points.size();
        lineParams = cv::Vec4f(1.0, 0.0, points[0].x, avg_y);  // Horizontal line
    }
}

void drawEncompassingQuadrilateral(cv::Mat& image, const vector<vector<cv::Point>>& validQuadrilaterals, 
                                 int imageNumber, const int groundTruth[][9]) {
    if (validQuadrilaterals.empty()) return;

    // Debug: Draw all valid quadrilaterals in blue
    cv::Mat debugImage = image.clone();
    for (const auto& quad : validQuadrilaterals) {
        for (size_t i = 0; i < quad.size(); i++) {
            cv::line(debugImage, quad[i], quad[(i+1)%quad.size()], cv::Scalar(255, 0, 0), 2);
        }
    }
    cv::imwrite("../Results_Verification/debug_quads_" + std::to_string(imageNumber) + ".png", debugImage);

    // Collect all points and convert to Point2f
    vector<cv::Point2f> allPoints;
    for (const auto& quad : validQuadrilaterals) {
        for (const auto& point : quad) {
            allPoints.push_back(cv::Point2f(static_cast<float>(point.x), 
                                          static_cast<float>(point.y)));
        }
    }

    // Debug: Print number of points
    std::cout << "Number of points for RANSAC: " << allPoints.size() << std::endl;

    // Fit line using RANSAC
    cv::Vec4f lineParams;
    fitLineRANSAC(allPoints, lineParams);

    cv::Point2f direction(lineParams[0], lineParams[1]);
    cv::Point2f point(lineParams[2], lineParams[3]);
    
    // Debug: Print direction vector
    std::cout << "Direction vector: (" << direction.x << ", " << direction.y << ")" << std::endl;

    // Calculate perpendicular direction
    cv::Point2f perpDirection(-direction.y, direction.x);
    const double MAX_WIDTH = 100.0;  // Maximum allowed width in pixels
    const double MIN_WIDTH = 1.0;   // Minimum allowed width in pixels
    // Find the extreme points along the perpendicular direction
    double minPerpProj = std::numeric_limits<double>::max();
    double maxPerpProj = std::numeric_limits<double>::lowest();

    // First pass to find base line (minimum projection)
    for (const auto& pt : allPoints) {
        double perpProj = (pt.x - point.x) * perpDirection.x + 
                        (pt.y - point.y) * perpDirection.y;
        minPerpProj = std::min(minPerpProj, perpProj);
    }

    // Second pass to find max projection, but only consider points within MAX_WIDTH
    for (const auto& pt : allPoints) {
        double perpProj = (pt.x - point.x) * perpDirection.x + 
                        (pt.y - point.y) * perpDirection.y;
        // Only consider points that are within MAX_WIDTH of our minimum
        if (perpProj - minPerpProj <= MAX_WIDTH) {
            maxPerpProj = std::max(maxPerpProj, perpProj);
        }
    }

    // Calculate width and check if it's valid
    double width = maxPerpProj - minPerpProj;
    

    std::cout << "Detected width: " << width << " pixels" << std::endl;

    if (width > MAX_WIDTH || width < MIN_WIDTH) {
        std::cout << "Width " << width << " pixels is outside acceptable range ["
                  << MIN_WIDTH << ", " << MAX_WIDTH << "]. Skipping detection." << std::endl;
        return;
    }

    // Add padding (but ensure we don't exceed max width)
    double padding = std::min(width * 0., (MAX_WIDTH - width) / 2);  // 10% padding or whatever space is left
    maxPerpProj += padding;
    minPerpProj -= padding;

    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // For left edge (x = 0)
    double t_left = -point.x / direction.x;
    cv::Point2f leftIntersect(0, point.y + t_left * direction.y);

    // For right edge (x = imageWidth - 1)
    double t_right = (imageWidth - 1 - point.x) / direction.x;
    cv::Point2f rightIntersect(imageWidth - 1, point.y + t_right * direction.y);

    // Debug: Print intersection points
    std::cout << "Left intersect: (" << leftIntersect.x << ", " << leftIntersect.y << ")" << std::endl;
    std::cout << "Right intersect: (" << rightIntersect.x << ", " << rightIntersect.y << ")" << std::endl;

    // Calculate the four corners with added bounds checking
    auto createPoint = [imageWidth, imageHeight](float x, float y) {
        return cv::Point(
            std::max(0, std::min(imageWidth - 1, static_cast<int>(x))),
            std::max(0, std::min(imageHeight - 1, static_cast<int>(y)))
        );
    };

    cv::Point topLeft = createPoint(
        leftIntersect.x,
        leftIntersect.y + perpDirection.y * minPerpProj
    );
    cv::Point topRight = createPoint(
        rightIntersect.x,
        rightIntersect.y + perpDirection.y * minPerpProj
    );
    cv::Point bottomLeft = createPoint(
        leftIntersect.x,
        leftIntersect.y + perpDirection.y * maxPerpProj
    );
    cv::Point bottomRight = createPoint(
        rightIntersect.x,
        rightIntersect.y + perpDirection.y * maxPerpProj
    );

    // Verify the final quadrilateral width
    double finalHeight1 = cv::norm(topLeft - bottomLeft);
    double finalHeight2 = cv::norm(topRight - bottomRight);
    double avgHeight = (finalHeight1 + finalHeight2) / 2;

    if (avgHeight > MAX_WIDTH) {
        std::cout << "Final quadrilateral height " << avgHeight 
                  << " exceeds maximum allowed width. Skipping detection." << std::endl;
        return;
    }

    // Debug: Print corner points
    std::cout << "Corners: TL(" << topLeft.x << "," << topLeft.y << ") "
              << "TR(" << topRight.x << "," << topRight.y << ") "
              << "BL(" << bottomLeft.x << "," << bottomLeft.y << ") "
              << "BR(" << bottomRight.x << "," << bottomRight.y << ")" << std::endl;
    std::cout << "Final quadrilateral heights: Left=" << finalHeight1 
              << "px, Right=" << finalHeight2 << "px" << std::endl;

    vector<cv::Point> polygonPoints = {topLeft, topRight, bottomRight, bottomLeft};

    // Draw the final quadrilateral
    cv::Mat overlay = image.clone();
    cv::fillConvexPoly(overlay, polygonPoints, cv::Scalar(0, 255, 255));
    double alpha = 0.3;
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
    for (int i = 0; i < 4; i++) {
        cv::line(image, polygonPoints[i], polygonPoints[(i+1)%4], cv::Scalar(0, 255, 255), 2);
    }

    evaluateDetection(image, polygonPoints, imageNumber, groundTruth);
}


    double calculateIoU(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask) {
        cv::Mat intersection, union_;
        
        // Calculate intersection and union
        cv::bitwise_and(predictedMask, groundTruthMask, intersection);
        cv::bitwise_or(predictedMask, groundTruthMask, union_);
        
        // Count non-zero pixels
        double intersectionArea = cv::countNonZero(intersection);
        double unionArea = cv::countNonZero(union_);
        
        // Calculate IoU
        if (unionArea > 0) {
            return intersectionArea / unionArea;
        }
        return 0.0;
    }

    cv::Mat createMaskFromPoints(const vector<cv::Point>& points, const cv::Size& imageSize) {
        cv::Mat mask = cv::Mat::zeros(imageSize, CV_8UC1);
        vector<vector<cv::Point>> contours = {points};
        cv::fillPoly(mask, contours, cv::Scalar(255));
        return mask;
    }

void evaluateDetection(cv::Mat& image, const vector<cv::Point>& predictedPoints, 
                    int imageNumber, const int groundTruth[][9]) {
    // Find the corresponding ground truth for the image
    for (int i = 0; i < 19; i++) {
        if (groundTruth[i][0] == imageNumber) {
            // Create ground truth points
            vector<cv::Point> groundTruthPoints = {
                cv::Point(groundTruth[i][1], groundTruth[i][2]),
                cv::Point(groundTruth[i][3], groundTruth[i][4]),
                cv::Point(groundTruth[i][7], groundTruth[i][8]),
                cv::Point(groundTruth[i][5], groundTruth[i][6])
            };

            // Create masks for both predicted and ground truth
            cv::Mat predictedMask = createMaskFromPoints(predictedPoints, image.size());
            cv::Mat groundTruthMask = createMaskFromPoints(groundTruthPoints, image.size());

            // Calculate intersection for recall
            cv::Mat intersection;
            cv::bitwise_and(predictedMask, groundTruthMask, intersection);
            
            // Calculate areas
            double intersectionArea = cv::countNonZero(intersection);
            double predictedArea = cv::countNonZero(predictedMask);
            double groundTruthArea = cv::countNonZero(groundTruthMask);
            double totalImageArea = image.rows * image.cols;

            // Calculate metrics
            double iou = calculateIoU(predictedMask, groundTruthMask);
            double recall = (intersectionArea / groundTruthArea) * 100;  // % of ground truth covered
            
            // Calculate false positive area (predicted area that's not in ground truth)
            cv::Mat falsePositive;
            cv::bitwise_xor(predictedMask, intersection, falsePositive);
            double falsePositiveArea = cv::countNonZero(falsePositive);
            double falsePositiveRate = (falsePositiveArea / predictedArea) * 100;  // % of prediction that's false positive

            // Calculate percentages for console output
            double predictedPercentage = (predictedArea / totalImageArea) * 100;
            double groundTruthPercentage = (groundTruthArea / totalImageArea) * 100;

            // Log metrics to CSV
            string filename = "pc" + std::to_string(imageNumber) + ".png";
            logMetricsToCSV(filename, iou, falsePositiveRate, recall, 
                           predictedPercentage, groundTruthPercentage);

            // Put the metrics on the image
            cv::putText(image, 
                    "IoU: " + std::to_string(iou).substr(0, 5), 
                    cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    1, 
                    cv::Scalar(0, 255, 0), 
                    2);

            cv::putText(image, 
                    "Recall: " + std::to_string(recall).substr(0, 5) + "%", 
                    cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    1, 
                    cv::Scalar(0, 255, 255), 
                    2);

            cv::putText(image, 
                    "FP Rate: " + std::to_string(falsePositiveRate).substr(0, 5) + "%", 
                    cv::Point(10, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    1, 
                    cv::Scalar(128, 0, 128), 
                    2);

            // Save masks for debugging if needed
            cv::imwrite("../Results_Verification/predicted_mask_" + filename, predictedMask);
            cv::imwrite("../Results_Verification/ground_truth_mask_" + filename, groundTruthMask);
            cv::imwrite("../Results_Verification/intersection_mask_" + filename, intersection);
            cv::imwrite("../Results_Verification/false_positive_mask_" + filename, falsePositive);

            // Print to console as well
            cout << "Image " << imageNumber << " Evaluation:" << endl;
            cout << "IoU: " << iou << endl;
            cout << "Recall (GT Coverage): " << recall << "%" << endl;
            cout << "False Positive Rate: " << falsePositiveRate << "%" << endl;
            cout << "Predicted Area: " << predictedPercentage << "% of image" << endl;
            cout << "Ground Truth Area: " << groundTruthPercentage << "% of image" << endl;
            cout << "-------------------" << endl;

            break;
        }
    }
}
    // Function to log data to CSV
void logMetricsToCSV(const string& filename, double iou, double falsePositiveRate, 
                     double recall, double predictedAreaPercentage, double groundTruthAreaPercentage) {
    
    static bool firstWrite = true;
    std::ofstream csvFile;
    
    if (firstWrite) {
        // Create new file with headers
        csvFile.open("../Results_Verification/detection_metrics.csv");
        csvFile << "Filename,IoU,False Positive Rate (%),Recall (%),Predicted Area (%),Ground Truth Area (%)\n";
        firstWrite = false;
    } else {
        // Append to existing file
        csvFile.open("../Results_Verification/detection_metrics.csv", std::ios::app);
    }
    
    // Set precision for floating point numbers
    csvFile << std::fixed << std::setprecision(2);
    
    // Write the data
    csvFile << filename << ","
            << iou << ","
            << falsePositiveRate << ","
            << recall << ","
            << predictedAreaPercentage << ","
            << groundTruthAreaPercentage << "\n";
    
    csvFile.close();
}