#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

cv::Mat toGreyscale(const cv::Mat& image);

int main() {
    string Dataset = "../Dataset/";
    string Results = "../ResultsThresh/";

    // Define image iterator
    fs::directory_iterator iterpos(Dataset);

    for (fs::directory_iterator finish; iterpos != finish; iterpos++) {
        // Convert image to cv2
        cv::Mat image = cv::imread(iterpos->path().string());

        // Convert to greyscale
        cv::Mat greyimage = toGreyscale(image);

        cv::Mat SmoothImage;

        cv::medianBlur(greyimage, SmoothImage, 1);

        cv::Mat binaryimage;
        int thresholdlim = 210;  // Adjust this value to fine-tune the threshold
        cv::threshold(SmoothImage, binaryimage, thresholdlim, 255, cv::THRESH_BINARY);

        // Saves Image to results folder
        string outputImagePath = Results + iterpos->path().filename().string();
        bool success = cv::imwrite(outputImagePath, binaryimage);

        cout << "Applied to Image: " << outputImagePath << endl;
    }

    return 0;
}

cv::Mat toGreyscale(const cv::Mat& image) {
    cv::Mat greyimage;
    cv::cvtColor(image, greyimage, cv::COLOR_BGR2GRAY);
    return greyimage;
}