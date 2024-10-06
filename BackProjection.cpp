#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

cv::Mat toGreyscale(const cv::Mat& image);

int main() {
    string Dataset = "../Dataset/";
    string Results = "../Results/";

    // Define image iterator
    fs::directory_iterator iterpos(Dataset);
    int hBins = 512;  // Hue
    int lBins = 256;  // Lightness
    int channels[] = {0, 1}; // H and L channels
    int histSize[] = {hBins, lBins};
    float hRanges[] = {0, 180}; // Hue ranges
    float lRanges[] = {0, 256}; // Lightness ranges
    const float* ranges[] = {hRanges, lRanges};

    cv::Mat sample_crossings, hist;
    cv::cvtColor(cv::imread("../Pedestrian Crossing Samples.png"),  sample_crossings, cv::COLOR_BGR2HLS);

    cv::calcHist(&sample_crossings, 1, channels, cv::Mat(), hist, 2, histSize, ranges,true, false);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);


    for (fs::directory_iterator finish; iterpos != finish; iterpos++) {
        // Convert image to cv2
        cv::Mat image = cv::imread(iterpos->path().string());

        // // Convert to greyscale
        // cv::Mat greyimage = toGreyscale(image);

        cv::Mat SmoothImage;

        cv::medianBlur(image, SmoothImage, 5);

        cv::Mat hlsimage;
        cvtColor(SmoothImage, hlsimage, cv::COLOR_BGR2HLS);

        cv::Mat backProj;
        int channels_target[] = {0, 1};
        cv::calcBackProject(&hlsimage, 1, channels_target, hist, backProj, ranges, 1);

        cv::Mat thresholded;
        cv::threshold(backProj, thresholded, 10, 255, cv::THRESH_BINARY);

     

        // Saves Image to results folder
        string outputImagePath = Results + iterpos->path().filename().string();
        bool success = cv::imwrite(outputImagePath, thresholded);

        cout << "Applied to Image: " << outputImagePath << endl;
    }

    return 0;
}

cv::Mat toGreyscale(const cv::Mat& image) {
    cv::Mat greyimage;
    cv::cvtColor(image, greyimage, cv::COLOR_BGR2GRAY);
    return greyimage;
}