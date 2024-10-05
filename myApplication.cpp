#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

int main() {

    string Dataset = "Dataset/";
    string Results = "Results/";

    //Define image interator
    fs ::directory_iterator iterpos(Dataset);
    for(fs::directory_iterator finish; iterpos != finish; iterpos++){ //Do all tasks within for loop


        //Does greyscale to image
        cv::Mat image = cv::imread(iterpos->path().string());
        cv::Mat greyimage;
        cv::cvtColor(image, greyimage, cv::COLOR_BGR2GRAY);

        //Saves Image to results folder
        string outputImagePath = Results + iterpos->path().filename().string();
        cv::imwrite(outputImagePath, greyimage);

        cout << "Applied to Image: " << outputImagePath << endl;

    }

    return 0;

}
