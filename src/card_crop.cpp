#include <opencv2/opencv.hpp>
#include "headers.h"

using namespace cv;
using namespace std;

int main() {

    Mat src = imread("../data/pokemon_cards/caz_oddish1.png");
    Mat color_map;
    Mat regionMap;
    int topN = 8;
    int minArea = 500;
    segment(src, color_map, regionMap, topN, minArea);
    
    imshow("original", src);
    imshow("segmentation", color_map);
    waitKey(0);

    // Find region pixels
    std::vector<cv::Point> regionPixels;
    cv::findNonZero(regionMap == 1, regionPixels);

    // Check if regionPixels vector is empty, which is regionID does not exist
    if (regionPixels.empty()) {
        return 0; 
    }

    // Compute the oriented bounding box
    cv::RotatedRect rect = cv::minAreaRect(regionPixels);
    // Draw the oriented bounding box
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(src, vertices[i], vertices[(i + 1) % 4],
                 cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("rect", src);
    waitKey(0);

    return 0;
}
