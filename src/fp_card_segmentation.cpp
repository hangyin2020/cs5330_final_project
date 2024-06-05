#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // Read the image
    Mat image = imread("../data/pokemon_cards/caz_oddish1.png");

    imshow("original", image);
    // Convert the image to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Threshold or perform edge detection to obtain a binary image
    // Example using Canny edge detection
    Mat edges;
    Canny(gray, edges, 20, 60, 3);

    // Dilate the edges to make them thicker
    Mat dilatedEdges;
    dilate(edges, dilatedEdges, Mat(), Point(-1,-1), 1); // You can adjust the kernel size for more or less dilation

    // Find contours in the binary image
    vector<vector<Point>> contours;
    findContours(dilatedEdges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    imshow("edges", dilatedEdges);
    waitKey(0);

    // Find the largest contour
    double maxArea = 0;
    vector<Point> largestContour;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            largestContour = contour;
        }
    }

    // Draw the bounding rectangle around the largest contour
    if (!largestContour.empty()) {
        Rect boundingRect = cv::boundingRect(largestContour);
        rectangle(image, boundingRect, Scalar(0, 255, 0), 2);
    }

    // Display the result
    imshow("Largest Contour Rectangle", image);
    waitKey(0);

    // Approximate the contour to a rectangle
    RotatedRect rotatedRect = minAreaRect(largestContour);

    // Calculate the rotation angle to align the longer side vertically
    float angle = rotatedRect.angle;
    if (rotatedRect.size.width > rotatedRect.size.height) {
        angle -= 90;
    }

    printf("angle is %f", angle);

    // Create rotation matrix
    Mat rotationMatrix = getRotationMatrix2D(rotatedRect.center, angle, 1.0);

    // Apply affine transformation to rotate the image
    Mat rotated;
    warpAffine(image, rotated, rotationMatrix, image.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

    // Display the result
    imshow("Rotated", rotated);
    waitKey(0);

    return 0;
}
