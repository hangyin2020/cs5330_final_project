/*

    Wenqing Fan
    Hang Yin
    Spring 2024
    CS 5330 Computer Vision

    Task 3 (Segment the image)
    Segment function for a source image, it will return a color map with the 
    topN region with different colors. Also will return a regionMap, all pixels
    will have its region number. The minimum area for a region is minArea,
    anything smaller than minArea will be ignored for now.

*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


// Custom sort function to sort the regions based on the size
bool compareRegionAreas(const std::pair<int, int> &a,
                        const std::pair<int, int> &b) {
    return a.first > b.first;  // Sort in descending order
}

// Segment function for a source image, it will return a color map with the 
// topN region with different colors. Also will return a regionMap, all pixels
// will have its region number.
int segment(cv::Mat &src, cv::Mat &color_map, cv::Mat &regionMap, int topN, int minArea) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Threshold the image
    cv::Mat thresh;
    cv::threshold(gray, thresh, 160, 190, cv::THRESH_BINARY_INV);

    // Apply connected component labeling
    cv::Mat labeled_image, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        thresh, labeled_image, stats, centroids, 8, CV_32S);

    // Create a color map using a predefined palette
    std::vector<cv::Vec3b> color_palette = {
        cv::Vec3b(0, 255, 0),    // Green
        cv::Vec3b(0, 0, 255),    // Red
        cv::Vec3b(255, 0, 0),    // Blue
        cv::Vec3b(255, 255, 0),  // Cyan
        cv::Vec3b(255, 0, 255),  // Magenta
        cv::Vec3b(0, 255, 255)   // Yellow
        
        // Here we only have 6 colors for now
    };

    // Create a color map
    color_map = cv::Mat::zeros(labeled_image.size(), CV_8UC3);

    // Store region areas and their corresponding labels
    std::vector<std::pair<int, int>> region_areas;

    // Assign colors to labeled regions based on size threshold
    for (int label = 1; label < num_labels; ++label) {
        // Check if the total number of pixels in the region exceeds the
        // threshold
        if (stats.at<int32_t>(label, cv::CC_STAT_AREA) > minArea) {
            region_areas.push_back(std::make_pair(
                stats.at<int32_t>(label, cv::CC_STAT_AREA), label));
        }
    }

    // Sort regions by area in descending order
    std::sort(region_areas.begin(), region_areas.end(), compareRegionAreas);

    // Keep only the top N regions
    int num_regions_to_keep =
        std::min(topN, static_cast<int>(region_areas.size()));

    // Assign new labels to the top N regions starting from 1
    int new_label = 1;

    // Clear regionMap
    regionMap = cv::Mat::zeros(labeled_image.size(), CV_32S);

    // Assign colors to the top N largest regions
    for (int i = 0; i < num_regions_to_keep; ++i) {
        int old_label = region_areas[i].second;
        regionMap.setTo(
            new_label,
            labeled_image ==
                old_label);  // Assign new_label to regionMap where the pixel
                             // matches the current label in labeled_image

        // Get color from the predefined palette
        cv::Vec3b color = color_palette[i];

        // Create a mask for the current labeled region
        cv::Mat mask = (labeled_image == old_label);

        // Assign color to the color map for the current region
        color_map.setTo(color, mask);

        // Increment new label for the next region
        new_label++;
    }

    return 0;
}
