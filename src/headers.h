#ifndef HEADERS_H
#define HEADERS_H

#include <opencv2/opencv.hpp>
#include <iostream>

int segment(cv::Mat &src, cv::Mat &color_map, cv::Mat &regionMap, int topN, int minArea);


#endif
