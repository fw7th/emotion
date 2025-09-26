#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

typedef struct Bbox {
  cv::Point pt1;
  cv::Point pt2;

} Bbox;

typedef struct FrameInfo {
  cv::Mat frame;
  std::vector<Bbox> bboxes;
  std::vector<std::string> predictions;
  std::vector<float> confidences;

} FrameInfo;
