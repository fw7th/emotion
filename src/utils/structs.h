#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

// Defines face bbox positions
struct Bbox {
  cv::Point pt1;
  cv::Point pt2;

};

// Info from one frame to be passed with a smart pointer
struct FrameInfo {
  cv::Mat frame;
  std::vector<Bbox> bboxes;
  std::vector<std::string> predictions;
  std::vector<float> confidences;

};
