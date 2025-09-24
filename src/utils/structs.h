#pragma once

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

typedef struct UltraStruct {
  cv::Mat frame;
  std::vector<cv::Mat> crops;
  std::string prediction;
  float confidence;

} UltraStruct;
