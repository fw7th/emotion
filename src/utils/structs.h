#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

typedef struct UltraStruct {
  cv::Mat frame;
  std::vector<cv::Mat> crops;
  int emotion_index;

} UltraStruct;
