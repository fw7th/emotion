#pragma once

#include <algorithm>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "customqueue.h"
#include "net.h"
#include "structs.h"

class Emotion {
 private:
  static constexpr int FRAME_SIZE = 64;
  static constexpr int NUM_EMOTIONS = 7;
  static std::unordered_map<int, std::string> emotions_;

  ncnn::Net emotion;
  ncnn::Option opt;

  cv::Mat bright_frame;
  cv::Mat gray_frame;
  int num_cores;

  std::pair<int, float> predict(cv::Mat &frame);

  std::pair<int, float> finalPred(ncnn::Mat &input1);

  void preprocess(const cv::Mat &frame);

  void softmax(ncnn::Mat &nums);

  std::pair<std::string, float> infer(cv::Mat &frame);

  cv::Mat roiCrop(float x1, float y1, float x2, float y2, cv::Mat &frame);

 public:
  Emotion(ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue_,
          ts::TSQueue<std::unique_ptr<FrameInfo>> &output_queue_,
          const std::string &bin_path_, const std::string &param_path_);

  Emotion(const Emotion &) = delete;  // Delete copy constructor

  Emotion &operator=(const Emotion &) = delete;  // Delete copy assignment

  ~Emotion();

  const std::string &bin_path;
  const std::string &param_path;

  ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue;
  ts::TSQueue<std::unique_ptr<FrameInfo>> &output_queue;

  void load();
};
