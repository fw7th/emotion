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
  static constexpr int FRAME_SIZE = 96;
  static constexpr int NUM_EMOTIONS = 7;
  static std::unordered_map<int, std::string> emotions_;

  ncnn::Net emotion;
  ncnn::Option opt;

  cv::Mat bright_frame;
  cv::Mat gray_frame;
  int num_cores;

  int maxIndex(ncnn::Mat &probs);
  int predict(cv::Mat &frame1);
  int finalPred(ncnn::Mat &input1);
  void preprocess(const cv::Mat &frame);
  void softmax(ncnn::Mat &nums);
  void infer(cv::Mat &infer);

 public:
  Emotion(ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue_,
          const std::string &bin_path_, const std::string &param_path_);
  Emotion(const Emotion &) = delete;             // Delete copy constructor
  Emotion &operator=(const Emotion &) = delete;  // Delete copy assignment
  ~Emotion();
  const std::string &bin_path;
  const std::string &param_path;
  ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue;
  void load();
};
