#pragma once

#include "customqueue.h"
#include "net.h"
#include "structs.h"
#include <opencv2/core/mat.hpp>
#include <unordered_map>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

class Emotion {
private:
  static constexpr int FRAME_SIZE = 112;
  static constexpr int NUM_EMOTIONS = 7;
  static std::unordered_map<int, std::string> emotions_;

  ncnn::Net emotion;
  ncnn::Option opt;

  cv::Mat gray_frame;
  cv::Mat resized_frame;
  cv::Mat processed_frame;
  cv::Mat normalized_frame;

  int maxIndex(ncnn::Mat &probs);
  int predict(cv::Mat &frame1);
  int finalPred(ncnn::Mat &input1);

  void preprocess(const cv::Mat &frame);
  void softmax(ncnn::Mat &nums);
  void infer(cv::Mat &infer);

public:
  Emotion(ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue_,
          const std::string &bin_path_, const std::string &param_path_);

  Emotion(const Emotion &) = delete;            // Delete copy constructor
  Emotion &operator=(const Emotion &) = delete; // Delete copy assignment
  ~Emotion();

  ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue;

  const std::string &bin_path;
  const std::string &param_path;

  void load();
};
