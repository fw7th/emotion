#include "emo.h"

#include <cfloat>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <utility>

#include "mat.h"

Emotion::Emotion(ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue_,
                 ts::TSQueue<std::unique_ptr<FrameInfo>>& output_queue_,
                 const std::string& bin_path_, const std::string& param_path_)
    : input_queue(input_queue_),
      output_queue(output_queue_),
      bin_path(bin_path_),
      param_path(param_path_) {
  num_cores = std::thread::hardware_concurrency();

  // Pre-allocate OpenCV processing buffers
  gray_frame.create(FRAME_SIZE, FRAME_SIZE, CV_8UC1);
  bright_frame.create(FRAME_SIZE, FRAME_SIZE, CV_8UC1);

  // Load and configure NCNN model
  emotion.load_param(param_path.data());
  emotion.load_model(bin_path.data());

  emotion.opt.use_vulkan_compute = false;
  emotion.opt.use_fp16_arithmetic = true;
  emotion.opt.use_int8_arithmetic = false;
  emotion.opt.use_packing_layout = true;
  emotion.opt.use_sgemm_convolution = true;
  emotion.opt.use_winograd_convolution = true;
}

Emotion::~Emotion() { emotion.clear(); }

std::unordered_map<int, std::string> Emotion::emotions_ = {
    {0, "Angry"},   {1, "Disgust"}, {2, "Fear"},    {3, "Happy"},
    {4, "Neutral"}, {5, "Sad"},     {6, "Surprise"}};

void Emotion::load() {
  int frame_count = 0;
  float frame_time = 0;

  while (true) {
    auto loop_start = std::chrono::steady_clock::now();

    // Get frame from input queue
    if (input_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto opt_ultra_ptr = input_queue.pop();
    if (!opt_ultra_ptr.has_value()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    cv::Mat img = opt_ultra_ptr.value()->frame;
    if (img.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    // Process each detected face
    const std::vector<Bbox>& boxes = opt_ultra_ptr.value()->bboxes;
    if (!boxes.empty()) {
      for (const auto& box : boxes) {
        cv::Mat crop = roiCrop(box.pt1.x, box.pt1.y, box.pt2.x, box.pt2.y, img);
        auto result = infer(crop);
        opt_ultra_ptr.value()->predictions.emplace_back(result.first);
        opt_ultra_ptr.value()->confidences.emplace_back(result.second);
      }
    }

    output_queue.push(opt_ultra_ptr.value());

    // Calculate timing and FPS
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

    frame_time += loop_time.count() * 1000;
    frame_count++;

    if (frame_count % 30 == 0) {
      std::cout << "================================\n";
      std::cout << "+++ EMOTION DETECTION +++\n";
      float avg_processing_time = frame_time / 30;
      float fps = 1 / (avg_processing_time / 1000);
      std::cout << "[TIMING] Average Frame Processing: " << std::fixed
                << std::setprecision(2) << avg_processing_time << " ms\n";
      std::cout << "[TIMING] FPS: " << std::fixed << std::setprecision(2) << fps
                << " fps\n";
      std::cout << "[TIMING] Total Emotion Loop: " << std::fixed
                << std::setprecision(2) << loop_time.count() * 1000 << "ms\n";
      std::cout << "================================\n";
      frame_time = 0;
      frame_count = 0;
    }
  }
}

std::pair<std::string, float> Emotion::infer(cv::Mat& frame) {
  preprocess(frame);

  auto result = predict(bright_frame);
  int predicted_class = result.first;
  float confidence = result.second;
  std::string& prediction = emotions_[predicted_class];

  return {prediction, confidence};
}

std::pair<int, float> Emotion::predict(cv::Mat& frame) {
  int w = frame.cols;
  int h = frame.rows;

  // Convert to NCNN format and normalize
  ncnn::Mat inmat = ncnn::Mat::from_pixels_resize(
      frame.data, ncnn::Mat::PIXEL_GRAY, w, h, FRAME_SIZE, FRAME_SIZE);
  float mean[1] = {127.5f};
  float norm[1] = {1 / 127.5f};
  inmat.substract_mean_normalize(mean, norm);

  // Run inference
  ncnn::Extractor extractor = emotion.create_extractor();
  extractor.set_light_mode(true);
  extractor.set_num_threads(num_cores - 1);
  extractor.input("in0", inmat);

  ncnn::Mat out1;
  extractor.extract("out0", out1);

  return finalPred(out1);
}

cv::Mat Emotion::roiCrop(float x1, float y1, float x2, float y2,
                         cv::Mat& frame) {
  float width = (x2 - x1);
  float height = (y2 - y1);

  // Clamp to frame boundaries to prevent out-of-bounds access
  int roi_x = std::max(0, (int)(x1));
  int roi_y = std::max(0, (int)(y1));
  int roi_w = std::min((int)width, frame.cols - roi_x);
  int roi_h = std::min((int)height, frame.rows - roi_y);

  cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
  return frame(roi);
}

void Emotion::preprocess(const cv::Mat& frame) {
  // Convert to grayscale and apply brightness enhancement
  cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
  double brightness_value = 25;
  bright_frame = gray_frame + cv::Scalar(brightness_value);
}

void Emotion::softmax(ncnn::Mat& nums) {
  // Find maximum for numerical stability
  float max_val = -FLT_MAX;
  for (int i = 0; i < nums.w; i++) {
    max_val = std::max(max_val, nums[i]);
  }

  // Calculate exponential sum
  float exp_sum = 0;
  for (int i = 0; i < nums.w; i++) {
    exp_sum += std::exp(nums[i] - max_val);
  }

  // Apply softmax normalization
  for (int j = 0; j < nums.w; j++) {
    nums[j] = std::exp(nums[j] - max_val) / exp_sum;
  }
}

std::pair<int, float> Emotion::finalPred(ncnn::Mat& probs) {
  softmax(probs);

  int predicted_class = 0;
  float confidence = -FLT_MAX;

  // Find class with highest confidence
  for (int i = 0; i < probs.w; ++i) {
    float val = probs[i];
    if (val > confidence) {
      confidence = val;
      predicted_class = i;
    }
  }

  return {predicted_class, confidence};
}
