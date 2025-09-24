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
#include "smoothing.h"

Emotion::Emotion(ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue_,
                 ts::TSQueue<std::unique_ptr<UltraStruct>> &output_queue_,
                 const std::string &bin_path_, const std::string &param_path_)
    : input_queue(input_queue_),
      output_queue(output_queue_),
      bin_path(bin_path_),
      param_path(param_path_)

{
  num_cores = std::thread::hardware_concurrency();

  std::cout << "Pre-allocated all buffers!" << std::endl;

  // Pre-allocate OpenCV processing buffers
  gray_frame.create(FRAME_SIZE, FRAME_SIZE, CV_8UC1);
  bright_frame.create(FRAME_SIZE, FRAME_SIZE, CV_8UC1);

  emotion.load_param(param_path.data());
  emotion.load_model(bin_path.data());

  emotion.opt.use_vulkan_compute = false;  // CPU inference
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

  cv::namedWindow("Emotion Detection", cv::WINDOW_NORMAL);
  cv::resizeWindow("Emotion Detection", 480, 320);

  while (true) {
    auto loop_start = std::chrono::steady_clock::now();

    // Check queue and get frame
    if (input_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto opt_ultra_ptr = input_queue.pop();

    if (!opt_ultra_ptr.has_value()) {
      // std::cerr << "Error: opt_ultra_ptr is empty" << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    cv::Mat img = opt_ultra_ptr.value()->frame;

    preprocess(img);

    auto predict_start = std::chrono::steady_clock::now();

    // Unpack predictions and confidence
    auto result = predict(bright_frame);
    int predicted_class = result.first;
    float confidence = result.second;

    auto predict_end = std::chrono::steady_clock::now();
    auto predict_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            predict_end - predict_start);

    std::string prediction = emotions_[predicted_class];

    cv::Mat &frame = opt_ultra_ptr.value()->frame;
    cv::putText(frame, prediction, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN,
                1.0, cv::Scalar(100, 150, 20), 3);

    output_queue.push(opt_ultra_ptr.value());

    // Calculate total loop time
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

    // FPS calculation
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
      std::cout << "[SANITY] Predicted Class: " << prediction << "\n";
      std::cout << "[SANITY] Prediction Confidence: " << confidence << "\n";
      std::cout << "[TIMING] Prediction Function Time: " << std::fixed
                << std::setprecision(2) << predict_time.count() * 1000
                << " ms\n";
      std::cout << "================================\n";
      frame_time = 0;
      frame_count = 0;
    }
  }
  cv::destroyAllWindows();
}

void Emotion::infer(cv::Mat &frame) {
  static int loop_count = 0;

  auto final_start = std::chrono::steady_clock::now();

  preprocess(frame);

  auto predict_start = std::chrono::steady_clock::now();

  // Unpack predictions and confidence
  auto result = predict(bright_frame);
  int predicted_class = result.first;
  float confidence = result.second;

  auto predict_end = std::chrono::steady_clock::now();
  auto predict_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      predict_end - predict_start);

  std::string prediction = emotions_[predicted_class];

  cv::putText(frame, prediction, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(100, 150, 20), 3);

  auto final_end = std::chrono::steady_clock::now();
  auto final_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      final_end - final_start);

  if (loop_count % 30 == 0) {
    std::cout << "+++ Infer Method +++\n";
    std::cout << "[SANITY] Predicted Class: " << prediction << "\n";
    std::cout << "[SANITY] Prediction Confidence: " << confidence << "\n";
    std::cout << "[TIMING] Prediction Function Time: " << std::fixed
              << std::setprecision(2) << predict_time.count() * 1000 << " ms\n";

    std::cout << "[TIMING] Full Prediction Time: " << std::fixed
              << std::setprecision(2) << final_time.count() * 1000 << " ms\n";
    std::cout << "================================\n";
  }
  loop_count++;
}

std::pair<int, float> Emotion::predict(cv::Mat &frame) {
  int _predict_count = 0;
  int w = frame.cols;
  int h = frame.rows;

  ncnn::Mat inmat = ncnn::Mat::from_pixels_resize(
      frame.data, ncnn::Mat::PIXEL_GRAY, w, h, 96, 96);
  float mean[1] = {127.5f};
  float norm[1] = {1 / 127.5f};
  inmat.substract_mean_normalize(mean, norm);

  ncnn::Extractor extractor = emotion.create_extractor();
  extractor.set_light_mode(true);
  extractor.set_num_threads(num_cores - 1);
  extractor.input("in0", inmat);

  auto extract_out_start = std::chrono::steady_clock::now();
  ncnn::Mat out1;
  extractor.extract("out0", out1);

  auto extract_out_end = std::chrono::steady_clock::now();
  auto extract_out_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          extract_out_end - extract_out_start);

  if (_predict_count % 30 == 0) {
    /*
    std::cout << "+++ Predict Method +++\n";
    std::cout << "[TIMING] Extraction Out Time: " << std::fixed
              << std::setprecision(2) << extract_out_time.count() * 1000
              << " ms\n";

    std::cout << "================================\n";
    */
  }

  auto result = finalPred(out1);
  _predict_count++;

  return result;
}

void Emotion::preprocess(const cv::Mat &frame) {
  // Turn img to grayscale
  cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

  // Brighten frame
  double brightness_value = 50;
  bright_frame = gray_frame + cv::Scalar(brightness_value);
}

void Emotion::softmax(ncnn::Mat &nums) {
  // Check the size of the input
  //std::cout << "Input vector size: " << nums.w << std::endl;

  // Print the raw input values before softmax
  /*
  std::cout << "Raw input values: ";

  for (int i = 0; i < nums.w; i++) {
    std::cout << nums[i] << " ";
  }
  std::cout << std::endl;
  */
  float confidence_scores = -FLT_MAX;
  for (int i = 0; i < nums.w; i++)
    confidence_scores = std::max(confidence_scores, nums[i]);

  float exp_sum = 0;
  for (int i = 0; i < nums.w; i++)
    exp_sum += std::exp(nums[i] - confidence_scores);

  for (int j = 0; j < nums.w; j++) {
    nums[j] = std::exp(nums[j] - confidence_scores) / exp_sum;
  }
  // Print the output values after softmax
  /*
  std::cout << "Softmax output values: ";
  for (int i = 0; i < nums.w; i++) {
    std::cout << nums[i] << " ";
  }
  std::cout << std::endl;
  */
}

std::pair<int, float> Emotion::finalPred(ncnn::Mat &probs) {
  softmax(probs);
  int predicted_class = 1;
  float confidence = -FLT_MAX;

  for (int i = 0; i < probs.w; ++i) {
    float val = probs[i];
    if (val > confidence) {
      confidence = val;
      predicted_class = i;
    }
  }
  return {predicted_class, confidence};
}
