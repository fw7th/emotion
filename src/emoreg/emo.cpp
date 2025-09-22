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

Emotion::Emotion(ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue_,
                 const std::string &bin_path_, const std::string &param_path_)
    : input_queue(input_queue_),
      bin_path(bin_path_),
      param_path(param_path_)

{
  std::cout << "Pre-allocated all buffers!" << std::endl;
  // Pre-allocate OpenCV processing buffers
  gray_frame.create(FRAME_SIZE, FRAME_SIZE, CV_8UC1);
  bright_frame.create(FRAME_SIZE, FRAME_SIZE, CV_8UC1);

  emotion.opt.use_vulkan_compute = false;  // CPU inference
  emotion.opt.num_threads = 4;
  emotion.opt.use_fp16_packed = true;
  emotion.opt.use_fp16_storage = true;
  emotion.opt.use_fp16_arithmetic = false;
  emotion.opt.use_int8_storage = true;
  emotion.opt.use_int8_arithmetic = false;

  emotion.load_param(param_path.data());
  emotion.load_model(bin_path.data());
}

Emotion::~Emotion() { emotion.clear(); }

std::unordered_map<int, std::string> Emotion::emotions_ = {
    {0, "Angry"},   {1, "Disgust"}, {2, "Fear"},    {3, "Happy"},
    {4, "Neutral"}, {5, "Sad"},     {6, "Surprise"}};

void Emotion::load() {
  int frame_count = 0;
  auto fps_timer_start = std::chrono::steady_clock::now();

  cv::namedWindow("Emotion Detection", cv::WINDOW_NORMAL);
  cv::resizeWindow("Emotion Detection", 480, 320);

  while (true) {
    auto loop_start = std::chrono::steady_clock::now();

    // Check queue and get frame
    if (input_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto opt_track_ptr = std::move(input_queue.pop());

    if (!opt_track_ptr.has_value()) {
      // std::cerr << "Error: opt_track_ptr is empty" << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    cv::Mat &frame = opt_track_ptr.value()->frame;

    infer(frame);

    if (frame.empty()) {
      // std::cout << "Emotion frame is empty.\n";
    }

    cv::imshow("Emotion Detection", frame);
    cv::waitKey(1);

    // Calculate total loop time
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

    if (frame_count % 30 == 0) {
      /*
      std::cout << "=== EMOTION " << frame_count << " TIMING ===\n";
      std::cout << "[TIMING] Frame Pointer Access:     " << std::fixed
                << std::setprecision(2) << frame_access_time.count() * 1000
                << " ms\n";
      std::cout << "[TIMING] Emotion Detection: " << std::fixed
                << std::setprecision(2) << detect_time.count() * 1000
                << " ms\n";
      std::cout << "[TIMING] Total Emotion Loop:     " << std::fixed
                << std::setprecision(2) << loop_time.count() * 1000 << " ms\n";
      std::cout << "================================\n";
      */
    }

    // FPS calculation
    frame_count++;
    auto fps_timer_end = std::chrono::steady_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            fps_timer_end - fps_timer_start);

    if (elapsed_seconds.count() >= 1.0) {
      double fps = frame_count / elapsed_seconds.count();
      /*
      std::cout << "[DEBUG] Emotion Detector FPS = " << std::fixed
                << std::setprecision(1) << fps << "\n";
      */
      frame_count = 0;
      fps_timer_start = fps_timer_end;
    }
  }
  cv::destroyAllWindows();
}

void Emotion::infer(cv::Mat &frame) {
  static int loop_count = 0;

  auto final_start = std::chrono::steady_clock::now();

  preprocess(frame);

  auto predict_start = std::chrono::steady_clock::now();

  int emotion_idx = predict(bright_frame);

  auto predict_end = std::chrono::steady_clock::now();
  auto predict_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      predict_end - predict_start);

  std::string emotion_text = emotions_[emotion_idx];

  cv::putText(frame, emotion_text, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN,
              1.0, cv::Scalar(0, 0, 200), 3);

  auto final_end = std::chrono::steady_clock::now();
  auto final_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      final_end - final_start);

  if (loop_count % 30 == 0) {
    /*
    std::cout << "+++ INFER METHOD +++\n";
    std::cout << "[INFERENCE] Detected emotion: " << emotion_text << ".\n";

    std::cout << "[TIMING] Prediction Function Time: " << std::fixed
              << std::setprecision(2) << predict_time.count() * 1000 << " ms\n";

    std::cout << "[TIMING] Full Prediction Time:     " << std::fixed
              << std::setprecision(2) << final_time.count() * 1000 << " ms\n";

    std::cout << "================================\n";
    */
  }
  loop_count++;
}

int Emotion::predict(cv::Mat &frame1) {
  int _predict_count = 0;
  int w = frame1.cols;
  int h = frame1.rows;

  ncnn::Mat inmat = ncnn::Mat::from_pixels_resize(
      frame1.data, ncnn::Mat::PIXEL_GRAY, w, h, 96, 96);
  float mean[1] = {127.5f};
  float norm[1] = {1 / 127.5f};
  inmat.substract_mean_normalize(mean, norm);

  ncnn::Extractor extractor = emotion.create_extractor();
  extractor.set_light_mode(true);
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
    std::cout << "+++ PREDICT METHOD +++\n";
    std::cout << "[TIMING] Extraction Out Time: " << std::fixed
              << std::setprecision(2) << extract_out_time.count() * 1000
              << " ms\n";

    std::cout << "================================\n";
    */
  }

  int emotion_idx = finalPred(out1);
  _predict_count++;

  return emotion_idx;
}

void Emotion::preprocess(const cv::Mat &frame) {
  // Turn img to grayscale
  cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

  // Brighten frame
  double brightness_value = 60;
  bright_frame = gray_frame + cv::Scalar(brightness_value);
}

int Emotion::maxIndex(ncnn::Mat &probs) {
  float max_val = -FLT_MAX;
  int max_index = 0;

  for (int i = 0; i < probs.w; ++i) {
    float val = probs[i];
    if (val > max_val) {
      max_val = val;
      max_index = i;
    }
  }
  return max_index;
}

void Emotion::softmax(ncnn::Mat &nums) {
  float max_val = -FLT_MAX;
  for (int i = 0; i < nums.w; i++) max_val = std::max(max_val, nums[i]);

  float exp_sum = 0;
  for (int i = 0; i < nums.w; i++) exp_sum += std::exp(nums[i] - max_val);

  for (int i = 0; i < nums.w; i++)
    nums[i] = std::exp(nums[i] - max_val) / exp_sum;
}

int Emotion::finalPred(ncnn::Mat &input) {
  softmax(input);
  int emotion_idx = maxIndex(input);

  return emotion_idx;
}
