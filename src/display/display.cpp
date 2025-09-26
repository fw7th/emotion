#include "display.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

Display::Display(ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue_)
    : input_queue(input_queue_) {
  std::cout << "Display Started.\n";
}

Display::~Display() {}

void Display::display() {
  int frame_count = 0;
  float frame_time = 0;

  cv::namedWindow("Display", cv::WINDOW_NORMAL);
  cv::resizeWindow("Display", 480, 320);

  while (true) {
    auto loop_start = std::chrono::steady_clock::now();

    // Check queue and get frame
    if (input_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto opt_emotion_ptr = input_queue.pop();

    if (!opt_emotion_ptr.has_value()) {
      // std::cerr << "Error: opt_emotion_ptr is empty" << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    cv::Mat &frame = opt_emotion_ptr.value()->frame;

    if (frame.empty()) {
      // std::cout << "Emotion frame is empty.\n";
    }

    cv::imshow("Display", frame);
    cv::waitKey(1);

    // Calculate total loop time
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

    // FPS calculation
    frame_time += loop_time.count() * 1000;
    frame_count++;

    if (frame_count % 30 == 0) {
      std::cout << "================================\n";
      std::cout << "+++ DISPLAY +++\n";
      float avg_processing_time = frame_time / 30;
      float fps = 1 / (avg_processing_time / 1000);
      std::cout << "[TIMING] Average Frame Processing: " << std::fixed
                << std::setprecision(2) << avg_processing_time << " ms\n";
      std::cout << "[TIMING] FPS: " << std::fixed << std::setprecision(2) << fps
                << " fps\n";
      std::cout << "[TIMING] Total Display Loop: " << std::fixed
                << std::setprecision(2) << loop_time.count() * 1000 << "ms\n";
      std::cout << "================================\n";
      frame_time = 0;
      frame_count = 0;
    }
  }
  cv::destroyAllWindows();
}
