#include "reader.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <utility>

namespace read {

Reader::Reader(ts::TSQueue<cv::Mat> &output_queue_)
    : output_queue(output_queue_) {};

void Reader::setSource(std::variant<int, std::string> s) {
  source = std::move(s);
}

// Reads frames from video source, downscales for processing, and queues every
// 3rd frame
void Reader::read_frames() {
  // Validate source is configured
  if (std::holds_alternative<std::string>(source) &&
      std::get<std::string>(source).empty()) {
    std::cerr << "Source not set, cannot read frames\n ";
    return;
  }

  cv::VideoCapture cap;

  // Open source (handles both camera index and file path via variant)
  std::visit([&cap](auto &&value) { cap.open(value); }, source);

  if (!cap.isOpened()) {
    std::cerr << "Source could not be read. \n";
    return;
  }

  // Lock camera settings to prevent auto-adjustments
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);  // Manual exposure
  cap.set(cv::CAP_PROP_EXPOSURE, -7);      // Fixed exposure value
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

  // FPS calculation variables
  int frame_count = 0;
  float frame_time = 0;

  while (cap.isOpened()) {
    auto loop_start = std::chrono::steady_clock::now();
    cv::Mat capture;
    cap >> capture;
    if (capture.empty()) break;

    try {
      // Frame skipping: only process every 3rd frame to reduce load
      if (frame_count % 3 == 0) {
        output_queue.push(std::move(capture));
      }
    } catch (...) {
      std::cerr << "Error: Exception caught.\n";
      break;
    }

    // Calculate total loop time
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

    // FPS Calculation
    frame_time += loop_time.count() * 1000;
    frame_count++;

    if (frame_count % 90 == 0) {
      std::cout << "================================\n";
      std::cout << "+++ READER +++\n";
      float avg_processing_time = frame_time / 90;
      float fps = 1 / (avg_processing_time / 1000);
      std::cout << "[TIMING] Average Frame Loading: " << std::fixed
                << std::setprecision(2) << avg_processing_time << " ms\n";
      std::cout << "[TIMING] Frame Reader FPS: " << std::fixed
                << std::setprecision(2) << fps << "fps\n";
      std::cout << "================================\n";
      frame_time = 0;
      frame_count = 0;
    }
  }
  cap.release();
}

}  // namespace read
