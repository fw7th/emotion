#include "reader.h"

#include <chrono>
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
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // Manual exposure
  cap.set(cv::CAP_PROP_EXPOSURE, -6);         // Fixed exposure value
  cap.set(cv::CAP_PROP_GAIN, 0);              // Fixed gain

  // Force consistent frame timing
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Reduce buffering

  // FPS calculation variables
  int frame_count = 0;
  auto start = std::chrono::steady_clock::now();

  while (cap.isOpened()) {
    auto time1 = std::chrono::steady_clock::now();

    cv::Mat capture;
    cap >> capture;
    if (capture.empty()) break;

    // Downscale for faster processing (640x320 vs original resolution)
    cv::Mat new_cap;
    cv::resize(capture, new_cap, cv::Size(640, 320));

    try {
      // Frame skipping: only process every 3rd frame to reduce load
      if (frame_count % 3 == 0) {
        output_queue.push(std::move(new_cap));
      }

      /*
      if (frame_count % 50 == 0) {
        std::cout << "Frame passed to queue " << new_cap.size << "\n";
      }
      */
    } catch (...) {
      std::cerr << "Error: Exception caught.\n";
      break;
    }

    frame_count++;

    // Calculate and reset FPS counter every second
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elasped_seconds = end - start;
    auto duration_secs =
        std::chrono::duration_cast<std::chrono::seconds>(elasped_seconds);
    int secs = static_cast<int>(duration_secs.count());

    if (secs >= 1) {
      float fps = frame_count / 1.0;
      // std::cout << "[DEBUG] Frame Reader FPS = " << fps << "\n";
      frame_count = 0;
      start = end;
    }

    // Timing debug code (currently disabled)
    auto time2 = std::chrono::steady_clock::now();
    auto timed = std::chrono::duration_cast<std::chrono::duration<double>>(
        time2 - time1);
    /*
    if (frame_count % 200 == 0) {
      std::cout << "[TIMING] Reader Loop = " << timed.count() << "secs. \n";
    }
    */
  }

  cap.release();
}

}  // namespace read
