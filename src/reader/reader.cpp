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

void Reader::read_frames() {

  // Check if source is set
  if (std::holds_alternative<std::string>(source) &&
      std::get<std::string>(source).empty()) {
    std::cerr << "Source not set, cannot read frames\n ";
    return;
  }

  cv::VideoCapture cap;

  // resolve variant
  std::visit([&cap](auto &&value) { cap.open(value); }, source);

  if (!cap.isOpened()) {
    std::cerr << "Source could not be read. \n";
    return;
  }

  int frame_count = 0;
  auto start = std::chrono::steady_clock::now();

  while (cap.isOpened()) {
    auto time1 = std::chrono::steady_clock::now();
    cv::Mat capture;
    cap >> capture;

    if (capture.empty())
      break;

    cv::Mat new_cap;
    cv::resize(capture, new_cap, cv::Size(640, 320));

    try {
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

} // namespace read
