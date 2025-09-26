#include "reader.h"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <utility>

Reader::Reader(ts::TSQueue<std::unique_ptr<FrameInfo>>& output_queue_)
    : output_queue(output_queue_) {}

void Reader::setSource(std::variant<int, std::string> s) {
  source = std::move(s);
}

void Reader::read_frames() {
  // Validate source configuration
  if (std::holds_alternative<std::string>(source) &&
      std::get<std::string>(source).empty()) {
    std::cerr << "Error: Source not configured\n";
    return;
  }

  cv::VideoCapture cap;

  // Open video source (camera or file)
  std::visit([&cap](auto&& value) { cap.open(value); }, source);

  if (!cap.isOpened()) {
    std::cerr << "Error: Failed to open video source\n";
    return;
  }

  // Configure capture settings
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

  int frame_count = 0;

  while (cap.isOpened()) {
    cv::Mat capture;
    cap >> capture;

    if (capture.empty()) break;

    try {
      // Process every 3rd frame to reduce computational load
      if (frame_count % 3 == 0) {
        auto frame_info = std::make_unique<FrameInfo>();
        frame_info->frame = capture;
        output_queue.push(std::move(frame_info));
      }
      frame_count++;
    } catch (...) {
      std::cerr << "Error: Exception during frame processing\n";
      break;
    }
  }

  cap.release();
}
