#include "display.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

Display::Display(ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue_)
    : input_queue(input_queue_) {
  std::cout << "Display Started.\n";
}

Display::~Display() {}

void Display::textBox(cv::Mat &frame, const cv::Point pt,
                      const std::string &text) {
  int x_start = pt.x;
  int y_start = pt.y - 15;

  int x_end = pt.x + 55;
  int y_end = pt.y;

  cv::rectangle(frame, cv::Point(x_start, y_start), cv::Point(x_end, y_end),
                cv::Scalar(150, 255, 0), cv::FILLED);

  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.4;
  int thickness = 1;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

  cv::Point textOrigin(pt.x + 3, pt.y - 4);

  cv::Scalar textColor(255, 255, 255);  // White text
  cv::putText(frame, text, textOrigin, fontFace, fontScale, textColor,
              thickness, cv::LINE_AA);
}

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

    auto emotion_ptr_wrapped = input_queue.pop();

    if (!emotion_ptr_wrapped.has_value()) {
      // std::cerr << "Error: opt_emotion_ptr is empty" << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    auto &emotion_ptr = emotion_ptr_wrapped.value();
    cv::Mat &frame = emotion_ptr->frame;
    const std::vector<Bbox> &boxes = emotion_ptr->bboxes;
    const std::vector<std::string> &emotions = emotion_ptr->predictions;

    for (int i = 0; i < boxes.size(); i++) {
      textBox(frame, boxes[i].pt1, emotions[i]);
      cv::rectangle(frame, boxes[i].pt1, boxes[i].pt2, cv::Scalar(150, 255, 0),
                    1);
    }

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
