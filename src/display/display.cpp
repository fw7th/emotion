#include "display.h"
#include "smoothing.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

Display::Display(ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue_)
    : input_queue(input_queue_) {
  std::cout << "Display module initialized\n";
}

Display::~Display() { cv::destroyAllWindows(); }

void Display::textBox(cv::Mat &frame, const cv::Point pt,
                      const std::string &text) {
  // Calculate text dimensions for proper box sizing
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.4;
  int thickness = 1;
  int baseline = 0;

  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

  // Create background box with padding
  int padding = 3;
  int x_start = pt.x;
  int y_start = pt.y - textSize.height - padding * 2;
  int x_end = pt.x + textSize.width + padding * 2;
  int y_end = pt.y;

  // Draw background rectangle
  cv::rectangle(frame, cv::Point(x_start, y_start), cv::Point(x_end, y_end),
                cv::Scalar(150, 255, 0), cv::FILLED);

  // Draw text
  cv::Point textOrigin(pt.x + padding, pt.y - padding);
  cv::putText(frame, text, textOrigin, fontFace, fontScale,
              cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

void Display::display() {
  int frame_count = 0;
  float frame_time = 0;
  RobustEmotionStabilizer stabilizer; // define prediction smoother

  // Create display window
  cv::namedWindow("Display", cv::WINDOW_NORMAL);
  cv::resizeWindow("Display", 480, 320);

  while (true) {
    auto loop_start = std::chrono::steady_clock::now();

    // Get frame from input queue
    if (input_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto emotion_ptr_wrapped = input_queue.pop();
    if (!emotion_ptr_wrapped.has_value()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    auto &emotion_ptr = emotion_ptr_wrapped.value();
    cv::Mat &frame = emotion_ptr->frame;

    if (frame.empty()) {
      continue;
    }

    // Draw bounding boxes and emotion labels
    const std::vector<Bbox> &boxes = emotion_ptr->bboxes;
    const std::vector<std::string> &emotions = emotion_ptr->predictions;
    const std::vector<float> &conf = emotion_ptr->confidences;

    for (size_t i = 0; i < boxes.size() && i < emotions.size(); i++) {
      std::string emotion = stabilizer.stabilize(emotions[i], conf[i]);

      // Draw emotion label with background
      textBox(frame, boxes[i].pt1, emotion);

      // Draw bounding box
      cv::rectangle(frame, boxes[i].pt1, boxes[i].pt2, cv::Scalar(150, 255, 0),
                    1);
    }

    // Performance monitoring
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

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

    // Display frame
    auto now = std::chrono::steady_clock::now();
    static auto last_display = now;
    int display_interval_ms = 1000 / 100; // cap at 100 fps

    if (std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                              last_display)
            .count() >= display_interval_ms) {
      cv::imshow("Display", frame);
      last_display = now;
    }
    cv::waitKey(1);             // keep window responsive
    if (cv::waitKey(1) == 27) { // ESC key to exit
      break;
    }
  }

  cv::destroyAllWindows();
}
