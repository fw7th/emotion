#include "tracker.h"
#include "funcs.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <utility>

Tracker::Tracker(ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue_,
                 ts::TSQueue<std::unique_ptr<UltraStruct>> &output_queue_,
                 float track_thresh_, int track_buffer_, float match_thresh_,
                 int frame_rate_)
    : input_queue(input_queue_), output_queue(output_queue_) {
  track_thresh = track_thresh_;
  track_buffer = track_buffer_;
  match_thresh = match_thresh_;
  frame_rate = frame_rate_;
}

void Tracker::track() {
  int frame_count = 0;
  auto fps_timer_start = std::chrono::steady_clock::now();

  while (true) {
    auto loop_start = std::chrono::steady_clock::now();

    if (input_queue.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      /*
      if (frame_count % 50 == 0) {
         std::cout << "Detection queue is empty\n";
      }
      */
    }

    auto frame_access_start = std::chrono::steady_clock::now();
    auto opt_ultra_ptr = std::move(input_queue.pop());
    // print_type(opt_ultra_ptr);

    if (!opt_ultra_ptr.has_value()) {
      // std::cerr << "Error: opt_ultra_ptr is empty" << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    auto &ultra_ptr = opt_ultra_ptr.value();

    auto frame_access_end = std::chrono::steady_clock::now();
    auto tracker_access_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            frame_access_end - frame_access_start);

    // Do Tracker Work

    output_queue.push(std::move(ultra_ptr));

    // Calculate total loop time
    auto loop_end = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        loop_end - loop_start);

    if (frame_count % 20 == 0) {

      /*
      std::cout << "=== TRACKER " << frame_count << " TIMING ===\n";
      std::cout << "[TIMING] Tracker Pointer Access:     " << std::fixed
                << std::setprecision(2) << tracker_access_time.count() * 1000
                << " ms\n";
      std::cout << "[TIMING] Emotion Detection: " << std::fixed
                << std::setprecision(2) << detect_time.count() * 1000
                << " ms\n";
      std::cout << "[TIMING] Total Tracker Loop:     " << std::fixed
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

  }
}
