#pragma once
#include "customqueue.h"
#include <opencv2/core.hpp>
#include <string>
#include <variant>

#include "structs.h"


/**
 * Video frame reader that captures from camera or file and feeds frames to a queue.
 * Handles frame resizing and skipping for performance optimization.
 */
class Reader {
private:
  std::variant<int, std::string> source; // Camera index (int) or file path (string)

public:
  /**
   * @param output_queue Thread-safe queue to receive processed frames
   */
  Reader(ts::TSQueue<std::unique_ptr<FrameInfo>> &output_queue_);
  
  ts::TSQueue<std::unique_ptr<FrameInfo>> &output_queue;

  /**
   * Set video source for capture
   * @param s Camera index (e.g., 0 for default camera) or file path
   */
  void setSource(std::variant<int, std::string> s);

  /**
   * Start frame capture loop. Blocks until source ends or error occurs.
   * Resizes frames to 640x320 and pushes every 3rd frame to queue.
   */
  void read_frames();
};

