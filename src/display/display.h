#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "customqueue.h"
#include "structs.h"

class Display {
 private:
  void textBox(cv::Mat &frame, const cv::Point pt, const std::string &text);

 public:
  Display(ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue_);
  ~Display();
  Display(const Display &) = delete;
  Display &operator=(const Display &) = delete;

  ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue;
  void display();
};
