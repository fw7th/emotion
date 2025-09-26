#pragma once

#include "customqueue.h"
#include "structs.h"

class Display {
 public:
  Display(ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue_);
  ~Display();
  Display(const Display &) = delete;
  Display &operator=(const Display &) = delete;

  ts::TSQueue<std::unique_ptr<FrameInfo>> &input_queue;
  void display();
};
