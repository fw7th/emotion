#pragma once

#include "customqueue.h"
#include "structs.h"

class Display {
 public:
  Display(ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue_);
  Display(const Display &) = delete;
  Display &operator=(const Display &) = delete;
  ~Display();

  ts::TSQueue<std::unique_ptr<UltraStruct>> &input_queue;
  void display();
};
