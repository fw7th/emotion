#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

// Thread-safe queue template
namespace ts {
template <typename T> class TSQueue {
private:
  std::queue<T> _queue;
  mutable std::mutex _mtx;
  std::condition_variable _cv;

public:
  TSQueue() = default;

  TSQueue(const TSQueue &) = delete;
  TSQueue &operator=(const TSQueue &) = delete;

  void push(T &item) {
    std::lock_guard<std::mutex> lock(_mtx);
    _queue.push(std::move(item));
    _cv.notify_one();
  }

  void push(T &&item) {
    std::lock_guard<std::mutex> lock(_mtx);
    _queue.push(std::move(item));
    _cv.notify_one();
  }

  std::optional<T> pop() {
    std::lock_guard<std::mutex> lock(_mtx);
    if (_queue.empty())
      return std::nullopt;

    T item = std::move(_queue.front());
    _queue.pop();
    return item;
  }

  bool empty() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _queue.empty();
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _queue.size();
  }
};
} // namespace ts
