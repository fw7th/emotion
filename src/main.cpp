#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <variant>

#include "customqueue.h"
#include "emo.h"
#include "reader.h"
#include "tracker.h"
#include "ultraface.h"

int main() {
  std::string input;

  std::string bin_ultra = "../../data/version-slim/slim_320.bin";
  std::string param_ultra = "../../data/version-slim/slim_320.param";

  std::string bin_emo = "../../data/detector/opt.bin";
  std::string param_emo = "../../data/detector/opt.param";

  std::cout << "Enter input source: ";
  std::cin >> input;

  std::variant<int, std::string> source;

  // Load bin and param files
  std::ifstream ultra_bin_file(bin_ultra, std::ios::binary);
  std::ifstream emote_bin_file(bin_emo, std::ios::binary);
  if (!ultra_bin_file) {
    std::cerr << "Can't open ultra bin file\n";
    return 1;
  }

  if (!emote_bin_file) {
    std::cerr << "Can't open emote bin file\n";
    return 1;
  }

  std::ifstream ultra_param_file(param_ultra);
  std::ifstream emote_param_file(param_emo);
  if (!ultra_param_file) {
    std::cerr << "Can't open ultra param file\n";
    return 1;
  }

  if (!emote_param_file) {
    std::cerr << "Can't open emote param file\n";
    return 1;
  }

  // Try to parse as int (webcam ID)
  try {
    source = std::stoi(input);
  } catch (...) {
    source = input;
  }

  ts::TSQueue<cv::Mat> reader_queue;
  ts::TSQueue<std::unique_ptr<UltraStruct>> detect_queue;
  ts::TSQueue<std::unique_ptr<UltraStruct>> tracker_queue;

  read::Reader reader(reader_queue);
  reader.setSource(source);

  UltraFace ultraface(reader_queue, detect_queue, bin_ultra, param_ultra, 64,
                      64, 1,
                      0.7); // config model input

  Tracker tracks(detect_queue, tracker_queue);

  Emotion emote(tracker_queue, bin_emo, param_emo);

  std::thread t1([&]() { reader.read_frames(); });
  std::thread t2([&]() { ultraface.infer(); });
  std::thread t6([&]() { tracks.track(); });
  std::thread t9([&]() { emote.load(); });

  t1.join();
  t2.join();
  t6.join();
  t9.join();

  return 0;
}
