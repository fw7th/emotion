#pragma once

#include <algorithm>
#include <chrono>
#include <deque>
#include <map>
#include <string>

class ConfidenceStabilizer {
 private:
  struct EmotionScore {
    std::string emotion;
    float confidence;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::deque<EmotionScore> history;
  float confidence_threshold = 0.7f;

 public:
  std::string stabilize(const std::string& emotion, float confidence) {
    auto now = std::chrono::steady_clock::now();

    // Add current detection
    history.push_back({emotion, confidence, now});

    // Remove old entries (e.g., older than 2 seconds)
    auto cutoff = now - std::chrono::milliseconds(2000);
    while (!history.empty() && history.front().timestamp < cutoff) {
      history.pop_front();
    }

    // Weight by confidence and recency
    std::map<std::string, float> weighted_scores;

    // Recent emotions with high confidence get highest scores
    for (const auto& entry : history) {
      auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - entry.timestamp)
                        .count();
      float time_weight = std::exp(-age_ms / 1000.0f);  // Exponential decay

      weighted_scores[entry.emotion] += entry.confidence * time_weight;
    }

    // Return highest weighted emotion
    return std::max_element(
               weighted_scores.begin(), weighted_scores.end(),
               [](const auto& a, const auto& b) { return a.second < b.second; })
        ->first;
  }
};

class HysteresisStabilizer {
  // Stabilizes detected emotion to suppress fast switching
 private:
  std::string current_stable_emotion = "";
  std::string candidate_emotion = "";
  int candidate_count = 0;
  int confirmation_frames = 5;  // Need 5 consistent frames to switch

 public:
  std::string stabilize(const std::string& detected_emotion) {
    if (current_stable_emotion.empty()) {
      current_stable_emotion = detected_emotion;
      return current_stable_emotion;
    }

    if (detected_emotion == current_stable_emotion) {
      // Reset candidate if we're back to current emotion
      candidate_emotion = "";
      candidate_count = 0;
      return current_stable_emotion;
    }

    if (detected_emotion == candidate_emotion) {
      candidate_count++;
      if (candidate_count >= confirmation_frames) {
        // Switch to new emotion
        current_stable_emotion = candidate_emotion;
        candidate_emotion = "";
        candidate_count = 0;
      }
    } else {
      // New candidate
      candidate_emotion = detected_emotion;
      candidate_count = 1;
    }

    return current_stable_emotion;
  }
};

class RobustEmotionStabilizer {
 private:
  ConfidenceStabilizer confidence_stabilizer;
  HysteresisStabilizer hysteresis_stabilizer;

 public:
  std::string stabilize(const std::string& emotion, float confidence) {
    // First apply confidence-based smoothing
    std::string confidence_result =
        confidence_stabilizer.stabilize(emotion, confidence);

    // Then apply hysteresis to prevent rapid switching
    return hysteresis_stabilizer.stabilize(confidence_result);
  }
};
