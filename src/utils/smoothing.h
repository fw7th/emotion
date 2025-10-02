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
  float confidence_threshold = 0.6f;

 public:
  std::string stabilize(const std::string& emotion, float confidence) {
    auto now = std::chrono::steady_clock::now();

    // Add detection
    history.push_back({emotion, confidence, now});

    // Remove stale entries
    auto cutoff = now - std::chrono::milliseconds(1500);
    while (!history.empty() && history.front().timestamp < cutoff) {
      history.pop_front();
    }

    std::map<std::string, float> weighted_scores;

    // Weight by confidence × recency
    for (const auto& entry : history) {
      auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - entry.timestamp)
                        .count();
      float time_weight = std::exp(-age_ms / 1000.0f);
      weighted_scores[entry.emotion] += entry.confidence * time_weight;
    }

    // Return max score
    return std::max_element(
               weighted_scores.begin(), weighted_scores.end(),
               [](const auto& a, const auto& b) { return a.second < b.second; })
        ->first;
  }
};

class HysteresisStabilizer {
 private:
  std::string current_stable_emotion = "";
  std::string candidate_emotion = "";
  int candidate_count = 0;
  int confirmation_frames = 3;

 public:
  std::string stabilize(const std::string& detected_emotion) {
    if (current_stable_emotion.empty()) {
      current_stable_emotion = detected_emotion;
      return current_stable_emotion;
    }

    if (detected_emotion == current_stable_emotion) {
      candidate_emotion = "";
      candidate_count = 0;
      return current_stable_emotion;
    }

    if (detected_emotion == candidate_emotion) {
      candidate_count++;
      if (candidate_count >= confirmation_frames) {
        current_stable_emotion = candidate_emotion;
        candidate_emotion = "";
        candidate_count = 0;
      }
    } else {
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
    std::string confidence_result =
        confidence_stabilizer.stabilize(emotion, confidence);
    return hysteresis_stabilizer.stabilize(confidence_result);
  }
};

struct Person {
    int id;
    cv::Rect bbox;
    std::string emotion;
    float confidence;
    int frames_since_seen = 0;
};

class MultiPersonEmotionStabilizer {
private:
    std::map<int, ConfidenceStabilizer> person_stabilizers;
    std::vector<Person> tracked_people;
    int next_person_id = 0;
    int max_frames_missing = 30;
    float iou_threshold = 0.5f;
    
    float calculateIoU(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        
        if (x2 <= x1 || y2 <= y1) return 0.0f;
        
        int intersection = (x2 - x1) * (y2 - y1);
        int union_area = a.area() + b.area() - intersection;
        
        return static_cast<float>(intersection) / union_area;
    }
    
public:
    std::vector<Person> update(const std::vector<cv::Rect>& bboxes, 
                               const std::vector<std::string>& emotions,
                               const std::vector<float>& confidences) {
        
        // Increment missing counters
        for (auto& person : tracked_people) {
            person.frames_since_seen++;
        }
        
        std::vector<bool> detection_matched(bboxes.size(), false);
        
        // Match detections
        for (size_t i = 0; i < bboxes.size(); i++) {
            int best_match_id = -1;
            float best_iou = 0.0f;
            
            for (auto& person : tracked_people) {
                if (person.frames_since_seen < max_frames_missing) {
                    float iou = calculateIoU(bboxes[i], person.bbox);
                    if (iou > iou_threshold && iou > best_iou) {
                        best_iou = iou;
                        best_match_id = person.id;
                    }
                }
            }
            
            if (best_match_id != -1) {
                auto it = std::find_if(tracked_people.begin(), tracked_people.end(),
                    [best_match_id](const Person& p) { return p.id == best_match_id; });
                
                it->bbox = bboxes[i];
                it->emotion = person_stabilizers[best_match_id].stabilize(emotions[i], confidences[i]);
                it->confidence = confidences[i];
                it->frames_since_seen = 0;
                detection_matched[i] = true;
            }
        }
        
        // Add new people
        for (size_t i = 0; i < bboxes.size(); i++) {
            if (!detection_matched[i]) {
                Person new_person;
                new_person.id = next_person_id++;
                new_person.bbox = bboxes[i];
                new_person.emotion = emotions[i];
                new_person.confidence = confidences[i];
                new_person.frames_since_seen = 0;
                
                tracked_people.push_back(new_person);
                person_stabilizers[new_person.id] = ConfidenceStabilizer();
            }
        }
        
        // Remove inactive people
        tracked_people.erase(
            std::remove_if(tracked_people.begin(), tracked_people.end(),
                [this](const Person& p) {
                    if (p.frames_since_seen >= max_frames_missing) {
                        person_stabilizers.erase(p.id);
                        return true;
                    }
                    return false;
                }), 
            tracked_people.end());
        
        return tracked_people;
    }
};
