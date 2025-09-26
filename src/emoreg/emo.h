#pragma once
#include <algorithm>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "customqueue.h"
#include "net.h"
#include "structs.h"

/**
 * Emotion Detection Module
 * Uses NCNN-based CNN model to classify facial expressions into 7 emotion categories
 * Processes cropped face regions and outputs emotion predictions with confidence scores
 */
class Emotion {
private:
    static constexpr int FRAME_SIZE = 64;
    static constexpr int NUM_EMOTIONS = 7;
    
    // Emotion label mapping
    static std::unordered_map<int, std::string> emotions_;
    
    // NCNN model and processing buffers
    ncnn::Net emotion;
    cv::Mat bright_frame;  // Pre-allocated processing buffer
    cv::Mat gray_frame;    // Pre-allocated grayscale buffer
    int num_cores;

    /**
     * Perform emotion classification on preprocessed face crop
     * @param frame Input face crop (will be resized to 64x64)
     * @return Pair of (predicted_class_index, confidence_score)
     */
    std::pair<int, float> predict(cv::Mat& frame);

    /**
     * Extract final prediction from model output using softmax
     * @param input1 Raw model output logits
     * @return Pair of (predicted_class_index, confidence_score)
     */
    std::pair<int, float> finalPred(ncnn::Mat& input1);

    /**
     * Preprocess face crop: convert to grayscale and apply brightness enhancement
     * @param frame Input BGR face crop
     */
    void preprocess(const cv::Mat& frame);

    /**
     * Apply softmax normalization to model outputs
     * @param nums Input/output NCNN Mat containing logits/probabilities
     */
    void softmax(ncnn::Mat& nums);

    /**
     * Complete inference pipeline for a single face crop
     * @param frame Input face crop
     * @return Pair of (emotion_label, confidence_score)
     */
    std::pair<std::string, float> infer(cv::Mat& frame);

    /**
     * Extract face region from full frame using bounding box coordinates
     * @param x1,y1,x2,y2 Bounding box coordinates
     * @param frame Source frame
     * @return Cropped face region with padding
     */
    cv::Mat roiCrop(float x1, float y1, float x2, float y2, cv::Mat& frame);

public:
    /**
     * Constructor - initializes emotion detection model
     * @param input_queue_ Queue to receive frames with face detections
     * @param output_queue_ Queue to output frames with emotion predictions
     * @param bin_path_ Path to NCNN model binary file
     * @param param_path_ Path to NCNN model parameter file
     */
    Emotion(ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue_,
            ts::TSQueue<std::unique_ptr<FrameInfo>>& output_queue_,
            const std::string& bin_path_, const std::string& param_path_);

    // Disable copying
    Emotion(const Emotion&) = delete;
    Emotion& operator=(const Emotion&) = delete;

    ~Emotion();

    // Public member variables
    const std::string& bin_path;
    const std::string& param_path;
    ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue;
    ts::TSQueue<std::unique_ptr<FrameInfo>>& output_queue;

    /**
     * Main processing loop - continuously processes frames with face detections
     * Performs emotion classification on each detected face and adds results to frame data
     */
    void load();
};
