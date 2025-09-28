//
//  UltraFace.hpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

// Portions adapted from Ultra-Light-Fast-Generic-Face-Detector-1MB by Linzaer (MIT License).

#ifndef ULTRAFACE_H
#define ULTRAFACE_H

#pragma once
#include <algorithm>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "customqueue.h"
#include "gpu.h"
#include "net.h"
#include "structs.h"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2

/**
 * Face detection information structure
 */
struct FaceInfo {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float score;           // Detection confidence
    float* landmarks;      // Facial landmarks (unused)
};

/**
 * UltraFace: Lightweight face detection using NCNN inference engine
 * Processes frames from input queue and outputs detected faces to output queue
 */
class UltraFace {
public:
    /**
     * Constructor - initializes the face detection model
     * @param input_queue_ Queue to receive frames for processing
     * @param output_queue_ Queue to output processed frames with detections
     * @param bin_path Path to NCNN model binary file
     * @param param_path Path to NCNN model parameter file
     * @param input_width Model input width
     * @param input_length Model input height
     * @param score_threshold_ Minimum confidence threshold for detections
     * @param iou_threshold_ IoU threshold for NMS
     * @param topk_ Maximum number of detections to keep (-1 for unlimited)
     */
    UltraFace(ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue_,
              ts::TSQueue<std::unique_ptr<FrameInfo>>& output_queue_,
              const std::string& bin_path, const std::string& param_path,
              int input_width, int input_length, 
              float score_threshold_ = 0.7, float iou_threshold_ = 0.3, 
              int topk_ = -1);

    // Disable copying
    UltraFace(const UltraFace&) = delete;
    UltraFace& operator=(const UltraFace&) = delete;

    ~UltraFace();

    /**
     * Main inference loop - processes frames continuously from input queue
     */
    void infer();

    // Public member variables
    const std::string& bin_path;
    const std::string& param_path;
    ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue;
    ts::TSQueue<std::unique_ptr<FrameInfo>>& output_queue;

private:
    /**
     * Perform face detection on a single image
     * @param img Input image in NCNN Mat format
     * @param face_list Output vector of detected faces
     * @return 0 on success, -1 on error
     */
    int detect(ncnn::Mat& img, std::vector<FaceInfo>& face_list);

    /**
     * Generate bounding boxes from network output
     */
    void generateBBox(std::vector<FaceInfo>& bbox_collection, ncnn::Mat scores,
                      ncnn::Mat boxes, float score_threshold, int num_anchors);

    /**
     * Apply Non-Maximum Suppression to filter overlapping detections
     */
    void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
             int type = blending_nms);

private:
    // NCNN model and configuration
    ncnn::Net ultraface;
    ncnn::Option opt;
    
    // Model parameters
    int image_w, image_h;      // Current image dimensions
    int in_w, in_h;           // Model input dimensions
    int num_anchors;          // Total number of anchor boxes
    int num_cores;            // Available CPU cores
    int topk;                 // Maximum detections to keep
    float score_threshold;    // Confidence threshold
    float iou_threshold;      // NMS IoU threshold

    // Model preprocessing constants
    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0f/128, 1.0f/128, 1.0f/128};
    const float center_variance = 0.1f;
    const float size_variance = 0.2f;

    // Anchor box configuration
    const std::vector<std::vector<float>> min_boxes = {
        {10.0f, 16.0f, 24.0f},
        {32.0f, 48.0f},
        {64.0f, 96.0f},
        {128.0f, 192.0f, 256.0f}
    };
    const std::vector<float> strides = {8.0f, 16.0f, 32.0f, 64.0f};

    // Runtime computed values
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;
    std::vector<std::vector<float>> priors;
};

#endif /* ULTRAFACE_H */
