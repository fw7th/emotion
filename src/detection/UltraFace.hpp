//
//  UltraFace.hpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#ifndef UltraFace_hpp
#define UltraFace_hpp

#pragma once

#include "gpu.h"
#include "memory"
#include "net.h"
#include "reader.hpp"
#include <algorithm>
#include <deque>
#include <iostream>
#include <opencv2/core.hpp>
#include <queue>
#include <string>
#include <vector>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms                                                           \
    2 /* mix nms was been proposaled in paper blaze face, aims to minimize the \
         temporal jitter*/

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    float *landmarks;
} FaceInfo;

typedef struct UltraStruct {
    cv::Mat frame;
    std::vector<cv::Mat> crops;

} UltraStruct;

class UltraFace
{
  public:
    UltraFace(read::Reader &obj, const std::string &bin_path,
              const std::string &param_path, int input_width, int input_length,
              int num_thread_ = 4, float score_threshold_ = 0.7,
              float iou_threshold_ = 0.3, int topk_ = -1);

    UltraFace(const UltraFace &) = delete;            // Delete copy constructor
    UltraFace &operator=(const UltraFace &) = delete; // Delete copy assignment
    ~UltraFace();

    int detect(ncnn::Mat &img, std::vector<FaceInfo> &face_list);

    cv::Mat roiCrop(float x1, float y1, float x2, float y2, cv::Mat &frame);

    void infer();

    std::queue<std::unique_ptr<std::vector<UltraStruct>>>
        ultraface_queue; // queue to recieve detections
  private:
    void generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores,
                      ncnn::Mat boxes, float score_threshold, int num_anchors);

    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output,
             int type = blending_nms);

  private:
    read::Reader &shared_obj;
    ncnn::Net ultraface;

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    int topk;
    float score_threshold;
    float iou_threshold;

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
        {10.0f, 16.0f, 24.0f},
        {32.0f, 48.0f},
        {64.0f, 96.0f},
        {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};
};

#endif /* UltraFace_hpp */
