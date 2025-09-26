#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "customqueue.h"
#include "structs.h"

/**
 * Display Module
 * Handles visualization of emotion detection results by drawing bounding boxes 
 * and emotion labels on video frames in real-time
 */
class Display {
private:
    /**
     * Draw emotion label with background box above bounding box
     * @param frame Target frame to draw on
     * @param pt Top-left corner of bounding box
     * @param text Emotion label text to display
     */
    void textBox(cv::Mat& frame, const cv::Point pt, const std::string& text);

public:
    /**
     * Constructor - initializes display module
     * @param input_queue_ Queue to receive processed frames with emotion data
     */
    explicit Display(ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue_);

    // Disable copying
    Display(const Display&) = delete;
    Display& operator=(const Display&) = delete;

    ~Display();

    // Input queue for processed frames
    ts::TSQueue<std::unique_ptr<FrameInfo>>& input_queue;

    /**
     * Main display loop - continuously processes and displays frames
     * Shows emotion detection results with bounding boxes and labels
     * Includes performance monitoring and FPS calculation
     */
    void display();
};
