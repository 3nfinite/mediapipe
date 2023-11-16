/**
 * @file libblendshape.h
 * @author Sen Pan (sen@hypevr.com)
 * @brief
 * @version 0.1
 * @date 2023-10-27
 *
 * @copyright Copyright (c) 2023, 3NFINITE
 *
 */

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_LANDMARKER_VOLYGON_BLENDSHAPE_H_

#define MEDIAPIPE_TASKS_CC_VISION_FACE_LANDMARKER_VOLYGON_BLENDSHAPE_H_

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

class BlendshapeGenerator {
 public:
  BlendshapeGenerator(int input_image_width, int input_image_height);
  ~BlendshapeGenerator();

 public:
  void Initialize(std::string ai_model_path);
  std::vector<std::pair<std::string, float>> Compute(cv::Mat input_image);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_LANDMARKER_VOLYGON_BLENDSHAPE_H_
