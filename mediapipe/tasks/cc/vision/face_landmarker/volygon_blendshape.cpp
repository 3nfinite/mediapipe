/**
 * @file libblendshape.cpp
 * @author Sen Pan (sen@hypevr.com)
 * @brief
 * @version 0.1
 * @date 2023-10-27
 *
 * @copyright Copyright (c) 2023, 3NFINITE
 *
 */

#include "volygon_blendshape.h"

#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "stb_image.h"

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceBlendshapesGraphOptions;

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;
using ::mediapipe::tasks::vision::face_landmarker::proto::
    FaceLandmarkerGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kNormLandmarksName[] = "norm_landmarks";
constexpr char kBlendshapesTag[] = "BLENDSHAPES";
constexpr char kBlendshapesName[] = "blendshapes";

struct BlendshapeGenerator::Impl {
  int input_image_width = -1;
  int input_image_height = -1;

  mediapipe::api2::builder::Graph graph;
  std::unique_ptr<TaskRunner> task_runner;

  bool init = false;
  uint8_t* image_data_buffer;
  std::shared_ptr<mediapipe::ImageFrame> image_frame;
  std::map<std::string, mediapipe::Packet> input;
};

BlendshapeGenerator::BlendshapeGenerator(int input_image_width,
                                         int input_image_height) {
  impl_ = std::make_unique<BlendshapeGenerator::Impl>();

  impl_->input_image_width = input_image_width;
  impl_->input_image_height = input_image_height;
}

BlendshapeGenerator::~BlendshapeGenerator() {}

void BlendshapeGenerator::Initialize(std::string ai_model_path) {
  auto& face_landmarker = impl_->graph.AddNode(
      "mediapipe.tasks.vision.face_landmarker."
      "FaceLandmarkerGraph");

  auto* options = &face_landmarker.GetOptions<FaceLandmarkerGraphOptions>();

  options->mutable_base_options()->mutable_model_asset()->set_file_name(
      ai_model_path);

  options->mutable_face_detector_graph_options()->set_num_faces(1);
  options->mutable_base_options()->set_use_stream_mode(true);

  impl_->graph[Input<mediapipe::Image>(kImageTag)].SetName(kImageName) >>
      face_landmarker.In(kImageTag);

  face_landmarker.Out(kNormLandmarksTag).SetName(kNormLandmarksName) >>
      impl_->graph[Output<std::vector<mediapipe::NormalizedLandmarkList>>(
          kNormLandmarksTag)];
  face_landmarker.Out(kBlendshapesTag).SetName(kBlendshapesName) >>
      impl_->graph[Output<std::vector<mediapipe::ClassificationList>>(
          kBlendshapesTag)];

  impl_->task_runner =
      TaskRunner::Create(
          impl_->graph.GetConfig(),
          absl::make_unique<
              mediapipe::tasks::core::MediaPipeBuiltinOpResolver>())
          .value();

  impl_->image_frame = std::make_shared<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, impl_->input_image_width,
      impl_->input_image_height);

  impl_->image_data_buffer = impl_->image_frame->MutablePixelData();

  impl_->input["image"] = mediapipe::MakePacket<mediapipe::Image>(
      mediapipe::Image(std::move(impl_->image_frame)));
}

std::vector<std::pair<std::string, float>> BlendshapeGenerator::Compute(
    cv::Mat input_image) {
  auto t0 = std::chrono::high_resolution_clock::now();

  cv::Mat frame_rgb;
  cv::cvtColor(input_image, frame_rgb, cv::COLOR_BGR2RGB);

  int image_size = impl_->input_image_width * impl_->input_image_height * 3;

  std::memcpy(impl_->image_data_buffer, frame_rgb.ptr(), image_size);

  auto t1 = std::chrono::high_resolution_clock::now();

  auto output_packets = impl_->task_runner->Process(impl_->input);

  std::vector<std::pair<std::string, float>> blendshape;

  if ((*output_packets)[kBlendshapesName].IsEmpty()) {
    return blendshape;
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  const std::vector<mediapipe::ClassificationList>& blendshapes =
      (*output_packets)[kBlendshapesName]
          .Get<std::vector<mediapipe::ClassificationList>>();

  for (int i = 0; i < 52; ++i) {
    std::string blendshape_label = blendshapes[0].classification(i).label();
    float blendshape_score = blendshapes[0].classification(i).score();

    blendshape.push_back({blendshape_label, blendshape_score});
  }

  int prepare_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  int inference_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  return blendshape;
}
