/*
 *  Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
 *  All Rights Reserved.
 *  Authors:
 *      Daniel González Vargas <daniel.gonzalez@ridgerun.com>
 *      Daniel Rojas Marín <daniel.rojas@ridgerun.com>
 *
 *  The contents of this software are proprietary and confidential to RidgeRun,
 *  LLC.  No part of this program may be photocopied, reproduced or translated
 *  into another programming language without prior written consent of
 *  RidgeRun, LLC.  The user is free to modify the source code after obtaining
 *  a software license from RidgeRun.  All source code changes must be provided
 *  back to RidgeRun without any encumbrance.
 */

#ifdef HAVE_VS_CONFIG_H
#include <vsconfig.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rvs/allocators/cuda/cuda.hpp>
#include <rvs/allocators/host.hpp>
#include <rvs/common/argparser.hpp>
#include <rvs/iallocator.hpp>
#include <rvs/iintegrator.hpp>
#include <rvs/iinterpolator.hpp>
#include <rvs/image.hpp>
#include <rvs/image/rgba.hpp>
#include <rvs/istabilizer.hpp>
#include <rvs/iundistort.hpp>
#include <rvs/quaternion.hpp>
#include <rvs/stabilizer/spherical-exponential.hpp>
#include <rvs/utils/timer.hpp>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

/* Output video resolution */
static const int kOutputSize[] = {1280, 720};
/* Input video resolution */
static const int kInputSize[] = {2704, 2028};
/* Calibration video resolution */
static const int kCalSize[] = {4000, 3000};
/* Scaling factor for us-s coversion */
static constexpr double kFactor = 1e6;
/* Timestamp transposing coefficients */
static constexpr std::array<double, 2> kCoeffs = {-0.02951839, 0.04200435};

/* Image format to use */
using FormatType = rvs::RGBA<uint8_t>;
/* Image allocator to use */
using ImageType = rvs::Image<FormatType, rvs::HostAllocator>;
using ImageTypeCuda = rvs::Image<FormatType, rvs::CudaAllocator>;
/* Calibration matrices */
static std::array<float, 9> kCamMatrix = {1758.0517044105839,
                                          0.0,
                                          2000.0,
                                          0.0,
                                          1758.1725189833712,
                                          1500.0,
                                          0.0,
                                          0.0,
                                          1.0};
static std::vector<float> kDistCoeffs = {0.05827406, -0.00479972, 0.01750087,
                                         -0.00891355};

/* This function preprocesses the orientation data from the gyroscope.
 * It is loaded in a SensorPayload vector. */
rvs::RuntimeError LoadGyroSamples(
    std::vector<rvs::SensorPayload>& raw_gyro_vector,  // NOLINT
    std::vector<double>& timestamp_double_vector,      // NOLINT
    std::ifstream& csv_file_raw_gyro) {
  rvs::SensorPayload sample;
  std::string line;
  std::istringstream string_line;
  double timestamp_double_single = 0;
  char comma;
  rvs::RuntimeError ret{};

  /* check if file can be opened */
  if (!csv_file_raw_gyro.is_open()) {
    std::cerr << "Error  RAW_GYRO_INPUT_FILE CSV file not found." << std::endl;
    return rvs::RuntimeError(rvs::RuntimeError::FileError, "File not found");
  }

  /* Put full rawgyro in a SensorPayload vector */
  while (std::getline(csv_file_raw_gyro, line)) {
    string_line.clear();
    string_line.str(line);
    string_line >> timestamp_double_single >> comma >> sample.gyro.x >> comma >>
        sample.gyro.y >> comma >> sample.gyro.z;

    timestamp_double_vector.push_back(timestamp_double_single);

    /* Update the timastap with factor and cast it */
    raw_gyro_vector.push_back(sample);
    rvs::SensorPayload& last_element = raw_gyro_vector.back();
    last_element.gyro.timestamp =
        static_cast<uint64_t>(timestamp_double_single * kFactor);
  }
  return ret;
}

/* Prepare a payload vector for a specific time interval (a single frame).
 * If the current sample is larger than the interval, it updated the time
 * interval. */
bool PushSamplesForInterval(
    std::vector<rvs::SensorPayload>& raw_gyro_subsamples_vector,  // NOLINT
    uint32_t& sample_index,                                       // NOLINT
    double& time_step,                                            // NOLINT
    double interval, const std::vector<rvs::SensorPayload>& raw_gyro_vector,
    const std::vector<double> timestamp_double_vector) {
  while (true) {
    if (sample_index == timestamp_double_vector.size()) {
      return true;
    }
    if (timestamp_double_vector[sample_index] > time_step) {
      time_step += interval;
      return false;
    }
    raw_gyro_subsamples_vector.push_back(raw_gyro_vector[sample_index]);
    sample_index++;
  }
}

/* Readjust timestamps for the frame considering deltas between each measurement
 * and its accumulated effect. */
void TransposeTimestamps(
    std::vector<std::pair<rvs::Quaternion<double>, uint64_t>>&
        sample_vector  // NOLINT
) {
  for (uint64_t i = 0; i < sample_vector.size(); i++) {
    sample_vector[i].second += (kCoeffs[1] * kFactor);
    sample_vector[i].second /= (1 - kCoeffs[0]);
  }
}

/* Instantiate a shared pointer Image Allocator  for a frame */
std::shared_ptr<rvs::IImage> FormatFrame(cv::Mat& frame,  // NOLINT
                                         cv::Size& size,  // NOLINT
                                         rvs::UndistortAlgorithms backend,
                                         const bool is_output) {
  FormatType::ImagePtr host_ptr;
  FormatType::StrideType frame_stride;
  host_ptr = frame.ptr<uint8_t>();
  frame_stride = {static_cast<uint>(frame.step)};
  std::shared_ptr<rvs::IImage> outframe = nullptr;

  if (rvs::UndistortAlgorithms::kFishEyeCuda == backend) {
#ifdef HAVE_CUDA
    auto allocator = std::dynamic_pointer_cast<rvs::CudaAllocator>(
        rvs::IAllocator::Build(rvs::Allocators::kCudaManaged));
    auto outframecuda = std::make_shared<ImageTypeCuda>(
        size.width, size.height, frame_stride, nullptr, 0ul, allocator);
    FormatType::ImagePtr device_ptr = outframecuda->GetData();
    auto device_size = outframecuda->GetSize();
    if (!is_output) {
      cudaMemcpy(device_ptr, host_ptr, device_size, cudaMemcpyHostToDevice);
    } else {
      frame = cv::Mat(frame.rows, frame.cols, frame.type(), device_ptr);
    }
    outframe = outframecuda;
#endif
  } else {
    outframe = std::make_shared<ImageType>(size.width, size.height,
                                           frame_stride, &host_ptr);
  }

  return outframe;
}

/**
 * @brief Prints the welcome message of the tool
 */
static void PrintHeader() {
  std::cout << "---------------------------------------------------------\n"
               " RidgeRun Video Stabilisation Library                    \n"
               " Example Concept                                         \n"
               "---------------------------------------------------------\n\n";
}

/**
 * @brief Prints the help message
 */
static void PrintHelp() {
  std::cout << std::endl
            << "Usage:                                                     \n"
               "    rvs-complete-concept [-n NUMBER_OF_FRAMES] [-f INPUT_VIDE"
               "O] [-g GYRO_DATA]\n [-o OUTPUT_VIDEO] [-b BACKEND] [-w WIDTH]"
               " [-h HEIGHT]\n             [-s FIELD_OF_VIEW_SCALE]      \n\n"
               "Options:                                                   \n"
               "    --help: prints this message                            \n"
               "    -g: raw gyro data in CSV                               \n"
               "    -f: input video file to stabilise                      \n"
               "    -o: output video file stabilised. Def: output.mp4      \n"
               "    -b: backend. Options: opencv (def), opencl, cuda       \n"
               "    -n: number of frames to stabilise                      \n"
               "    -w: output width (def: 1280)                           \n"
               "    -h: output height (def: 720)                           \n"
               "    -s: field of view scale (def: 2.4)                     \n";
}

int main(int argc, char** argv) {
  /* Declare variables */
  rvs::RuntimeError ret{};
  rvs::ArgParser parser{argc, argv};
  INIT_PROFILER(rvsprofiler);

  PrintHeader();

  std::string raw_gyro_path;
  std::string mp4_video_path;
  std::string out_video_path;
  std::string backend_str;
  int number_frames = 0;
  int width = kOutputSize[0];
  int height = kOutputSize[1];
  double fov = 2.4f;

  rvs::UndistortAlgorithms backend;

  if (parser.Exists("--help")) {
    PrintHelp();
    return 0;
  }

  raw_gyro_path =
      parser.Exists("-g") ? parser.GetOption("-g") : RAW_GYRO_INPUT_FILE;
  mp4_video_path = parser.Exists("-f") ? parser.GetOption("-f") : TEST_VIDEO;
  out_video_path = parser.Exists("-o") ? parser.GetOption("-o") : "output.mp4";
  backend_str = parser.Exists("-b") ? parser.GetOption("-b") : "opencv";
  number_frames =
      parser.Exists("-n") ? std::stoi(parser.GetOption("-n")) : number_frames;
  width = parser.Exists("-w") ? std::stoi(parser.GetOption("-w")) : width;
  height = parser.Exists("-h") ? std::stoi(parser.GetOption("-h")) : height;
  fov = parser.Exists("-s") ? std::stod(parser.GetOption("-s")) : fov;

  if ("cuda" == backend_str) {
    std::cout << "Using CUDA backend" << std::endl;
    backend = rvs::UndistortAlgorithms::kFishEyeCuda;
  } else if ("opencl" == backend_str) {
    std::cout << "Using OpenCL backend" << std::endl;
    backend = rvs::UndistortAlgorithms::kFishEyeOpenCL;
  } else {
    std::cout << "Using OpenCV backend" << std::endl;
    backend = rvs::UndistortAlgorithms::kFishEyeOpenCV;
    backend_str = "opencv";
  }

  /* Gyroscope data variables */
  std::ifstream csv_file_raw_gyro(raw_gyro_path);
  std::vector<rvs::SensorPayload> raw_gyro_vector{},
      raw_gyro_subsamples_vector{};
  std::vector<double> timestamp_double_vector;

  /* Input video variables */
  cv::VideoCapture inputVideo;
  inputVideo.setExceptionMode(true);
  inputVideo.open(mp4_video_path, cv::CAP_ANY);
  cv::Size output_size = cv::Size(width, height);
  int codec = 0;
  float fps = 0;
  int nframes = 0;
  int frameidx = 0;

  /* Payload data integration variables */
  std::vector<std::pair<rvs::Quaternion<double>, uint64_t>> integrated_vector{};
  rvs::Quaternion<double> initial_orientation{0.70710678, 0, 0.70710678, 0};
  uint64_t initial_time = 0;
  bool last_run = false;
  uint32_t sample_index = 0;
  double frame_interval = 0;
  double time_step = 0;

  /* Interpolation vectors */
  std::vector<std::pair<rvs::Quaternion<double>, uint64_t>>
      accumulated_buffer{}, interpolated_buffer{}, interpolated_frame_buffer{};

  /* Stabilization variables */
  std::vector<rvs::Quaternion<double>> stabilized_buffer{};
  rvs::Quaternion<double> rot_between{0, 0, 0, 0};

  /* RVS component parameters */
  rvs::IntegratorSettings intr_settings{true, false, false};
  rvs::Quaternion<double> init{0, 0, 0, 0};
  rvs::SphericalExponentialParams slerp_params{0.206, init};
  auto integrator_settings =
      std::make_shared<rvs::IntegratorSettings>(intr_settings);
  auto interpolator_settings = std::make_shared<rvs::InterpolatorSettings>();
  auto stabilizer_params =
      std::make_shared<rvs::SphericalExponentialParams>(slerp_params);

  /* RVS component instantiation */
  auto integrator{rvs::IIntegrator::Build(
      rvs::IntegratorAlgorithms::kSimpleComplementaryIntegrator,
      integrator_settings)};
  auto interpolator = rvs::IInterpolator::Build(
      rvs::InterpolatorAlgorithms::kSlerp, interpolator_settings);
  auto stabilizer = rvs::IStabilizer::Build(
      rvs::StabilizerAlgorithms::kSphericalExponential, stabilizer_params);
  auto undistort = rvs::IUndistort::Build(backend);

  /* Video frame variables */
  cv::Size input_size;
  cv::Mat input_frame, output_frame;

  /* Set up parameters and runtime variables prior to stabilization */
  /* Loading samples into vectors */
  LoadGyroSamples(raw_gyro_vector, timestamp_double_vector, csv_file_raw_gyro);

  /* Set integrator's initial orientation and time */
  integrator->Reset(initial_orientation, initial_time);

  /* Open video and discard a frame */
  if (!inputVideo.isOpened()) {
    std::cerr << "Error opening input video." << std::endl;
    return -1;
  }
  inputVideo.grab();
  nframes = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_COUNT));
  number_frames = number_frames == 0 ? nframes : number_frames;

  /* Extract video properties */
  codec = static_cast<int>(inputVideo.get(cv::CAP_PROP_FOURCC));
  fps = static_cast<float>(inputVideo.get(cv::CAP_PROP_FPS));
  frame_interval = 1. / fps;
  time_step = frame_interval;

  /* Update integrator parameters */
  interpolator_settings->interval = frame_interval * kFactor;

  /* Set video output */
  cv::VideoWriter outputVideo(out_video_path, codec, fps, output_size);
  if (!outputVideo.isOpened()) {
    std::cerr << "Error opening output video." << std::endl;
    return -1;
  }

  /* Compute the matrices */
  undistort->SetCameraMatrices(kCamMatrix, kDistCoeffs, kCalSize[0],
                               kCalSize[1]);

  /* Main loop to stabilize video */
  std::cout << "Video File: " << TEST_VIDEO << std::endl;
  std::cout << "Gyro Data: " << RAW_GYRO_INPUT_FILE << std::endl;
  std::cout << "Starting Processing the Video" << std::endl;

  /* Get profiler instance */
  GET_PROFILE_INSTANCE(rvs_image_creation, rvsprofiler);
  GET_PROFILE_INSTANCE(rvs_stabilization_computation, rvsprofiler);
  GET_PROFILE_INSTANCE(rvs_image_correction, rvsprofiler);
  GET_PROFILE_INSTANCE(other_image_processing, rvsprofiler);
  while (!last_run && inputVideo.isOpened()) {
    frameidx++;
    std::cout << "\rProcessing frame: " << frameidx << "/" << number_frames
              << " (" << ((frameidx * 100) / number_frames) << "%)"
              << std::flush;
    std::shared_ptr<rvs::IImage> rvs_input_img, rvs_output_img;
    /*
     * Integration
     * Grab a set of samples that correspond to a single frame and:
     * - Integrate them into quaternions
     * - adjust the timestamps
     */
    raw_gyro_subsamples_vector.clear();

    rvs_stabilization_computation->reset();
    last_run = PushSamplesForInterval(raw_gyro_subsamples_vector, sample_index,
                                      time_step, frame_interval,
                                      raw_gyro_vector, timestamp_double_vector);

    ret = integrator->Apply(integrated_vector, raw_gyro_subsamples_vector);

    TransposeTimestamps(integrated_vector);

    accumulated_buffer.insert(accumulated_buffer.end(),
                              integrated_vector.begin(),
                              integrated_vector.end());

    /*
     * Interpolation
     * Take the accumulated orientation vector and interpolate.
     * If successful, it will clear up the accumulated vector.
     * It will accumulate interpolation results until there are 3 samples.
     */
    ret = interpolator->Apply(interpolated_buffer, accumulated_buffer);
    if (ret.GetCode() == rvs::RuntimeError::IncompatibleParameters) {
      continue;
    }

    interpolated_frame_buffer.push_back(interpolated_buffer[0]);
    if (interpolated_frame_buffer.size() < 3) {
      continue;
    }

    /*
     * Stabilization
     * Stabilize orientations based on interpolated results for the frame and
     * rotate middle value around the original interpolated orientation. Then,
     * popping the head of the interpolated buffer.
     */
    ret = stabilizer->Apply(stabilized_buffer, interpolated_frame_buffer, fps);
    rvs::Quaternion<double> rot_between =
        stabilized_buffer[1].Conjugate() * interpolated_frame_buffer[1].first;
    interpolated_frame_buffer.erase(interpolated_frame_buffer.begin());
    rvs_stabilization_computation->tick();

    /*
     * Image preparation
     * Convert input frame to BGRA format and allocate pointer for the input and
     * output images.
     */
    other_image_processing->reset();
    inputVideo >> input_frame;
    if (input_frame.empty()) {
      std::cerr << "Input video is empty." << std::endl;
      break;
    }

    cv::cvtColor(input_frame, input_frame, cv::COLOR_BGR2BGRA);
    output_frame.create(output_size.height, output_size.width,
                        input_frame.type());
    input_size = cv::Size(input_frame.cols, input_frame.rows);
    other_image_processing->tick();

    rvs_image_creation->reset();
    rvs_input_img = FormatFrame(input_frame, input_size, backend, false);
    rvs_output_img = FormatFrame(output_frame, output_size, backend, true);
    rvs_image_creation->tick();

    /* Transform image and write frame into output video */
    rvs_image_correction->reset();
    ret = undistort->Apply(rvs_output_img, rvs_input_img, rot_between, fov);
    if (ret.IsError()) {
      std::cerr << "ERROR: " << ret.what() << std::endl;
    }
    rvs_image_correction->tick();
    cv::cvtColor(output_frame, output_frame, cv::COLOR_BGRA2BGR);
    outputVideo.write(output_frame);

    if (number_frames == frameidx) {
      break;
    }
  }

  /* Sum times */
  std::cout << std::endl << rvsprofiler;
  double avg = rvs_image_creation->average + rvs_image_correction->average +
               rvs_stabilization_computation->average;
  std::cout << "Finished. RVS over " << backend_str << " took: " << avg
            << " seconds per frame. On an image: " << width << "x" << height
            << " pixels" << std::endl;

  /* Release input and output videos */
  inputVideo.release();
  outputVideo.release();
  return 0;
}
