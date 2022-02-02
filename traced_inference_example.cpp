#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/grad_mode.h>

#include <chrono>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <torchvision/vision.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

// using namespace torch::indexing;
using namespace std::chrono;
using clk = std::chrono::high_resolution_clock;

const float EPS = 1e-5;

struct Box {
  Box(float left, float top, float right, float bottom, float score)
      : left(left), top(top), right(right), bottom(bottom), score(score) {}
  float left, top, right, bottom, score;

  float area() const {
    auto w = right - left;
    auto h = bottom - top;

    return std::max(0.F, w * h);
  }
};

std::ostream &operator<<(std::ostream &out, const Box &b) {
  out << "(" << b.left << ", " << b.top << ") -- (" << b.right << ", "
      << b.bottom << ") : " << b.score;
  return out;
}

using Boxes = std::vector<Box>;

Box overlap(const Box &lhs, const Box &rhs) {
  return {std::max(lhs.left, rhs.left), std::max(lhs.top, rhs.top),
          std::min(lhs.right, rhs.right), std::min(lhs.bottom, rhs.bottom),
          0.F};
}

float iou_of(const Box &box0, const Box &box1, float eps = EPS) {
  auto intersection_area = overlap(box0, box1).area();
  return intersection_area /
         (box0.area() + box1.area() - intersection_area + eps);
}

std::vector<float> iou_of(const Boxes &boxes0, const Box &box,
                          float eps = EPS) {
  std::vector<float> ret;
  std::transform(
      boxes0.cbegin(), boxes0.cend(), std::back_inserter(ret),
      [&box](const Box &box0) -> float { return iou_of(box0, box); });
  return ret;
}

template <class BoxesIt>
std::vector<float> iou_of(BoxesIt first, BoxesIt last, const Box &box,
                          float eps = EPS) {
  std::vector<float> ret;
  std::transform(
      first, last, std::back_inserter(ret),
      [&box](const Box &box0) -> float { return iou_of(box0, box); });
  return ret;
}

std::vector<float> iou_of(const Boxes &boxes0, const Boxes &boxes1,
                          float eps = EPS) {
  std::vector<float> ret;
  std::transform(
      boxes0.cbegin(), boxes0.cend(), boxes1.cbegin(), std::back_inserter(ret),
      [](const Box &box0, const Box &box1) { return iou_of(box0, box1); });
  return ret;
}

template <class BoxesIt, class IousIt>
BoxesIt removeGivenIous(BoxesIt boxes_first, BoxesIt boxes_last,
                        IousIt ious_first, IousIt ious_last,
                        float iou_threshold) {
  assert(std::distance(boxes_first, boxes_last) ==
         std::distance(ious_first, ious_last));

  auto ious_it = ious_first;
  auto boxes_it = boxes_first;

  while ((ious_it != ious_last) && (boxes_it != boxes_last)) {
    if (*ious_it <= iou_threshold) {
      *boxes_first++ = std::move(*boxes_it);
    }

    ++ious_it;
    ++boxes_it;
  };

  return boxes_first;
}

Boxes hard_nms(Boxes &boxes, int top_k, int candidate_size) {
  Boxes ret;
  std::sort(boxes.begin(), boxes.end(), [](const Box &box0, const Box &box1) {
    return box0.score > box1.score;
  });

  const float iou_threshold = 0.45F;

  while (!boxes.empty()) {
    auto box = boxes.at(0);
    ret.push_back(box);

    // pop_front
    boxes.erase(boxes.begin());

    if (top_k > 0 && top_k == ret.size()) {
      break;
    }

    auto ious = iou_of(boxes.begin(), boxes.end(), box);

    boxes.erase(removeGivenIous(boxes.begin(), boxes.end(), ious.begin(),
                                ious.end(), iou_threshold),
                boxes.end());
  }
  return ret;
}

template <typename T> std::string rounded(T number) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << number;
  return ss.str();
}

cv::Mat draw_boxes(const cv::Mat &orig_image, const Boxes &boxes,
                   const std::vector<std::string> &labels_str) {
  assert(boxes.size() == labels_str.size());

  cv::Mat ret = orig_image;
  for (int i = 0; i < boxes.size(); i++) {
    const auto &box = boxes.at(i);
    cv::rectangle(ret, cv::Point2f{box.left, box.top},
                  cv::Point2f{box.right, box.bottom}, {255, 255, 0}, 4);
    // std::string label = labels_str.at(i) + ": " + std::to_string(box.score);
    const std::string label = labels_str.at(i) + ": " + rounded(box.score);
    cv::putText(ret, label, cv::Point2f{box.left + 20, box.top + 40},
                cv::FONT_HERSHEY_SIMPLEX, 1, {255, 0, 255}, 2);
  }

  return ret;
}

const auto CLASS_NAMES = std::vector<std::string>{
    "BACKGROUND", "aeroplane",   "bicycle", "bird",  "boat",
    "bottle",     "bus",         "car",     "cat",   "chair",
    "cow",        "diningtable", "dog",     "horse", "motorbike",
    "person",     "pottedplant", "sheep",   "sofa",  "train",
    "tvmonitor"};

int main(int argc, char **argv) {
  auto model_path = std::string();
  auto image_path = std::string();
  auto labels_path = std::string();
  std::vector<std::string> class_names;

  if (argc < 3) {
    std::cerr << "usage: sysiko_infer_example <path-to-exported-module> "
                 "<path_to_image> [path_to_labels]\n";
    std::cerr << "example: sysiko_infer_example models/model.pt\n";
    return -1;
  }
  if (argc >= 3) {
    model_path = std::string(argv[1]);
    image_path = std::string(argv[2]);
    if (argc == 3) {
      class_names = CLASS_NAMES;
    } else {
      // read class names from file put as third argument
    }
  }

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model | \n" << e.what() << "\n";
    return -1;
  }

  std::cout << "info: module loaded\n";

  // Read image input
  auto orig_image = cv::imread(image_path);
  std::cout << "info: image read\n";

  cv::Mat image;
  cv::cvtColor(orig_image, image, cv::COLOR_BGR2RGB);

  auto height = image.rows;
  auto width = image.cols;

  // Infer
  const auto size = 300;
  const auto mean = 127;
  const auto stddev = 128.0;

  // Transforms
  //  Resize
  cv::Mat img_resized;
  cv::resize(image, img_resized, {size, size});
  //  Subtract Means
  // cv::Mat img_subtracted;
  // img_resized.convertTo(img_subtracted, CV_32FC3, 1, -127);
  // //  Normalize
  // cv::Mat img_normalized;
  // img_subtracted.convertTo(img_normalized, CV_32FC3, 1.0 / stddev);

  cv::Mat img_normalized;
  img_resized.convertTo(img_normalized, CV_32FC3, 1.0 / stddev, -mean / stddev);

  //   Create input
  at::Tensor input_tensor = torch::from_blob(
      img_normalized.data,
      {1, img_normalized.rows, img_normalized.cols, img_normalized.channels()});
  input_tensor = input_tensor.permute({0, 3, 1, 2});

  input_tensor = input_tensor.cuda();

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);

  // Infer
  c10::IValue output;
  {
    at::NoGradGuard no_grad;
    auto start = clk::now();
    output = module.forward(inputs);
    auto stop = clk::now();
    std::cout << "Inference time: "
              << duration_cast<microseconds>(stop - start).count() / 1000000.F
              << " s" << std::endl;
  }

  // output is a tuple of tensors: [scores, boxes]
  auto scores = output.toTuple()->elements().at(0).toTensor();
  auto boxes = output.toTuple()->elements().at(1).toTensor();

  scores = scores.cpu();
  boxes = boxes.cpu();

  // since there is just one image on input:
  scores = scores[0]; // scores has now size [3000, 21]
  boxes = boxes[0];   // boxes has now size [3000, 4]

  // Postprocessing:
  const float prob_threshold = 0.4;
  const int top_k = 10;
  const int candidate_size = 200;

  auto scores_transp = scores.permute({1, 0});

  Boxes picked_boxes;
  std::vector<int> picked_labels;
  for (int class_index = 1; class_index < scores.size(1); class_index++) {
    auto probs = scores_transp[class_index];
    Boxes subset_boxes;
    for (int i = 0; i < scores.size(0); i++) {
      auto prob = probs[i].item<float>();
      if (prob > prob_threshold) {
        std::vector<float> prob_box(static_cast<float *>(boxes[i].data_ptr()),
                                    static_cast<float *>(boxes[i].data_ptr()) +
                                        boxes[i].numel());

        Box v_probable_bbox(prob_box[0], prob_box[1], prob_box[2], prob_box[3],
                            prob);

        subset_boxes.push_back(v_probable_bbox);
      }
    }

    if (subset_boxes.empty()) {
      continue;
    }

    subset_boxes = hard_nms(subset_boxes, top_k, candidate_size);

    picked_boxes.insert(picked_boxes.end(), subset_boxes.begin(),
                        subset_boxes.end());
    picked_labels.insert(picked_labels.end(), subset_boxes.size(), class_index);
  }

  std::for_each(picked_boxes.begin(), picked_boxes.end(),
                [&width, &height](Box &box) -> void {
                  box.left *= static_cast<float>(width);
                  box.top *= static_cast<float>(height);
                  box.right *= static_cast<float>(width);
                  box.bottom *= static_cast<float>(height);
                });

  std::vector<std::string> picked_labels_str;
  std::transform(picked_labels.begin(), picked_labels.end(),
                 std::back_inserter(picked_labels_str),
                 [](int idx) -> std::string { return CLASS_NAMES[idx]; });

  cv::Mat traced_out_img =
      draw_boxes(orig_image, picked_boxes, picked_labels_str);

  std::string out_img_path = "out/detected_traced_cpp.jpg";
  cv::imwrite(out_img_path, traced_out_img);

  std::cout << "info: found " << picked_boxes.size()
            << " objects. The output image is " << out_img_path << std::endl;

  std::cout << "info: ok\n";
  return 0;
}