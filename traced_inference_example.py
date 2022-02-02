# Based on run_ssd_example.py

from vision.transforms.transforms import (
    Compose, Resize, SubtractMeans, ToTensor
)

from vision.utils.misc import Timer
from vision.utils import box_utils

import numpy as np
import torch
import cv2
import sys
import os


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'info: device is {DEVICE}')


def get_filename(full_path) -> str:
    return os.path.split(full_path)[-1]


def draw_boxes(in_img, boxes, labels, class_names, probabilities):
    out_img = in_img
    for i in range(boxes.size(0)):
        box = boxes[i, :].numpy().astype(int)
        cv2.rectangle(out_img, (box[0], box[1]),
                      (box[2], box[3]), (255, 255, 0), 4)
        label = f"{class_names[labels[i]]}: {probabilities[i]:.2f}"
        cv2.putText(
            out_img,
            label,
            (box[0] + 20, box[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # font scale
            (255, 0, 255),
            2,
        )  # line type
    return out_img


if len(sys.argv) < 3:
    print(
        'usage: python sysiko_traced_infere_example <model_path> <image_path> [labels_path]')
    sys.exit(-1)

model_path = sys.argv[1]
image_path = sys.argv[2]
labels_path = None

if len(sys.argv) == 4:
    labels_path = sys.argv[3]

if labels_path and os.path.exists(labels_path):
    class_names = [name.strip() for name in open(labels_path).readlines()]
else:
    class_names = ("BACKGROUND",
                   "aeroplane",
                   "bicycle",
                   "bird",
                   "boat",
                   "bottle",
                   "bus",
                   "car",
                   "cat",
                   "chair",
                   "cow",
                   "diningtable",
                   "dog",
                   "horse",
                   "motorbike",
                   "person",
                   "pottedplant",
                   "sheep",
                   "sofa",
                   "train",
                   "tvmonitor",)


model = torch.jit.load(model_path)
print('info: module loaded')


# Real image input
# read image
orig_image = cv2.imread(image_path)

image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

height, width, _ = image.shape

# Infer
size = 300
mean = [127, 127, 127]
std = 128.0

resize = Resize(size)
img_resized, _, _ = resize(image)

subtract_means = SubtractMeans(mean)
img_subtracted, _, _ = subtract_means(img_resized)


def normalize(img, boxes, labels=None): return (img/std, boxes, labels)


img_normalized = img_subtracted / std

transforms = Compose([Resize(size), SubtractMeans(
    mean), lambda img, boxes, labels=None: (img/std, boxes, labels), ToTensor()])

transformed_image, _, _ = transforms(image)
transformed_images = transformed_image.unsqueeze(0)

# store tensor
transformed_images = transformed_images.to(DEVICE)


with torch.no_grad():
    timer = Timer()
    timer.start()
    ret_traced_scores, ret_traced_boxes = model.forward(transformed_images)
    print(f'Traced inference time: {timer.end()} s')

ret_traced_boxes = ret_traced_boxes[0].to('cpu')
ret_traced_scores = ret_traced_scores[0].to('cpu')


nms_method = None
top_k = 10
iou_threshold = 0.45
sigma = 0.5
prob_threshold = 0.4
candidate_size = 200

picked_box_probs = []
picked_labels = []
for class_index in range(1, ret_traced_scores.size(1)):
    probs = ret_traced_scores[:, class_index]
    mask = probs > prob_threshold
    probs = probs[mask]
    if probs.size(0) == 0:
        continue
    subset_boxes = ret_traced_boxes[mask, :]
    box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)

    # print(f'class: {class_names[class_index]}')
    # print(f'before nms: {box_probs}')
    box_probs = box_utils.nms(box_probs, nms_method,
                              score_threshold=prob_threshold,
                              iou_threshold=iou_threshold,
                              sigma=sigma,
                              top_k=top_k,
                              candidate_size=candidate_size)
    # print(f'after nms: {box_probs}')
    picked_box_probs.append(box_probs)
    picked_labels.extend([class_index] * box_probs.size(0))

# print(f'picked_boxes: {picked_box_probs}')

if not picked_box_probs:
    traced_boxes = torch.tensor([])
    traced_labels = torch.tensor([])
    traced_probs = torch.tensor([])
else:
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    traced_boxes = picked_box_probs[:, :4]
    traced_labels = torch.tensor(picked_labels)
    traced_probs = picked_box_probs[:, 4]


# draw boxes
traced_out_img = draw_boxes(
    orig_image, traced_boxes, traced_labels, class_names, traced_probs)

# store results
filename, fileext = os.path.splitext(get_filename(image_path))
# path = 'out/'+filename +"_detected_traced"+fileext
path = os.path.join('out', filename+'_detected_traced_py'+fileext)
cv2.imwrite(path, traced_out_img)
print(f"Found {len(traced_probs)} objects. The output image is {path}")

print('info: ok')
