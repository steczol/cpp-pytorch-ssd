from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import torch
import cv2
import sys
import os

label_path = 'models/voc-model-labels.txt'
model_path = 'models/mb2-ssd-lite-mp-0_686.pth'
image_path = 'crowd.jpg'

# find device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
net.to(DEVICE)

if not issubclass(type(net), torch.nn.Module):
    print('net is not nn.Module')
    sys.exit()

predictor = create_mobilenetv2_ssd_lite_predictor(
    net, candidate_size=200, device=DEVICE)


orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

image_tensor = predictor.transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(DEVICE)

# traced_network = torch.jit.trace(predictor.net.forward, image_tensor)
traced_network = torch.jit.trace(predictor.net, image_tensor)

traced_network.save('models/traced_mb2-ssd-lite-mp-0_686.pt')
print('traced network saved')
