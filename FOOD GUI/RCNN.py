import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torchvision.models
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import cv2

classes = ['Pear',
 'Orange',
 'Salad',
 'Grape',
 'Muffin',
 'Egg',
 'Banana',
 'Bread',
 'Cucumber',
 'Broccoli',
 'Cookie',
 'Carrot',
 'Cheese',
 'Strawberry',
 'Hot dog',
 'Bagel',
 'Lemon',
 'Apple',
 'Burrito',
 'Coffee',
 'French fries',
 'Pizza',
 'Juice',
 'Tomato',
 'Hamburger',
 'Pasta',
 'Waffle',
 'Potato',
 'Sandwich',
 'Doughnut',
 'Pancake',
 'Croissant']


def get_result_from_model(test_img, thresh):

    test_data = torchvision.datasets.ImageFolder('C:/Users/skyho/Desktop/test_image_folder/',loader = plt.imread,transform=transforms.ToTensor())

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 200),), aspect_ratios=((0.5, 1.0, 2.0),))
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    rcnn_v1 = FasterRCNN(backbone, num_classes=32, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    try:
        # model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(8,0.00005,2,15)
        checkpoint = torch.load('C:/Users/skyho/Desktop/final_model.pth', map_location='cpu') #, map_location='cpu'
        rcnn_v1.load_state_dict(checkpoint['model_state_dict'])
        rcnn_v1.eval()
    except IOError:
        print("Can't find saved model~")

    # result = []
    #data = torchvision.datasets.ImageFolder(img_path, loader=plt.imread, transform=transforms.ToTensor())
    result = []
    with torch.no_grad():
        result.append(rcnn_v1([test_data[0][0]]))

    # plot the boxes on the result image
    # print labels
    # save the image somewhere and return the path
    # cv2_im = cv2.imread(test_img)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2_im = []
    #for i in range(len(test_data)):
    cv2_im.append(cv2.imread(test_data.imgs[0][0]))

    #for i in range(len(test_data)):
    i = 0
    first_box = result[i][0]['boxes'][0].unsqueeze(0)
    box_id = 0

    for box in result[i][0]['boxes']:

        if (box_id==0 or jaccard(first_box,box.unsqueeze(0)).tolist()[0][0]<0.6):
            if result[i][0]['scores'].tolist()[box_id]>=thresh:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                #check other boxes

                flag = True
                for each in range(result[i][0]['boxes'].shape[0]):
                    if each!=box_id and result[i][0]['scores'].tolist()[each]>=thresh and jaccard(first_box,result[i][0]['boxes'][each].unsqueeze(0)).tolist()[0][0]<0.6:
                        o_x1 = int(result[i][0]['boxes'][each][0])
                        o_y1 = int(result[i][0]['boxes'][each][1])
                        o_x2 = int(result[i][0]['boxes'][each][2])
                        o_y2 = int(result[i][0]['boxes'][each][3])
                        if x1>=o_x1-3 and y1>=o_y1-3 and x2<=o_x2+3 and y2<=o_y2+3 and result[i][0]['labels'][box_id]==result[i][0]['labels'][each]:
                            flag = False
                            break
                    if flag:
                        cv2_im[i] = cv2.rectangle(cv2_im[i],(x1,y1),(x2,y2),(0,255,0),3)
                        cv2.putText(cv2_im[i],classes[result[i][0]['labels'][box_id]],
                            (x1,y2),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
        box_id += 1

    detection_result = test_img[:-4] + '_result.png'
    #for i in range(len(test_data)):
    cv2.imwrite(detection_result, cv2_im[i])
    return detection_result

    # for box in result[0][0]['boxes']:
    #     x1,x2,y1,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    #     cv2_im = cv2.rectangle(cv2_im,(x1,y1),(x2,y2),(0,255,0),5)
    # detection_result = test_img[:-4]+'_result.png'
    # cv2.imwrite(detection_result, cv2_im)
    # cv2.imwrite(result_path,cv2_im)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
