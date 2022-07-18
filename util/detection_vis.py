# Draw bbox on the images
from os.path import join as pjoin
import os
from PIL import Image
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import torch
import pickle

# For City2FoggyCity
# IMG_PATH = 'data/coco/val2017/'
# ANN_PATH = 'data/CocoFormatAnnos/cityscapes_foggy_val_cocostyle.json'

# For sim2city
# IMG_PATH = 'data/cityscapes/'
# ANN_PATH = 'data/CocoFormatAnnos/cityscapes_val_caronly_cocostyle.json'

# For city2bdd
IMG_PATH = 'data/bdd100k/images/100k/val/'
ANN_PATH = 'data/CocoFormatAnnos/bdd100k_labels_images_det_coco_val.json'

DEBUG = True

# For City2FoggyCity
# _cat_ids = [
#     1, 2, 3, 4, 5, 6, 7, 8
# ]

# For sim2city
# _cat_ids = [
#     0
# ]

# For city2bdd
_cat_ids = [
    1,2,3,4,5,6,7
]

num_classes = len(_cat_ids)
_classes = {
    ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)
}
_to_order = {cat_id: ind for ind, cat_id in enumerate(_cat_ids)}
coco = coco.COCO(ANN_PATH)
CAT_NAMES = [coco.loadCats([_classes[i + 1]])[0]['name'] \
             for i in range(num_classes)]
COLORS = [((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8) \
          for _ in range(num_classes)]
COLORS = torch.load("util/vis_colors.pth")

def get_num_classes():
    return _cat_ids

def print_category_res(category_res):
    for i in range(len(category_res)):
        print("{}:{:.4f}".format(CAT_NAMES[i],category_res[i]))

def unnormalize(np_array):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    np_array = np_array * std + mean
    np_array = np.uint8(np_array * 255)
    return np_array


def box_cxcywh_to_xyxy(box):
    bbox = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2],
                    dtype=np.int32)
    return bbox


def add_box(image, bbox, sc, cat_id):
    cat_id = _to_order[cat_id]
    cat_name = CAT_NAMES[cat_id]
    cat_size = cv2.getTextSize(cat_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    color = np.array(COLORS[cat_id]).astype(np.int32).tolist()
    txt = '{}{:.4f}'.format(cat_name, sc) if sc is not None else cat_name
    if bbox[1] - cat_size[1] - 2 < 0:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                      color, -1)
        cv2.putText(image, txt,
                    (bbox[0], bbox[1] + cat_size[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    else:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2),
                      color, -1)
        cv2.putText(image, txt,
                    (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    cv2.rectangle(image,
                  (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]),
                  color, 2)
    return image


def output_filter(output, sample_idx, threshold=0.5):
    sample_logits = output['pred_logits'][sample_idx].sigmoid()
    sample_boxes = output['pred_boxes'][sample_idx]

    sample_indices = torch.max(sample_logits, dim=-1).values > threshold

    sample_logits = sample_logits[sample_indices]
    sample_boxes = sample_boxes[sample_indices]
    sample_labels = torch.max(sample_logits, dim=-1).indices
    sample_confidence = torch.max(sample_logits, dim=-1).values

    filtered_output = {'boxes': sample_boxes,
                       'labels': sample_labels,
                       'confidence': sample_confidence}
    return filtered_output


def draw_boxes_on_image(draw_img, labels, img_shape):
    for box_idx in range(len(labels['boxes'])):
        box = labels['boxes'][box_idx].cpu().detach().numpy()
        box = [box[0] * img_shape[1], box[1] * img_shape[0], box[2] * img_shape[1], box[3] * img_shape[0]]
        bbox = box_cxcywh_to_xyxy(box)
        confidence = labels['confidence'][box_idx].cpu().detach().numpy() if 'confidence' in labels else None
        draw_img = add_box(draw_img.copy(), bbox, confidence,
                           labels['labels'].cpu().numpy()[box_idx])
    return draw_img


def detection_result_visualization(samples, targets, outputs, epoch, exp_dir, vis_interval=5):
    if not epoch % vis_interval == 0 or epoch == -1:
        return

    output_path = pjoin(exp_dir, 'detection_vis', f'epoch_{epoch}')
    os.makedirs(output_path, exist_ok=True)

    sample_tensors, sample_mask = samples.decompose()

    for idx in range(len(sample_tensors)):
        img_arr = unnormalize(sample_tensors[idx].cpu().numpy().transpose(1, 2, 0))

        sample_target = targets[idx]
        sample_output = output_filter(outputs, idx)

        img_shape = sample_target['size'].cpu().numpy()

        draw_targets_img = draw_boxes_on_image(img_arr.copy(), sample_target, img_shape)
        draw_prediction_img = draw_boxes_on_image(img_arr.copy(), sample_output, img_shape)

        # 1.image with targets, 2. image with prediction
        img = Image.fromarray(
            np.concatenate([draw_targets_img, draw_prediction_img], axis=1))
        img.save(pjoin(output_path, '{:04d}.jpg'.format(int(sample_target["image_id"].cpu().numpy()))))

def detection_result_visualization_video(samples, outputs, exp_dir):
    output_path = exp_dir
    
    img_arr = unnormalize(samples.cpu().numpy().transpose(1, 2, 0))

    def output_filter(output, sample_idx, threshold=0.5):
        sample_logits = output['pred_logits'][sample_idx].sigmoid()
        sample_boxes = output['pred_boxes'][sample_idx]

        sample_indices = torch.max(sample_logits, dim=-1).values > threshold

        sample_logits = sample_logits[sample_indices]
        sample_boxes = sample_boxes[sample_indices]
        sample_labels = torch.max(sample_logits, dim=-1).indices
        sample_confidence = torch.max(sample_logits, dim=-1).values

        filtered_output = {'boxes': sample_boxes,
                        'labels': sample_labels,
                        'confidence': sample_confidence}
        return filtered_output

    img_shape = (samples.shape[1],samples.shape[2])
    sample_output = output_filter(outputs, 0)
    draw_prediction_img = draw_boxes_on_image(img_arr.copy(), sample_output, img_shape)

    img = cv2.cvtColor(np.asarray(draw_prediction_img),cv2.COLOR_RGB2BGR)
    # img = Image.fromarray()
    return img
    # img.save(pjoin(output_path, img_name))

def teacher_student_detection_result_visualization(s_samples, t_samples, targets, pseudo_labels_1, pseudo_labels_2,
                                                   s_outputs, epoch, exp_dir, vis_interval=1):
    if not epoch % vis_interval == 0:
        return

    output_path = pjoin(exp_dir, 'detection_vis', f'epoch_{epoch}')
    os.makedirs(output_path, exist_ok=True)

    t_tensors, t_masks = t_samples.decompose()
    s_tensors, s_masks = s_samples.decompose()

    for idx in range(len(t_tensors)):
        s_img_arr = unnormalize(s_tensors[idx].cpu().numpy().transpose(1, 2, 0))
        t_img_arr = unnormalize(t_tensors[idx].cpu().numpy().transpose(1, 2, 0))

        sample_target, sample_pseudo_label_1 = targets[idx], pseudo_labels_1[idx]
        sample_output = output_filter(s_outputs, idx)

        img_shape = sample_target['size'].cpu().numpy()

        draw_targets_t_img = draw_boxes_on_image(t_img_arr.copy(), sample_target, img_shape)
        draw_pseudo_label1_t_img = draw_boxes_on_image(t_img_arr.copy(), sample_pseudo_label_1, img_shape)
        draw_prediction_s_img = draw_boxes_on_image(s_img_arr.copy(), sample_output, img_shape)

        if pseudo_labels_2 is not None:
            sample_pseudo_label_2 = pseudo_labels_2[idx]
            draw_pseudo_label2_t_img = draw_boxes_on_image(t_img_arr.copy(), sample_pseudo_label_2, img_shape)
            # 1.teacher image with targets, 2&3. teacher image with pseudo labels1&2, 4. student image with prediction
            img = Image.fromarray(
                np.concatenate(
                    [draw_targets_t_img, draw_pseudo_label1_t_img, draw_pseudo_label2_t_img, draw_prediction_s_img],
                    axis=1))
        else:
            # 1.teacher image with targets, 2. teacher image with pseudo labels, 3. student image with prediction
            img = Image.fromarray(
                np.concatenate([draw_targets_t_img, draw_pseudo_label1_t_img, draw_prediction_s_img], axis=1))
        img.save(pjoin(output_path, 'ts_{:04d}.jpg'.format(int(sample_target["image_id"].cpu().numpy()))))