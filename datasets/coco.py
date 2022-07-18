# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def _make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_coco(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "coco" / "train2017", root / "coco" / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "coco" / "val2017", root / "coco" / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=_make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


def get_transforms(image_set, aug_type):
    assert aug_type in ['strong', 'weak', 'typical'], ValueError(f'unknown aug_type `{image_set}`')
    if aug_type == 'typical':
        return _make_coco_transforms(image_set)
    else:
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if aug_type == 'weak':
            return T.Compose([_make_weak_transforms(image_set),
                              T.ToTensor(),
                              normalize])
        else:
            return T.Compose([_make_weak_transforms(image_set),
                              _make_strong_transforms(image_set),
                              normalize])


def _make_strong_transforms(image_set='train'):
    if image_set == 'train':
        return T.Compose([
            T.RandomApply(T.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(T.GaussianBlur([0.1, 2.0]), p=0.5),
            T.ToTensor(),
            # T.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            # T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            # T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random")
        ])

    raise ValueError(f'unknown {image_set}')


def _make_weak_transforms(image_set='train'):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            )
        ])

    raise ValueError(f'unknown {image_set}')


# class TeacherStudentDataset(Dataset):
#     def __init__(self, img_folder, ann_file, return_masks, cache_mode=False, local_rank=0, local_size=1):
#         self.weak_transforms = _make_weak_transforms()
#         self.strong_transforms = _make_strong_transforms()
#         self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#
#         self.coco_dataset = CocoDetection(img_folder, ann_file,
#                                           transforms=self.weak_transforms,
#                                           return_masks=return_masks,
#                                           cache_mode=cache_mode, local_rank=local_rank,
#                                           local_size=local_size)
#
#     def __getitem__(self, idx):
#         weak_samples, weak_targets = self.coco_dataset[idx]
#         weak_samples, weak_targets = T.Compose([T.ToTensor(),
#                                      self.normalize])(weak_samples, weak_targets)
#         strong_samples, strong_target = T.Compose([self.strong_transforms,
#                                      self.normalize])(weak_samples, weak_targets)
#         return {'teacher': (weak_samples, weak_targets),
#                 'student': (strong_samples, strong_target)}


class TeacherStudentDataset(TvCocoDetection):
    def __init__(self, img_folder, ann_file, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(TeacherStudentDataset, self).__init__(img_folder, ann_file,
                                                    cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.weak_transforms = _make_weak_transforms()
        self.strong_transforms = _make_strong_transforms()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(TeacherStudentDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # img, target = T.Compose([T.ToTensor(),
        #                          self.normalize])(img, target)
        # return img, target, img, target

        img, target = self.weak_transforms(img, target)
        t_img, t_target = T.Compose([T.ToTensor(),
                                     self.normalize])(img, target)
        s_img, s_target = T.Compose([self.strong_transforms,
                                     self.normalize])(img, target)
        return t_img, t_target, s_img, s_target


def build_dataset(image_set, domain, args):
    assert image_set in ['train', 'val']
    assert domain in ['src', 'tgt']

    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    # mode = 'instances'
    PATHS = {'cityscapes':
                 {'train': (root / "cityscapes" / "leftImg8bit" / "train",
                            root / "CocoFormatAnnos" / f'cityscapes_train_cocostyle.json'),
                  'val': (root / "cityscapes" / "leftImg8bit" / "val",
                          root / "CocoFormatAnnos" / f'cityscapes_val_cocostyle.json')},
             'foggy_cityscapes':
                 {'train': (root / "foggy_cityscapes" / "leftImg8bit_foggy" / "train",
                            root / "CocoFormatAnnos" / f'cityscapes_foggy_train_cocostyle.json'),
                  'val': (root / "foggy_cityscapes" / "leftImg8bit_foggy" / "val",
                          root / "CocoFormatAnnos" / f'cityscapes_foggy_val_cocostyle.json')},
             'sim10k':
                 {'train': (root / "sim10k" / "JPEGImages",
                            root / "CocoFormatAnnos" / f'sim10k_caronly.json')},
             'city_caronly':
                 {'train': (root / "cityscapes",
                            root / "CocoFormatAnnos" / f'cityscapes_train_caronly_cocostyle.json'),
                  'val': (root / "cityscapes",
                          root / "CocoFormatAnnos" / f'cityscapes_val_caronly_cocostyle.json')},
             'cityscapes_bddfmt':
                 {'train': (root / "cityscapes",
                            root / "CocoFormatAnnos" / f'cityscapes_train_bddfmt_cocostyle.json')},
             'bdd100k':
                 {'train': (root / "bdd100k/images/100k" / "train",
                            root / "CocoFormatAnnos" / f'bdd100k_labels_images_det_coco_train.json'),
                  'val': (root / "bdd100k/images/100k" / "val",
                          root / "CocoFormatAnnos" / f'bdd100k_labels_images_det_coco_val.json')}
             }

    if domain == 'src':
        if args.src_domain == 'coco':
            return build_coco(image_set, args)
        if args.src_domain == 'coco_panoptic':
            from .coco_panoptic import build as build_coco_panoptic
            return build_coco_panoptic(image_set, args)
        if args.src_domain in ['cityscapes', 'sim10k','cityscapes_bddfmt']:
            img_folder, ann_file = PATHS[args.src_domain][image_set]
            transforms = _make_coco_transforms(image_set)
            dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks,
                                    cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                    local_size=get_local_size())
            return dataset

    if domain == 'tgt':
        assert args.tgt_domain is not None, ValueError('Target domain dataset configured in args is None')
        img_folder, ann_file = PATHS[args.tgt_domain][image_set]
        if image_set == 'train':
            teacher_student_dataset = TeacherStudentDataset(img_folder, ann_file, return_masks=args.masks,
                                                            cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                                            local_size=get_local_size())
            return teacher_student_dataset
        if image_set == 'val':
            transforms = _make_coco_transforms(image_set)
            dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks,
                                    cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                    local_size=get_local_size())
            return dataset
