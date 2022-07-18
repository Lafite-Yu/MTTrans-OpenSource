# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_detr import SetCriterion, PostProcess, MomentumUpdator
import copy


def build(args):
    assert args.tgt_domain in ['foggy_cityscapes', 'bdd100k'] if args.src_domain == 'cityscapes' else ['city_caronly']
    num_classes_dataset = {'coco': 91, 'coco_panoptic': 250,
                           'cityscapes': 9, 'sim10k': 2,
                           'foggy_cityscapes': 9, 'city_caronly': 2, 'bdd100k': 9,'cityscapes_bddfmt': 9}
    assert num_classes_dataset[args.src_domain] == num_classes_dataset[args.tgt_domain] or args.tgt_domain is None
    num_classes = num_classes_dataset[args.src_domain]
    device = torch.device(args.device)

    if args.transformer_arch == 'def_detr':
        from .deformable_transformer import build_deformable_transformer
        from .deformable_detr import DeformableDETR
    elif args.transformer_arch == 'sfa':
        from .sfa_transformer import build_sfa as build_deformable_transformer
        from .deformable_detr import SFA as DeformableDETR
    else:
        raise ValueError('args.transformer_arch: ', args.transformer_arch)

    backbone_student = build_backbone(args)
    transformer_student = build_deformable_transformer(args)
    model_student = DeformableDETR(
        backbone_student,
        transformer_student,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage
    )

    backbone_teacher = build_backbone(args)
    transformer_teacher = build_deformable_transformer(args)
    model_teacher = DeformableDETR(
        backbone_teacher,
        transformer_teacher,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage
    )

    if args.masks:
        model_student = DETRsegm(model_student, freeze_detr=(args.frozen_weights is not None))
        model_teacher = DETRsegm(model_teacher, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)

    domain_levels = 1.0
    if args.hda is not None:
        domain_levels += len(args.hda)

    # set domain_loss_coef
    if args.domain_loss_coef is not None:
        args.domain_enc_token_loss_coef = args.domain_loss_coef
        args.domain_enc_query_loss_coef = args.domain_loss_coef
        args.domain_dec_token_loss_coef = args.domain_loss_coef
        args.domain_dec_query_loss_coef = args.domain_loss_coef

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'unsup_loss_ce': args.unsup_cls_loss_coef,
        'unsup_loss_bbox': args.unsup_bbox_loss_coef,
        'unsup_loss_giou': args.unsup_giou_loss_coef,
        'loss_domain_enc_token': args.domain_enc_token_loss_coef * 0.5 / domain_levels,
        'loss_domain_dec_token': args.domain_dec_token_loss_coef * 0.5 / domain_levels,
        'loss_domain_enc_query': args.domain_enc_query_loss_coef * 0.5 / domain_levels,
        'loss_domain_dec_query': args.domain_dec_query_loss_coef * 0.5 / domain_levels,
        'loss_domain_enc_token_t': args.domain_enc_token_loss_coef * 0.5 / domain_levels,
        'loss_domain_dec_token_t': args.domain_dec_token_loss_coef * 0.5 / domain_levels,
        'loss_domain_enc_query_t': args.domain_enc_query_loss_coef * 0.5 / domain_levels,
        'loss_domain_dec_query_t': args.domain_dec_query_loss_coef * 0.5 / domain_levels,
    }
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_domain_enc_prototype'] = args.domain_enc_prototype_loss_coef * 2
    weight_dict['loss_domain_dec_prototype'] = args.domain_dec_prototype_loss_coef * 2
    weight_dict['loss_domain_enc_prototype_t'] = args.domain_enc_prototype_loss_coef * 2
    weight_dict['loss_domain_dec_prototype_t'] = args.domain_dec_prototype_loss_coef * 2

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.transformer_arch == 'sfa':
        losses += ['domain']
        if args.cmt:
            losses += ['cmt']
            cmt_weight_dict = {
                'loss_cmt_cls_js': args.cmt_cls_js_loss_coef,
                'loss_cmt_bbox_l1': args.cmt_bbox_l1_loss_coef,
                # 'loss_cmt_entropy': args.cmt_entrop_loss_coef,
            }
            weight_dict.update(cmt_weight_dict)
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, args.pseudo_label_policy,
                             focal_alpha=args.focal_alpha, cmt_start_epoch=args.cmt_start_epoch)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    momentum_updator = MomentumUpdator(momentum=args.momentum_update_coef,
                                       interval=args.momentum_update_interval,
                                       warm_up=args.momentum_update_warmup,
                                       decay_intervals=args.momentum_update_decay_intervals,
                                       decay_factor=args.momentum_update_decay_factor,
                                       share_query_embeds=args.share_query_embeds)

    # torch.save(model_student, 'SFA.pth')

    return model_student, model_teacher, criterion, postprocessors, momentum_updator
