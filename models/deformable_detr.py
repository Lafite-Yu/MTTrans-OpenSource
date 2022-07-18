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
import torchvision
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)
from .sfa_utils import grad_reverse
from .gcn import GraphConvolution
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks,
                                                                                                            pos,
                                                                                                            query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SFA(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # domain discriminator (simple MLP)
        self.domain_pred_enc = MLP(hidden_dim, hidden_dim, output_dim=2, num_layers=3)
        self.domain_pred_dec = MLP(hidden_dim, hidden_dim, output_dim=2, num_layers=3)

        self.domain_prototype_pred_dec = MLP(hidden_dim, hidden_dim, output_dim=2, num_layers=3)
        self.domain_prototype_dec_gcn = GraphConvolution(hidden_dim, hidden_dim)
        self.dec_prototype_nums = 9
        self.dec_feature_projection_before_gcn = MLP(300, 300, self.dec_prototype_nums, 2)

        # self.domain_category_prototype_dec = torch.zeros((1,num_classes, hidden_dim), requires_grad=True)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            self.domain_dec = nn.Embedding(1, hidden_dim * 2)
        # multiple domain queries can actually be adopted, but we use only one for demonstration
        self.domain_enc = nn.Embedding(1, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor, eta: float = 1.0):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print('============================BEGIN OF ITER==============================')
        # print('input shape:', samples.decompose()[0].shape, samples.decompose()[1].shape)
        # print('backbone output length:', len(features))
        # print('backbone output[0] shape:', features[0].decompose()[0].shape, features[0].decompose()[1].shape)
        # print('backbone output[1] shape:', features[1].decompose()[0].shape, features[1].decompose()[1].shape)
        # print('backbone output[2] shape:', features[2].decompose()[0].shape, features[2].decompose()[1].shape)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # print('srcs length:', len(srcs))
        # print('src[0] shape:', srcs[0].shape)
        # print('src[1] shape:', srcs[1].shape)
        # print('src[2] shape:', srcs[2].shape)
        # print('src[3] shape:', srcs[3].shape)
        # print('masks length:', len(masks), ' , shape of masks[0]:', masks[0].shape)
        # print('pos length:', len(pos), ' , shape of src[0]:', pos[0].shape)
        # print('===Begin of Transformer')

        query_embeds = None
        domain_dec = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            domain_dec = self.domain_dec.weight
        domain_enc = self.domain_enc.weight
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
        #     self.transformer(srcs, masks, pos, query_embeds)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, \
        inter_domain_enc, inter_domain_dec, inter_memory, inter_object_query = \
            self.transformer(srcs, masks, pos, query_embeds, domain_enc, domain_dec)

        # print('===End of Transformer')
        # print('Transformer output hs:', hs.shape)
        # print('Transformer output inter_memory:', inter_memory.shape)
        # print('Transformer encoder output inter_memory[-1]:', inter_memory[-1].shape)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # print('DETR decoders output outputs_class:', outputs_class.shape)
        # print('DETR decoders output outputs_coord:', outputs_coord.shape)
        # print('DETR last decoder output hs[-1]:', hs[-1].shape)
        # print('DETR last output outputs_class[-1]:', outputs_class[-1].shape)
        # print('DETR last output outputs_coord[-1]:', outputs_coord[-1].shape)
        # print('============================END OF ITER==============================\n')

        # use different loop for domain_enc and domain_dec
        # obtain encoder domain outputs of each level
        outputs_domains_enc = []
        assert len(inter_domain_enc.shape) == 4 and len(inter_memory.shape) == 4
        assert inter_domain_enc.shape[0] == inter_memory.shape[0]
        for lvl in range(inter_domain_enc.shape[0]):  # 1 or 6 level
            # obtain domain predictions for both token and domain query
            domain_pred_enc = self.domain_pred_enc(grad_reverse(
                torch.cat([inter_memory[lvl], inter_domain_enc[lvl]], dim=1), eta=eta
            ))
            outputs_domains_enc.append(domain_pred_enc)
        outputs_domains_enc = torch.stack(outputs_domains_enc)

        # obtain decoder domain outputs of each level
        outputs_domains_dec = []
        assert len(inter_domain_dec.shape) == 4 and len(inter_object_query.shape) == 4
        assert inter_domain_dec.shape[0] == inter_object_query.shape[0]
        for lvl in range(inter_domain_dec.shape[0]):
            # obtain domain predictions for both token and domain query
            domain_pred_dec = self.domain_pred_dec(grad_reverse(
                torch.cat([inter_object_query[lvl], inter_domain_dec[lvl]], dim=1), eta=eta
            ))
            outputs_domains_dec.append(domain_pred_dec)
        outputs_domains_dec = torch.stack(outputs_domains_dec)

        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coords[-1]
        batch_size = pred_logits.shape[0]

        # Domain Prototype Alignment for Encoder
        outputs_prototype_enc = None

        # Domain Prototype Alignment for Decoder
        decoder_outputs = hs[-1]
        # 将每张图片的300个object proposal的特征通过一个Linear，降维为9个feature，
        #  构建dual-level graph，通过GraphConvolution，将被aggregated的9个feature的特征的平均值作为1个prototype
        projected_decoder_outputs = self.dec_feature_projection_before_gcn(
            decoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        graph_input = torch.concat([projected_decoder_outputs, decoder_outputs], dim=1)
        adj_mask = torch.concat([torch.ones(batch_size, self.dec_prototype_nums, 300+self.dec_prototype_nums,
                                            device=decoder_outputs.device),
                                 torch.zeros(batch_size, 300, 300+self.dec_prototype_nums,
                                             device=decoder_outputs.device)], dim=1)
        graph_output = self.domain_prototype_dec_gcn(graph_input, adj_mask)
        decoder_outputs_prototype = graph_output[:, :, :self.dec_prototype_nums].mean(dim=-1)
        outputs_prototype_dec = self.domain_prototype_pred_dec(grad_reverse(decoder_outputs_prototype, eta=eta))
        # shape of outputs_prototype_dec: (N, 2)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'pred_domain_enc': outputs_domains_enc[-1], 'pred_domain_dec': outputs_domains_dec[-1],
               'pred_domain_prototype_enc': outputs_prototype_enc, 'pred_domain_prototype_dec': outputs_prototype_dec,
               'backbone_outputs': features, 'encoder_inputs': [srcs, masks],
               'encoder_outputs': inter_memory[-1], 'decoder_outputs': hs[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if len(outputs_domains_enc) > 1:
            out['aux_domain'] = self._set_aux_loss_domain(outputs_domains_enc, outputs_domains_dec)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @torch.jit.unused
    def _set_aux_loss_domain(self, outputs_domains_enc, outputs_domains_dec):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_domain_enc': a, 'pred_domain_dec': b}
                for a, b in zip(outputs_domains_enc[:-1], outputs_domains_dec[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, pseudo_label_policy, focal_alpha=0.25, cmt_start_epoch=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

        self.pseudo_label_policy = pseudo_label_policy

        self.dt_loss = nn.CrossEntropyLoss()
        self.dq_loss = nn.CrossEntropyLoss()
        self.dp_loss = nn.CrossEntropyLoss()
        self.cmt_start_epoch = cmt_start_epoch
        self.weight_dict_origin = copy.deepcopy(weight_dict)

    def set_epoch(self, epoch):
        if epoch < self.cmt_start_epoch:
            for k, v in self.weight_dict.items():
                if 'cmt' in k:
                    self.weight_dict[k] = 0
        else:
            self.weight_dict = copy.deepcopy(self.weight_dict_origin)
        print(self.weight_dict)

    def loss_cmt(self, outputs, targets, indices, num_boxes, domain_label):
        """
        outputs['pred_logits']  # (B, 300, cls+1), background id 0
        # outputs['pred_boxes']  # (B, 300, 4)
        """
        assert domain_label == 1, 'consistent matching loss is only for target domain'
        assert 'aux_outputs' in outputs, "require auxiliary outputs for consistent matching"
        assert len(outputs['pred_logits'].shape) == 3

        pred_logits_all = []
        pred_boxes_all = []
        for i, aux_outputs in enumerate(outputs['aux_outputs']):
            pred_logits_all.append(
                aux_outputs['pred_logits'].flatten(0, 1)
            )  # (B*300, cls+1)
            pred_boxes_all.append(aux_outputs['pred_boxes'].flatten(0, 1))
        pred_logits_all.append(outputs['pred_logits'].flatten(0, 1))
        pred_boxes_all.append(outputs['pred_boxes'].flatten(0, 1))
        pred_logits_all = torch.stack(pred_logits_all)
        pred_boxes_all = torch.stack(pred_boxes_all)

        # adopt output mean as anchor
        logit = torch.mean(pred_logits_all, dim=0).detach()
        boxes = torch.mean(pred_boxes_all, dim=0).detach()

        # # entropy loss on anchor (not detached)
        # logit_mean = torch.mean(pred_logits_all, dim=0)
        # pred_p = F.softmax(logit_mean, dim=-1)
        # loss_cmt_entropy = -1 * torch.sum(pred_p * F.log_softmax(logit_mean, dim=-1)) / pred_p.shape[0]

        # cmt loss for cls
        loss_cmt_cls_js = 0.
        # use log_softmax instead of softmax().log(), for stability.
        for i, pred_logits_tmp in enumerate(pred_logits_all):
            prob_mean = 0.5 * (F.softmax(pred_logits_tmp, dim=-1) + F.softmax(logit, dim=-1))
            loss_cmt_cls_js += 0.5 * (
                    F.kl_div(F.log_softmax(logit, dim=-1), prob_mean) +
                    F.kl_div(F.log_softmax(pred_logits_tmp, dim=-1), prob_mean)
            )
            # loss_cmt_cls_js += 0.5 * (
            #     F.kl_div(F.log_softmax(logit, dim=-1), F.softmax(pred_logits_tmp, dim=-1)) +
            #     F.kl_div(F.log_softmax(pred_logits_tmp, dim=-1), F.softmax(logit, dim=-1))
            # )

        # cmt loss for bbox
        loss_cmt_bbox_l1 = 0.
        for i, pred_boxes_tmp in enumerate(pred_boxes_all):
            # apply l1 and giou loss
            loss_cmt_bbox_l1 += F.l1_loss(boxes, pred_boxes_tmp)
        losses = {
            'loss_cmt_cls_js': loss_cmt_cls_js / len(pred_boxes_all),
            'loss_cmt_bbox_l1': loss_cmt_bbox_l1 / len(pred_boxes_all),
            # 'loss_cmt_entropy': loss_cmt_entropy,
        }
        # if domain_label == 1:
        #     losses = {k + f'_t': v for k, v in losses.items()}
        return losses

    def loss_domains(self, outputs, targets, indices, num_boxes, domain_label):
        """Classification loss (NLL)
        """
        assert 'pred_domain_enc' in outputs and 'pred_domain_dec' in outputs
        domain_label_list = [domain_label for _ in range(outputs['pred_domain_enc'].shape[0])]
        domain_pred_enc = outputs['pred_domain_enc']  # (B, len_enc+1, 2), +1 is for domain query
        domain_pred_dec = outputs['pred_domain_dec']  # (B, len_enc+1, 2)
        B, len_enc, len_dec = domain_pred_enc.shape[0], domain_pred_enc.shape[1] - 1, domain_pred_dec.shape[1] - 1
        domain_pred_enc_token, domain_pred_enc_query = torch.split(domain_pred_enc, len_enc, dim=1)
        domain_pred_dec_token, domain_pred_dec_query = torch.split(domain_pred_dec, len_dec, dim=1)

        domain_pred_enc_token = domain_pred_enc_token.flatten(0, 1)  # (B*len_enc, 2)
        domain_pred_enc_query = domain_pred_enc_query.squeeze(1)  # (B, 2)

        domain_pred_dec_token = domain_pred_dec_token.flatten(0, 1)  # (B*len_dec, 2)
        domain_pred_dec_query = domain_pred_dec_query.squeeze(1)  # (B, 2)

        # make domain label for domain_pred_enc_token, domain_pred_dec_token, and domain_query
        domain_label_query = torch.tensor(domain_label_list, dtype=torch.long,
                                          device=domain_pred_enc_query.device)  # (B,)
        domain_label_enc_token = domain_label_query[:, None].expand(B, len_enc).flatten(0, 1)
        domain_label_dec_token = domain_label_query[:, None].expand(B, len_dec).flatten(0, 1)

        # cross-entropy
        loss_domain_enc_token = self.dt_loss(domain_pred_enc_token, domain_label_enc_token)
        loss_domain_dec_token = self.dt_loss(domain_pred_dec_token, domain_label_dec_token)
        loss_domain_enc_query = self.dq_loss(domain_pred_enc_query, domain_label_query)
        loss_domain_dec_query = self.dq_loss(domain_pred_dec_query, domain_label_query)
        # torch.save([domain_pred_enc_token, domain_label_enc_token], 'debug_domain_enc_token.pth')
        # torch.save([domain_pred_enc_query, domain_label_query], 'debug_domain_enc_query.pth')

        losses = {
            'loss_domain_enc_token': loss_domain_enc_token,
            'loss_domain_dec_token': loss_domain_dec_token,
            'loss_domain_enc_query': loss_domain_enc_query,
            'loss_domain_dec_query': loss_domain_dec_query
        }
        # print('outputs', outputs)
        # Domain Prototype Enc
        if 'pred_domain_prototype_enc' in outputs:
            pred_domain_prototype_enc = outputs['pred_domain_prototype_enc']
            if pred_domain_prototype_enc is not None:
                assert len(pred_domain_prototype_enc.shape) == 2
                domain_prototype_enc_label = torch.ones(pred_domain_prototype_enc.shape[:1], dtype=torch.long,
                                                        device=pred_domain_prototype_enc.device) * domain_label
                loss_domain_enc_prototype = self.dp_loss(pred_domain_prototype_enc, domain_prototype_enc_label)
                losses['loss_domain_enc_prototype'] = loss_domain_enc_prototype
                # torch.save([pred_domain_prototype_enc, domain_prototype_enc_label, loss_domain_enc_prototype], 'domain_enc_prototype_losses.pth')


        # Domain Prototype Dec
        if 'pred_domain_prototype_dec' in outputs:
            pred_domain_prototype_dec = outputs['pred_domain_prototype_dec']
            if pred_domain_prototype_dec is not None:
                assert len(pred_domain_prototype_dec.shape) == 2
                domain_prototype_dec_label = torch.ones(pred_domain_prototype_dec.shape[:1], dtype=torch.long,
                                                        device=pred_domain_prototype_dec.device) * domain_label
                loss_domain_dec_prototype = self.dp_loss(pred_domain_prototype_dec, domain_prototype_dec_label)
                losses['loss_domain_dec_prototype'] = loss_domain_dec_prototype

                # torch.save([pred_domain_prototype_dec, domain_prototype_dec_label, loss_domain_dec_prototype], 'domain_dec_prototype_losses.pth')
                # print(loss_domain_dec_prototype, pred_domain_prototype_dec, domain_prototype_dec_label)


        if domain_label == 1:
            losses = {k + f'_t': v for k, v in losses.items()}

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def unsup_loss_labels(self, outputs_student, outputs_teacher):
        assert 'pred_logits' in outputs_student and 'pred_logits' in outputs_teacher
        assert outputs_student['pred_logits'].shape == outputs_teacher['pred_logits'].shape
        logits_shape = outputs_student['pred_logits'].shape
        student_logits = outputs_student['pred_logits'].reshape(-1, logits_shape[-1]).sigmoid()
        teacher_logits = outputs_teacher['pred_logits'].reshape(-1, logits_shape[-1]).sigmoid()

        # print('teacher == student: ', torch.all(student_logits == teacher_logits))
        # print('student == student: ', torch.all(student_logits == student_logits))
        # print('student_logits', student_logits.shape)
        # print('teacher_logits',teacher_logits.shape)
        # print('student_max', torch.max(student_logits))
        # print('teacher_max', torch.max(teacher_logits))

        ce_loss = F.cross_entropy(student_logits, teacher_logits)
        losses = {'unsup_loss_ce': ce_loss}
        return losses

    def unsup_loss_boxes(self, outputs_student, outputs_teacher):
        assert 'pred_boxes' in outputs_student and 'pred_boxes' in outputs_teacher
        assert outputs_student['pred_boxes'].shape == outputs_teacher['pred_boxes'].shape
        boxes_shape = outputs_student['pred_boxes'].shape
        student_boxes = outputs_student['pred_boxes'].reshape(-1, boxes_shape[-1])
        teacher_boxes = outputs_teacher['pred_boxes'].reshape(-1, boxes_shape[-1])

        # idx = self._get_src_permutation_idx(indices)
        # src_boxes = outputs['pred_boxes'][idx]
        # target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(student_boxes, teacher_boxes)

        losses = {}
        losses['unsup_loss_bbox'] = loss_bbox.sum()

        student_boxes = box_ops.box_cxcywh_to_xyxy(student_boxes)
        teacher_boxes = box_ops.box_cxcywh_to_xyxy(teacher_boxes)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(student_boxes, teacher_boxes))
        losses['unsup_loss_giou'] = loss_giou.sum() / len(outputs_student)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'domain': self.loss_domains,
            'cmt': self.loss_cmt,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, domain_adversarial, domain_label=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             loss_policy: decided by how the pseudo label is produced. Should be same with args.pseudo_label_policy
             domain_label: `0` for source domain and `1` for target domain
        """
        assert (domain_adversarial and domain_label is not None) or (not domain_adversarial and domain_label is None)
        losses = {}
        # print('output keys: ', outputs.keys())
        if self.pseudo_label_policy == 'traditional':
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                if domain_label == 0 and loss == 'cmt':
                    continue
                if loss in ['domain', 'cmt']:
                    if domain_label is None:
                        continue
                    kwargs = {'domain_label': domain_label}
                else:
                    kwargs = {}
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == 'cmt':
                            # cmt loss is only computed once in loss_cmt
                            continue
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        if loss == 'domain':
                            # aux domain loss is not handled here, as the number of aux can be different
                            assert 'pred_domain_enc' not in aux_outputs.keys()
                            assert 'pred_domain_dec' not in aux_outputs.keys()
                            continue
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            if 'aux_domain' in outputs and domain_adversarial:
                assert domain_label is not None
                for i, aux_domain in enumerate(outputs['aux_domain']):
                    loss = 'domain'
                    kwargs = {'domain_label': domain_label}
                    l_dict = self.get_loss(loss, aux_domain, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # SFA for the two-stage variant is also
            # 原始的SFA中这部分只对supervised loss计算
            if 'enc_outputs' in outputs:
                enc_outputs = outputs['enc_outputs']
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == 'domain':
                        # enc_output is not supervised by domain loss
                        assert 'pred_domain_enc' not in aux_outputs.keys()
                        assert 'pred_domain_dec' not in aux_outputs.keys()
                        continue
                    if loss == 'cmt':
                        # 在原始的SFA中，cmt loss不会出现在supervised loss中
                        continue
                    l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if self.pseudo_label_policy == 'by_consistency':
            if domain_label is not None:
                raise NotImplementedError(
                    'Mean teacher consistency loss is not implemented for domain adversarial learning.')
            losses.update(self.unsup_loss_labels(outputs, targets))
            losses.update(self.unsup_loss_boxes(outputs, targets))
        assert self.pseudo_label_policy in ['traditional', 'by_consistency'], \
            'Loss policy should consist with args.pseudo_label_policy'
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MomentumUpdator():
    def __init__(
            self,
            momentum=0.999,
            interval=1,
            warm_up=100,
            decay_intervals=None,
            decay_factor=0.1,
            share_query_embeds=True
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and (interval > 0 or interval == -1)
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor
        self.curr_step = 0
        self.share_query_embeds = share_query_embeds

    def update(self, model_student, model_teacher, distributed):
        """Update ema parameter every self.interval iterations."""
        if self.interval != -1 and self.curr_step % self.interval != 0:
            return
        # if is_module_wrapper(model):
        #     model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (self.curr_step + 1 + self.warm_up)
        )
        # runner.log_buffer.output["ema_momentum"] = momentum
        if distributed:
            model_student = model_student.module
            model_teacher = model_teacher.module
        self.__momentum_update(model_student, model_teacher, momentum)

    def after_every_update_interval(self):
        self.curr_step += 1
        self.momentum_coef_decay()

    def before_run(self, model_student, model_teacher, distributed):
        if distributed:
            model_student = model_student.module
            model_teacher = model_teacher.module
        self.__momentum_update(model_student, model_teacher, 0)

    def momentum_coef_decay(self):
        if self.decay_intervals is None:
            return
        else:
            from bisect import bisect_right
            self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
                self.decay_intervals, self.curr_step
            )

    def resume_step(self, curr_step):
        self.curr_step = curr_step
        self.momentum_coef_decay()

    def __momentum_update(self, model_student, model_teacher, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
                model_student.named_parameters(), model_teacher.named_parameters()
        ):
            assert src_name == tgt_name
            if src_name.startswith('query_embed') and self.share_query_embeds:
                tgt_parm.data = src_parm.data
            else:
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
