from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_inside_flags, anchor_target,
                        delta2bbox, force_fp32, ga_loc_target, ga_shape_target,
                        multi_apply, multiclass_nms)
from mmdet.ops import DeformConv, MaskedConv2d
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .anchor_head import AnchorHead
from mmdet.ops import nms
from ..utils import ConvModule
from mmdet.core.bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
import pdb


@HEADS.register_module
class LocalizationHead(AnchorHead):
    def __init__(
        self,
        in_channels,
        num_conv=2,
        num_classes=2,
        feat_channels=256,
        anchor_scales=[8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        anchoring_means=(.0, .0, .0, .0),
        anchoring_stds=(1.0, 1.0, 1.0, 1.0),
        target_means=(.0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox_first=dict(type='SmoothL1Loss', beta=1.0 / 9.0,
                       loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0,
                       loss_weight=1.0)):  # yapf: disable
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_conv = num_conv
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.anchoring_means = anchoring_means
        self.anchoring_stds = anchoring_stds
        self.target_means = target_means
        self.target_stds = target_stds
        self.loc_filter_thr = loc_filter_thr
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        self.num_level_anchors = [131072, 32768, 8192, 2048, 512]

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
        
        # one anchor per location
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_focal_loss = loss_cls['type'] in ['FocalLoss']
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        # build losses
        self.loss_loc = build_loss(loss_loc)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_first = build_loss(loss_bbox_first)
        self.loss_bbox = build_loss(loss_bbox)

        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.conv_loc = nn.Conv2d(self.in_channels, 1, 1)
        self.conv_reg_first = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)
        self.conv_cls = MaskedConv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.conv_reg = MaskedConv2d(self.feat_channels, self.num_anchors * 4,
                                     1)
        self.feature_adaptation = nn.ModuleList()
        for i in range(self.num_conv):
            self.feature_adaptation.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    padding=1))

    def init_weights(self):
        for m in self.feature_adaptation:
            normal_init(m.conv, std=0.01)
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_reg_first, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        loc_pred = self.conv_loc(x)
        bbox_pred_first = self.conv_reg_first(x)

        for conv in self.feature_adaptation:
             x = conv(x)

        # masked conv is only used during inference for speed-up
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return loc_pred, bbox_pred_first, cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_loc_mask(self,
                    loc_preds,
                    featmap_sizes,
                    img_metas,
                    use_loc_filter=False,
                    device='cuda'):

        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
  
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_loc_mask = []
            for i in range(num_levels):
                loc_pred = loc_preds[i][img_id].sigmoid().detach()
                if use_loc_filter:
                    loc_mask = loc_pred >= self.loc_filter_thr
                else:
                    loc_mask = loc_pred >= 0.0
                mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
                mask = mask.contiguous().view(-1)
 
                # anchor_list[img_id][i] = anchor_list[img_id][i][mask]
                multi_level_loc_mask.append(mask)
            loc_mask_list.append(multi_level_loc_mask)
        return loc_mask_list

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor,
                        cfg):
        loss_loc = self.loss_loc(
            loc_pred.reshape(-1, 1),
            loc_target.reshape(-1, 1).long(),
            loc_weight.reshape(-1, 1),
            avg_factor=loc_avg_factor)
        return loss_loc

    def loss_first_single(self, bbox_pred,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_bbox

    @force_fp32(
        apply_to=('bbox_preds', 'loc_preds'))
    def loss_first(self,
             loc_preds,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_labels=None,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = bbox_preds[0].device

        # get loc targets
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.anchor_scales,
            self.anchor_strides,
            center_ratio=cfg.center_ratio,
            ignore_ratio=cfg.ignore_ratio)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # self.num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # get anchor targets
        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos if self.cls_focal_loss else num_total_pos +
            num_total_neg)
        losses_bbox_first = []
        # get bbox regression losses on the  first stage
        for i in range(len(bbox_preds)):
            loss_bbox = self.loss_first_single(
                bbox_preds[i],
                bbox_targets_list[i],
                bbox_weights_list[i],
                num_total_samples=num_total_samples,
                cfg=cfg)
            losses_bbox_first.append(loss_bbox)

        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],
                loc_targets[i],
                loc_weights[i],
                loc_avg_factor=loc_avg_factor,
                cfg=cfg)
            losses_loc.append(loss_loc)
        return dict(
            loss_rpn_bbox_first=losses_bbox_first,
            loss_loc=losses_loc)

    def loss_second_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss_second(self,
             refined_anchors,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_labels=None,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_second(
            refined_anchors,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_second_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def anchor_target_second(self,
                  refined_anchors,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
        num_imgs = len(img_metas)
        assert len(refined_anchors)  == num_imgs

        # anchor number of multi levels

        # concat all level anchors and flags to a single tensor

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
             self.anchor_target_single,
             refined_anchors,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             target_means=target_means,
             target_stds=target_stds,
             cfg=cfg,
             label_channels=label_channels,
             sampling=sampling,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, self.num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, self.num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, self.num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, self.num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def anchor_target_single(self,
                             anchors,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             img_meta,
                             target_means,
                             target_stds,
                             cfg,
                             label_channels=1,
                             sampling=True,
                             unmap_outputs=True):
        if sampling:
            assign_result, sampling_result = assign_and_sample(
                anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
        else:
            bbox_assigner = build_assigner(cfg.assigner)
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            bbox_sampler = PseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, anchors,
                                                  gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                          sampling_result.pos_gt_bboxes,
                                          target_means, target_stds)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @force_fp32(
        apply_to=('bbox_preds'))
    def get_bboxes_first(self,
                   bbox_preds,
                   img_metas,
                   rescale=False):

        num_levels = len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        device = bbox_preds[0].device
        # get guided anchors
        
        anchor, valid_flag_list = self.get_anchors(
            featmap_sizes,
            img_metas,
            device=device)
        result_list = []
            
        for img_id in range(len(img_metas)):
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            anchor_list = [
                anchor[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_first_single(bbox_pred_list,
                                               anchor_list,
                                               img_shape,
                                               scale_factor, rescale)
            result_list.append(proposals)
        return result_list

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'loc_preds'))
    def get_bboxes_second(self,
                   cls_scores,
                   bbox_preds,
                   loc_preds,
                   refined_anchors,
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(
            loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        # get guided anchors
        loc_mask = self.get_loc_mask(
            loc_preds,
            featmap_sizes,
            img_metas,
            use_loc_filter=not self.training,
            device=device)
        num_level_anchors = [131072, 32768, 8192, 2048, 512]
        refined_anchors = images_to_levels(refined_anchors, num_level_anchors)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.training:
                anchor_list = [
                    refined_anchors[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                anchor_list = [
                    refined_anchors[i].detach() for i in range(num_levels)
                ]

            loc_mask_list = [
                loc_mask[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_second_single(cls_score_list, bbox_pred_list,
                                               anchor_list,
                                               loc_mask_list, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_first_single(self,
                      bbox_preds,
                      mlvl_anchors,
                      img_shape,
                      scale_factor,
                      rescale=False):
        mlvl_proposals = []
        for idx in range(len(bbox_preds)):
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,4)
            assert anchors.size()[-2:] == rpn_bbox_pred.size()[-2:]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        return proposals

    def get_bboxes_second_single(self,
                      cls_scores,
                      bbox_preds,
                      mlvl_anchors,
                      mlvl_masks,
                      img_shape,
                      scale_factor,
                      cfg,
                      rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            mask = mlvl_masks[idx]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
      
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            anchors = anchors[mask]
            scores = scores[mask]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,4)[mask, :]
            if scores.dim() == 0:
                rpn_bbox_pred = rpn_bbox_pred.unsqueeze(0)
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            # filter out too small bboxes
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                heatmap = heatmap[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            # proposals_heatmap = torch.cat([proposals, heatmap.unsqueeze(-1)], dim=-1)
            # NMS in current level
            # proposals_heatmap, nms_inds = nms(proposals_heatmap, cfg.nms_thr)
            proposals, _ = nms(proposals, cfg.nms_thr)
            # proposals = proposals[nms_inds]
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            # NMS across multi levels
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets