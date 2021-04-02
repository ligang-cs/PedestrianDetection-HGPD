import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import xavier_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.core import delta2bbox
from mmdet.ops import nms, DeformConv
from ..registry import HEADS
from ..utils import ConvModule, build_conv_layer
from .anchor_head import AnchorHead
import pdb


@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        # RPN head at stage 1
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        
        # Layers of feature fusion modules
        self.feature_conv1 = nn.Conv2d(
            self.in_channels, 256, 3, padding=1)
        self.feature_conv2 = nn.Conv2d(
            256, 256, 3, padding=1)
        self.feature_conv3 = nn.Conv2d(
            self.in_channels, 256, 3, padding=1)
        self.feature_conv4 = nn.Conv2d(
            256, self.in_channels, 3, padding=1)

        # RPN head at stage 2
        self.rpn_conv_stage2 = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls_stage2 = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg_stage2 = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.feature_conv1, std=0.01)
        normal_init(self.feature_conv2, std=0.01)
        normal_init(self.feature_conv3, std=0.01)
        normal_init(self.feature_conv4, std=0.01)

        normal_init(self.rpn_conv_stage2, std=0.01)
        normal_init(self.rpn_cls_stage2, std=0.01)
        normal_init(self.rpn_reg_stage2, std=0.01)

        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)

        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def two_step_regress_single(self, conv4, x):
        x = F.relu(self.feature_conv1(x), 
                    inplace=True)
        x = self.feature_conv2(x)
        conv4 = self.feature_conv3(conv4)
        x_add = x + conv4
        x_add = F.relu(x_add, inplace=True)
        x_add = F.relu(self.feature_conv4(x_add),
                    inplace=True)
        x_head = F.relu(self.rpn_conv_stage2(x_add),
                     inplace=True)
        rpn_cls_score = self.rpn_cls_stage2(x_head)
        rpn_bbox_pred = self.rpn_reg_stage2(x_head)

        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             is_stage1=False,
             gt_bboxes_ignore=None,
             refined_bboxes=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore,
            refined_bboxes=refined_bboxes)
        if is_stage1:
            return dict(
                loss_rpn_bbox_stage1=losses['loss_bbox'])
        else:
            return dict(
                loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
