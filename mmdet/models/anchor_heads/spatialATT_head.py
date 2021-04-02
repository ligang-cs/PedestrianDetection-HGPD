import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import xavier_init

from mmdet.core import force_fp32
from ..utils import ConvModule
from ..builder import build_loss
from ..registry import HEADS
from torch.nn.modules.utils import _pair

import numpy as np
import mmcv
import math
import pdb

@HEADS.register_module
class SpatialAttHead(nn.Module):
	""" mask generation for backbone feature"""

	def __init__(self, 
		num_convs,
		in_channel,
		out_channel,
		loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
		conv_cfg=None,
		norm_cfg=None):
		super(SpatialAttHead, self).__init__()
		self.num_convs = num_convs
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg

		self.branch_convs = nn.ModuleList()
		self.branch_convs.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
		if self.num_convs > 0:
			for i in range(self.num_convs):
				conv_in_channel=(
					self.in_channel if i ==0 else self.out_channel)
				self.branch_convs.append(
					ConvModule(
						conv_in_channel,
						self.out_channel,
						3,
						padding=2,
						dilation=2,
						conv_cfg=self.conv_cfg,
						norm_cfg=self.norm_cfg))
		self.conv_logits = nn.Conv2d(self.out_channel, 1, 1)

		self.loss_cls = build_loss(loss_cls)

	def init_weights(self):
		for m in self.branch_convs.modules():
			if isinstance(m, nn.Conv2d):
				xavier_init(m, distribution='uniform')
			elif isinstance(m, _BatchNorm):
				m.eval()

		nn.init.xavier_normal_(self.conv_logits.weight)
		nn.init.constant_(self.conv_logits.bias, -math.log(0.99/0.01))

	def forward(self, x):
		for conv in self.branch_convs:
			x = conv(x)

		mask = self.conv_logits(x)
		return mask

	@force_fp32(apply_to=('mask'))
	def loss(self, mask, gt_mask, gt_mask_weight):
		loss_mask = self.loss_cls(mask.reshape(-1, 1), 
			gt_mask.reshape(-1, 1).long(), 
			gt_mask_weight.reshape(-1, 1),
			avg_factor=torch.sum(gt_mask_weight))
		return loss_mask

	@force_fp32(apply_to=('mask', ))
	def loss_mask_guide(self, mask, gt_mask):
		criteria = nn.BCELoss()
		loss_mask = 0.5 * criteria(mask, gt_mask)
		return loss_mask

	def down_top(self, x, attention_map):
		x = list(x)
		x[0] = x[0] * attention_map
		for ind in range(1, len(x)):
			avg_fea = torch.mean(x[ind-1], dim=1, keepdim=True)
			max_fea,_ = torch.max(x[ind-1], dim=1, keepdim=True)
			concat_fea = torch.cat((avg_fea, max_fea), 1)
			for_att = F.interpolate(torch.sigmoid(self.down_top_conv(concat_fea)), scale_factor=0.5, mode='bilinear')
			x[ind] = x[ind] * for_att + x[ind]
		x = tuple(x)
		return x

	def generate_mask(self, x, gt_bboxes, gt_bboxes_ignore):
		num_img = len(gt_bboxes)
		height, width = x[1].size(-2), x[1].size(-1)
		mask = np.zeros((num_img, 1, height, width))
		mask_ignore = np.ones((num_img, 1, height, width))
		
		for ind in range(num_img):
			gt_bboxes_np = gt_bboxes[ind].cpu().numpy().astype(np.int32)
			gt_bboxes_ignore_np = gt_bboxes_ignore[ind].cpu().numpy().astype(np.int32)
			for box in gt_bboxes_np:
				x1, y1, x2, y2 = np.round(box/16).astype(np.int32)
				mask[ind, 0, y1:y2+1, x1: x2+1] = 1
			for box in gt_bboxes_ignore_np:
				x1, y1, x2, y2 = np.round(box/16).astype(np.int32)
				mask_ignore[ind, 0, y1:y2+1, x1: x2+1] = 0
		gt_masks = torch.from_numpy(mask).float().to(
			x[0].device)
		gt_masks_ignore = torch.from_numpy(mask_ignore).float().to(
			x[0].device)
		return gt_masks, gt_masks_ignore

	def apply_attention(self, attention_map, x):
		x = list(x)
		attention_map = attention_map.sigmoid()
		mask_P2 = F.interpolate(attention_map, scale_factor=2, mode='bilinear')
		mask_P4 = F.interpolate(attention_map, scale_factor=0.5, mode='bilinear')
		x[0] = x[0] * mask_P2
		x[1] = x[1] * attention_map
		x[2] = x[2] * mask_P4
		x = tuple(x)
		return x

	""" code for mask-guide attention"""

	def get_mask(self, sampling_results, gt_masks):
		pos_proposals = [res.pos_bboxes for res in sampling_results]
		pos_assigned_gt_inds = [
			res.pos_assigned_gt_inds for res in sampling_results
		]
		mask_targets = self.mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks)
		return mask_targets

	def mask_target(self, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list):
		mask_targets = map(self.mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list)
		mask_targets = torch.cat(list(mask_targets))
		return mask_targets

	def mask_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
		mask_size = _pair(7)
		num_pos = pos_proposals.size(0)
		mask_targets = []
		proposals_np = pos_proposals.cpu().numpy()
		_, maxh, maxw = gt_masks.shape
		proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw-1)
		proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh-1)
		pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
		for i in range(num_pos):
			gt_mask = gt_masks[pos_assigned_gt_inds[i]]
			bbox = proposals_np[i, :].astype(np.int32)
			x1, y1, x2, y2 = bbox
			w = np.maximum(x2 - x1 + 1, 1)
			h = np.maximum(y2 - y1 + 1, 1)
			target = mmcv.imresize(gt_mask[y1: y1+h, x1: x1+w], mask_size[::-1])
			mask_targets.append(target)
		mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
			pos_proposals.device)
		return mask_targets

