import numpy as np 
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule

@HEADS.register_module
class DoubleRegression(nn.Module):
	""" input: proposals 
		output:  refined proposals"""
	def __init__(self,
		in_channels,
		out_channels,
		num_prediction,
		loss_regression=dict(
			type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
		anchor_stride=[4, 8, 16, 32, 64]):
	super(DoubleRegression, self).__init__()
	self.in_channels = in_channels
	self.out_channels = out_channels
	self.num_prediction = num_prediction
	self.anchor_stride = anchor_stride

	self.loss_regression = build_loss(loss_regression)

	self.feature_adaptation = nn.ModuleList()
	self.feature_adaptation.append(
		ConvModule(
			self.in_channels,
			self.out_channels,
			3,
			padding=1))
	self.regression_conv = nn.Conv2d(self.out_channels,
                                  self.num_prediction, 1)

	def init_weights(self):
		for m in self.feature_adaptation:
			normal_init(m.conv, std=0.01)
		normal_init(self.regression_conv, std=0.01)

	def forward(self, x):
		for conv in self.feature_adaptation:
			x = conv(x)
		bbox_pred = self.regression_conv(x)
		return bbox_pred

	@force_fp32(apply_to=('bbox_preds'))
	def loss(self,
			proposal_list,
			bbox_preds,
			gt_bboxes,
			anchor_list,
			img_metas,
			cfg,
			gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]

		cls_reg_targets = self.anchor_target(
				proposal_list,
				anchor_list,
				gt_bboxes,
				img_metas,
				featmap_sizes,
				self.target_means,
				self.target_stds,
				cfg,
				gt_bboxes_ignore_list=gt_bboxes_ignore,
				sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (bbox_targets_list, bbox_weights_list,
				num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
       	losses_bbox = multi_apply(
            self.loss_single,
            bbox_preds,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_bbox=losses_bbox)

    def loss_single(self, bbox_pred, bbox_targets, bbox_weights, num_total_samples, cfg):
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_regression(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_bbox

	def anchor_target(self,
			proposal_list,
			anchor_list,
			gt_bboxes_list,
			img_metas,
			featmap_sizes,
			target_means,
			target_stds,
			cfg,
			gt_bboxes_ignore_list=None,
			sampling=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
		num_imgs = len(img_metas)
		assert len(proposal_list) == len(anchor_list) == num_imgs

	    # anchor number of multi levels
		num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
	    # concat all level anchors and flags to a single tensor

		for i in range(num_imgs):
			anchor_list[i] = torch.cat(anchor_list[i])
	 
		num_total_anchors = anchor_list[0].size(0)
	    # compute targets for each image
		if gt_bboxes_ignore_list is None:
			gt_bboxes_ignore_list = [None for _ in range(num_imgs)]

	    (all_bbox_targets, all_bbox_weights,
	     pos_inds_list, neg_inds_list) = multi_apply(
	         self.anchor_target_single,
	         proposal_list,
	         gt_bboxes_list,
	         gt_bboxes_ignore_list,
	         img_metas,
	         target_means=target_means,
	         target_stds=target_stds,
	         cfg=cfg,
	         sampling=sampling)

		bbox_targets_list = []
		bbox_weights_list = [] 
	    for i in range(num_imgs):
			proposals = proposals_list[i]
			proposal_flags = torch.zeros_like(anchor_list[i])
			anchor_index = torch.zeros_like(proposals)
			target_lvls = self.map_roi_levels(proposals, 5) 
			for ind, box in enumerate(proposals):
				target_lvl = target_lvls[ind]
				h, w = featmap_sizes[ind]
				x_center, y_center = torch.round((box[0]+box[2])*0.5), 
					torch.round((box[1]+box[3])*0.5)
				x_fea, y_fea = torch.round(x_center/self.anchor_stride[target_lvl]), 
					torch.round(y_center/self.anchor_stride[target_lvl])
				if target_lvl == 0:
					anchor_index[ind] = (y_fea*w + x_center).long() 
				else:
					anchor_index[ind] = (y_fea*w + x_center).long() + num_level_anchors[target_lvl-1]
			for j in anchor_index:
				proposal_flags[j] = 1
			bbox_targets = self.unmap(all_bbox_targets[i], num_total_anchors, proposal_flags)
			bbox_weights = self.unmap(all_bbox_weights[i], num_total_anchors, proposal_flags)
			bbox_targets_list.append(bbox_targets)
			bbox_weights_list.append(bbox_weights)

	    # sampled anchors of all images
		num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
		num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
		# split targets to a list w.r.t. multiple levels
		# labels_list = images_to_levels(all_labels, num_level_anchors)
		# label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
		bbox_targets_list = self.images_to_levels(all_bbox_targets, num_level_anchors)
		bbox_weights_list = self.images_to_levels(all_bbox_weights, num_level_anchors)
		return (bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)


	def images_to_levels(self, target, num_level_anchors):
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

	def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
	if data.dim() == 1:
		ret = data.new_full((count, ), fill)
		ret[inds.type(torch.bool)] = data
	else:
		new_size = (count, ) + data.size()[1:]
		ret = data.new_full(new_size, fill)
		ret[inds.type(torch.bool), :] = data
	return ret

	def anchor_target_single(self,
						 proposals,
						 gt_bboxes,
						 gt_bboxes_ignore,
						 img_meta,
						 target_means,
						 target_stds,
						 cfg,
						 sampling=True):
		if sampling:
			assign_result, sampling_result = assign_and_sample(
					proposals, gt_bboxes, gt_bboxes_ignore, None, cfg)
		else:
			bbox_assigner = build_assigner(cfg.assigner)
			assign_result = bbox_assigner.assign(anchors, gt_bboxes,
					gt_bboxes_ignore, gt_labels)
			bbox_sampler = PseudoSampler()
			sampling_result = bbox_sampler.sample(assign_result, anchors,
					gt_bboxes)

		num_valid_anchors = proposals.shape[0]
		bbox_targets = torch.zeros_like(proposals)
		bbox_weights = torch.zeros_like(proposals)
		# label_weights = proposals.new_zeros(num_valid_anchors, dtype=torch.float)

		pos_inds = sampling_result.pos_inds
		neg_inds = sampling_result.neg_inds

		if len(pos_inds) > 0:
		pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
			sampling_result.pos_gt_bboxes,
			target_means, target_stds)
		bbox_targets[pos_inds, :] = pos_bbox_targets
		bbox_weights[pos_inds, :] = 1.0

		return (bbox_targets, bbox_weights, pos_inds, neg_inds)

	def map_roi_levels(self, rois, num_levels):
		scale = torch.sqrt(
			(rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
		target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
		target_lvls = target_lvls.clamp(min=0, max=num_levels-1).long()
		return target_lvls





