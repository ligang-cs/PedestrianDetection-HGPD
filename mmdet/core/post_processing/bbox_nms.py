import torch

from mmdet.ops.nms import nms_wrapper
import pdb


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        """
        anchor_stride = [4, 8, 16, 32, 64]
        scale = torch.sqrt((_bboxes[:, 3] - _bboxes[:, 1] + 1) * (_bboxes[:, 2] - _bboxes[:, 0] + 1))
        min_anchor_size = scale.new_full(
            (1, ), float(8 * 4))
        target_lvls = torch.floor(
            torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=4).long() 
        heatmaps = _scores.new_full(_scores.size(), 0)
        for ind, box in enumerate(_bboxes):
            num_lvl = target_lvls[ind]
            x1, y1, x2, y2 = torch.round(box / anchor_stride[num_lvl]).long()
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            heatmap = loc_preds[num_lvl].sigmoid()[:, :, y1 : y2 + 1, x1 : x2 + 1]
            heatmap = heatmap >= 0.2
            heatmap_mean = torch.sum(heatmap).float() / area
            heatmaps[ind] = heatmap_mean
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        _scores = 0.9*_scores + 0.1*heatmaps
        """
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        # cls_dets_loc = torch.cat([_bboxes, heatmaps[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        # cls_dets_nms, nms_ind = nms_op(cls_dets_loc, **nms_cfg_)
        # cls_dets = cls_dets[nms_ind]
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
