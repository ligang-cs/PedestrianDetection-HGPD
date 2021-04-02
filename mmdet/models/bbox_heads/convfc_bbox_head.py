import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
import torch
from ..utils.norm import build_norm_layer
from mmcv.cnn import constant_init
import random
from mmdet.core.bbox.geometry import bbox_overlaps
import numpy as np
import pdb


@HEADS.register_module
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        self.concat_fc = nn.Linear(2048, 1024)

        # self-attention module
        self.attention_fc1 = nn.ModuleList()
        self.attention_fc1.append(
            nn.Linear(1024, 512))
        self.attention_fc1.append(
            nn.Linear(512, 128))
        self.attention_logits = nn.Linear(128, 1)
        
        self.part_out = nn.Linear(3072, 1024)

        # affinity module
        self.affinity_fc1 = nn.ModuleList()
        self.affinity_fc1.append(
            nn.Linear(1024, 64))
        self.affinity_fc2 = nn.ModuleList()
        self.affinity_fc2.append(
            nn.Linear(1024, 64))
        self.weight_fc = nn.Linear(64, 1)

        self.norm_name, norm = build_norm_layer(dict(type='BN'), 3)
        self.add_module(self.norm_name, norm)

        self.parameter_matrix = nn.Linear(1024, 1024)
        self.parameter_matrix2 = nn.Linear(1024, 1024)
     
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
  
        for m in self.attention_fc1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.attention_logits.weight)
        nn.init.constant_(self.attention_logits.bias, 0)

        for m in self.affinity_fc1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.affinity_fc2.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.weight_fc.weight)
        nn.init.constant_(self.weight_fc.bias, 0)

        constant_init(self.norm, 1, bias=0)

        nn.init.xavier_uniform_(self.part_out.weight)
        nn.init.constant_(self.part_out.bias, 0)

        nn.init.xavier_uniform_(self.concat_fc.weight)
        nn.init.constant_(self.concat_fc.bias, 0)

        nn.init.xavier_uniform_(self.parameter_matrix.weight)
        nn.init.constant_(self.parameter_matrix.bias, 0)

        nn.init.xavier_uniform_(self.parameter_matrix2.weight)
        nn.init.constant_(self.parameter_matrix2.bias, 0)

    def forward(self, x, rois, num_proposal_list):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            
            for ind, fc in enumerate(self.shared_fcs):
                x = self.relu(fc(x))
                if ind == 0:
                    # intra-proposal
                    intra_feature = self.intra_graph(x)
                    # inter-proposal
                    x_full = x[:, 0]
                    x_neighbour = self.inter_graph(x_full, rois, num_proposal_list)
                    inter_feature = 0.9*x_full + 0.1*x_neighbour
                    # perform another fc layer on full-body features for better  regression
                    x = x_full

        x_reg = x
        x_concat = torch.cat((inter_feature, intra_feature), 1)
        x_cls = self.relu(self.concat_fc(x_concat))

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred

    def intra_graph(self, x):
        # Inra-proposal Graph
        x_original = x[:, 1:]
        full_body = x[:, 0]
        body_part = x[:, 1:]
        body_part_ = body_part.clone()
       
        # affinity module
        for fc in self.affinity_fc1:
            full_body = self.relu(fc(full_body))
        for fc in self.affinity_fc2:
            body_part = self.relu(fc(body_part))
        num_sample, num_body, feat_dim = body_part.size()
        full_body = full_body[:, None, None, :].expand((
            num_sample, 
            num_body, num_body, 
            feat_dim))
        body_part = body_part[:, None, :, :].expand((
            num_sample, 
            num_body, num_body, 
            feat_dim))
        affinity_matrix = body_part * full_body
        affinity_matrix = self.weight_fc(self.norm(affinity_matrix)).sigmoid()
       
        # self-attention
        for fc in self.attention_fc1:
            body_part_ = self.relu(fc(body_part_))
        body_part_value = self.attention_logits(body_part_).sigmoid().squeeze(-1)

        attention_matrix = body_part_.new_zeros((
                            num_sample, num_body, num_body))
        for i in range(num_body):
            for j in range(i, num_body):
                attention_matrix[:, i, j] = (body_part_value[:, i] + body_part_value[:, j])/2
        for i in range(num_body):
            for j in range(0, i):    
                attention_matrix[:, i, j] = attention_matrix[:, j, i]
    
        fusion_matrix = torch.sqrt(affinity_matrix.squeeze(-1)*attention_matrix)
        degree_matrix = self.generate_degree_matrix(fusion_matrix)
        fusion_matrix = torch.matmul(degree_matrix, fusion_matrix)     #Adjacent matrix
        enhanced_feature = torch.matmul(fusion_matrix, x_original)
        enhanced_feature = self.relu(self.parameter_matrix(enhanced_feature))

        enhanced_feature = self.relu(self.part_out(
                                    enhanced_feature.reshape(num_sample, -1)))
        return enhanced_feature

    def inter_graph(self, x_full, rois, num_proposal_list):
        # Inter-proposal Graph
        neighbour_feats = []
        num_proposal = np.cumsum(np.array(num_proposal_list))
        
        batch_size = len(num_proposal_list)
        for img_ind in range(batch_size):
            if img_ind == 0:
                overlaps = bbox_overlaps(
                    rois[:num_proposal[img_ind]], 
                    rois[:num_proposal[img_ind]])
                num = num_proposal[img_ind]
                x_body = x_full[:num_proposal[img_ind]]
            else:
                overlaps = bbox_overlaps(
                    rois[num_proposal[img_ind-1]: num_proposal[img_ind]], 
                    rois[num_proposal[img_ind-1]: num_proposal[img_ind]])
                num = num_proposal[img_ind] - num_proposal[img_ind-1]
                x_body = x_full[num_proposal[img_ind-1]: num_proposal[img_ind]]
            mask_tensor = 1 - torch.eye(num.item())
            overlaps = overlaps * mask_tensor.to(overlaps)
            degree_matrix = self.generate_degree_matrix(overlaps).squeeze(0)
            overlaps = torch.matmul(degree_matrix, overlaps)
            x_body_ = self.relu(
                    self.parameter_matrix2(torch.matmul(overlaps, x_body)))
            neighbour_feats.append(x_body_)
        x_neighbour = torch.cat(neighbour_feats, dim=0)
        return x_neighbour

    def generate_degree_matrix(self, matrix):
        # Generate degree matrix for adjacent matric
        if matrix.dim() != 3:
            matrix = matrix.unsqueeze(0)    # batch_size x N x N
        N = matrix.size(-1)
        
        matrix_sum = torch.sum(matrix, dim=-1)      
        matrix_sum_ = matrix_sum.reshape(-1)
        non_zero_ind = torch.nonzero(matrix_sum_).squeeze()
        matrix_sum_[non_zero_ind] = 1 / matrix_sum_[non_zero_ind]
        matrix_sum_ = matrix_sum_.reshape(-1, N)
        
        degree_matrix = matrix_sum_[:, :, None].expand_as(matrix)
        degree_matrix = degree_matrix * torch.eye(N).type_as(matrix)
        return degree_matrix

    @property
    def norm(self):
        return getattr(self, self.norm_name)

@HEADS.register_module
class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

def jitter_gt(proposals, gts):
    for gt in gts:
        x1, y1, x2, y2 = gt
        width, height = x2-x1+1, y2-y1+1
        for i in range(10):
            x_jitter = random.uniform(-0.2, 0.2)
            y_jitter = random.uniform(-0.2, 0.2)
            proposal = proposals.new_ones((1, 5))
            proposal[:, 0] = x1+x_jitter*width
            proposal[:, 1] = y1+y_jitter*height
            proposal[:, 2] = x2+x_jitter*width
            proposal[:, 3] = y2+y_jitter*height
            proposals = torch.cat((proposals, proposal), 0)
    return [proposals]
