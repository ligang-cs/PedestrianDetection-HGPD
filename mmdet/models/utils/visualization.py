import cv2
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

visualization_path = '/ligang/Works/DFL_Nighttime/Experiments/visualization/IoU_heatmap/'
def IoU_heatmap(gt_bboxes, IoU_map, img_name, ind):
	IoU_map = IoU_map.cpu().numpy()
	heatmap = np.uint8(255*IoU_map)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	for box in gt_bboxes:
		stride = 2**(ind+2)
		cv2.rectangle(heatmap, (int(box[0]/stride),int(box[1]/stride)),(int(box[2]/stride),int(box[3]/stride)), (255,255,255), 2)
	save_path = visualization_path + img_name[:-4] + '_{}'.format(ind) + '.jpg'
	cv2.imwrite(save_path, heatmap)

visual_path = '/ligang/Works/DFL_Nighttime/Experiments/visualization/cosine_similarity/'

def draw_heatmap(rois,  roi_feature, tag, img, img_name):
	mean=[123.675, 116.28, 103.53]
	std=[58.395, 57.12, 57.375]
	img = img.squeeze(0)
	img = img.cpu().numpy().transpose(1, 2, 0)
	img = mmcv.imdenormalize(img, np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32), True)
	roi_img = []
	for roi_id in range(len(rois)):
		x1 = int(rois[roi_id, 0])
		x2 = int(rois[roi_id, 2])
		y1 = int(rois[roi_id, 1])
		y2 = int(rois[roi_id, 3])
		roi_img.append(img[y1: y2+1, x1: x2+1, :])
	roi_fea = []
	for j in range(roi_feature.size(0)):
		roi_fea.append(roi_feature[j])
	assert len(roi_img) == len(roi_fea)

	for i in range(len(roi_img)):
		roi = roi_img[i]
		if roi.shape[0] > 50:
			heatmap = roi_fea[i].detach().cpu().numpy()
			heatmap = np.max(heatmap, axis=0)
			# heatmap = F.relu(torch.sum(roi_fea[i], dim=0))
			# heatmap = heatmap.detach().cpu().numpy()
			heatmap = np.maximum(heatmap, 0)
			heatmap /= np.max(heatmap)
			heatmap = cv2.resize(heatmap, (roi.shape[1], roi.shape[0]))
			heatmap = np.uint8(255*heatmap)
			heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
			superposed_img = heatmap*0.4 + roi
			# brightness = v_channel*0.4 + roi
			superposed_img = cv2.resize(superposed_img, (2 * heatmap.shape[1], 2 * heatmap.shape[0]))
			# v_channel =  cv2.resize(v_channel, (2 * heatmap.shape[1], 2 * heatmap.shape[0]))
			roi =  cv2.resize(roi, (2 * heatmap.shape[1], 2 * heatmap.shape[0]))
			save_path =visual_path + img_name+ '_{}'.format(tag[i]) +'_{}.jpg'.format(i)
			save_img = visual_path + img_name+ '_{}'.format(tag[i]) + '_{}_original.jpg'.format(i)
			cv2.imwrite(save_path, superposed_img)
			cv2.imwrite(save_img, roi)

def save_gradient(grad):
	global roi_grad
	roi_grad = grad

def grad_CAM(roi, roi_feature, roi_score, pos_inds, loss_ind_pos, img, img_name):
	roi_feature.register_hook(save_gradient)

	# one_hot = np.zeros((1, roi_score.size()[-1]), dtype=np.float32)
	# score_np = roi_score[0].cpu().data.numpy()
	# index = np.argmax(score_np)
	# one_hot[0][index] = 1
	# one_hot = torch.from_numpy(one_hot).requires_grad_(True)
	# one_hot = torch.sum(one_hot.cuda()*roi_score[0])
	score_grad = torch.sum(roi_score[pos_inds], dim=1)
	torch.mean(score_grad).backward(retain_graph=True)
	global roi_grad
	loss_ind = torch.cat([loss_ind_pos[-3:], loss_ind_pos[:3]])
	tag = ['easy','easy','easy','hard','hard','hard']
	draw_grad_CAM(roi[pos_inds][loss_ind, 1:], roi_grad[loss_ind], roi_feature[pos_inds][loss_ind], tag, img, img_name)