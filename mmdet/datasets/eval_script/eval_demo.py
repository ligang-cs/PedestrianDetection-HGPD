import os
from .coco import COCO
from .eval_MR_multisetup import COCOeval

def evaluate_MR(annFile, resFile, save_out):
	annType = 'bbox'      #specify type here
	print('Running demo for *%s* results.'%(annType))

	## running evaluation
	res_file = open(os.path.join(save_out, 'results.txt'), "w")
	MR_list = []
	for id_setup in range(0,4):
	    cocoGt = COCO(annFile)
	    cocoDt = cocoGt.loadRes(resFile)
	    imgIds = sorted(cocoGt.getImgIds())
	    cocoEval = COCOeval(cocoGt,cocoDt,annType)
	    cocoEval.params.imgIds  = imgIds
	    cocoEval.evaluate(id_setup)
	    cocoEval.accumulate()
	    MR_list.append(cocoEval.summarize(id_setup, res_file=res_file))
	res_file.close()
	return MR_list


