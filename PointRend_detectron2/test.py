import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")
from PIL import Image
import glob

# import PointRend project
from detectron2.projects import point_rend


# Hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='/mnt/nvideo1/M10802131_Zhenyu_Li_Graduation/Proposed_work/Datasets/new_GTA/val_img', help='The image root which need to generate instance map.')
parser.add_argument("--saveroot", type=str, default='/mnt/nvideo1/M10802131_Zhenyu_Li_Graduation/Proposed_work/Datasets/new_GTA/val_inst', help='The save root.')
parser.add_argument("--res", type=str, default='1080,1920', help='Width,Heigth')
args = parser.parse_args()

img_list = glob.glob(args.dataroot+'/*')
cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)
res = (int(args.res.split(',')[0]),int(args.res.split(',')[1]))

for i in img_list:
  print(i)
  im = cv2.imread(i)
  outputs = predictor(im)
  masks = outputs["instances"].to("cpu").pred_masks
  map = np.zeros(res,dtype=np.int16)
  for j in range(masks.shape[0]):
    map[np.where(masks[j]==True)]=j*2+120
  im = Image.fromarray(map)
  im.save(args.saveroot+'/'+i.split('/')[-1])