from sam2.build_sam import build_sam2_video_predictor
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

config_path = 'configs/sam2.1/sam2.1_hiera_t.yaml'
ckpt_path = '/media/tianyi/BSC-VideoCompletion/baselines/BSCV/model/modules/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_BSCV_lac_atd_dino_50eps.yaml/checkpoints/checkpoint.pt'

model_vid = build_sam2_video_predictor(config_path, ckpt_path, device='cuda')

def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6

print(f"Video model params: {count_params(model_vid):.1f} M")
