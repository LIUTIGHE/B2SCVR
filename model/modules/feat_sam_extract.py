import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

class SAM2Extractor (nn.Module):
    def __init__(self, checkpoint_path="/media/tianyi/BSC-VideoCompletion/B2SCVR/model/modules/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_BSCV_lac_atd_dino_50eps.yaml/checkpoints/checkpoint.pt") -> None:
    # def __init__(self, checkpoint_path=None):
        super(SAM2Extractor, self).__init__()    
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        # trainable cross-domain fusion layer
        
        
    def forward(self, x):
        sam_feat = self.encoder(x)
        return sam_feat
        