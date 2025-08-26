<div align="center">

   <h1>[ACM MM'25] Towards Blind Bitstream-corrupted Video Recovery: A Visual Foundation Model-driven Framework </h1>

> [Tianyi Liu](https://scholar.google.com/citations?user=Sdw8w_YAAAAJ&hl=zh-CN)<sup>1</sup>, [Kejun Wu](https://kejun-wu.github.io/)<sup>2</sup>, [Chen Cai](https://scholar.google.com/citations?user=awQEstcAAAAJ&hl=zh-CN)<sup>1</sup>, [Yi Wang](https://scholar.google.com/citations?user=MAG909MAAAAJ&hl=zh-CN)<sup>3</sup>, [Kim-Hui Yap](https://scholar.google.com/citations?user=nr86m98AAAAJ&hl=zh-CN)<sup>1</sup>, and [Lap-Pui Chau](https://scholar.google.com/citations?user=MYREIH0AAAAJ&hl=zh-CN)<sup>3</sup><br>
> <sup>1</sup>School of Electrical and Electronic Engineering, Nanyang Technological University<br>
> <sup>2</sup>School of Electronic Information and Communications, Huazhong University of Science and Technology<br>
> <sup>3</sup>Department of Electrical and Electronic Engineering, The Hong Kong Polytechnic University

<p align="center">
    <a href='https://arxiv.org/abs/2507.22481'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
</p>

</div>

## Installation

```bash
git clone https://github.com/LIUTIGHE/B2SCVR.git
conda create -n b2scvr python=3.10
conda activate b2scvr

# build mmcv first according to the official documents (can ignore the torch mismatch)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# install torch according to the official documents
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia  

# install DAC developed based on SAM2.1
cd ../model/modules/sam2
pip install -e .

# other requirements
cd ../../..
pip install -r requirements.txt
```

If ```ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor``` occurs, one possible solution is to manually modify the 8th row in ```degradations.py``` from the Error from ``` from torchvision.transforms.functional_tensor import rgb_to_grayscale ``` to ``` from torchvision.transforms.functional import rgb_to_grayscale ```

## Quick Test

0. Prepare inputs and model checkpoints: a corrupted video bitstream and the first corruption indication (e.g., the first corruption mask in frame 9 of ```inputs/trucks-race_2.h264```). Then download the model checkpoints via [this link](https://entuedu-my.sharepoint.com/:f:/g/personal/liut0038_e_ntu_edu_sg/EvxHRdWSFpZIhyiqHU-NYmEBGy5N1iJ4I69iigYtL7FBkw?e=GpPNnL), and put into ```checkpoints/``` folder.
   
1. Extract the corrupted frames and motion vector (mv) and prediction mode (pm) for each frame from the input corrupted video bitstream (e.g., ```inputs/trucks-race_2.h264```)
   ```python inputs.py --input inputs/trucks-race_2.h264```

3. Stage 1: Use DAC to detect and localize video corruption:
   ```bash
   cd model/modules/sam2
   bash run.sh  # if there is a loading error, mostly related to vos_inference.py line 277-278, which sets a fixed suffix
   ``` 

3. Stage 2: Use the CFC-based recovery model to perform restoration
   ```bash
   python test.py --ckpt checkpoints/B2SCVR.pth --input_video inputs/bsc_imgs/trucks-race --dac_mask inputs/results/trucks-race --width 432 --height 240  # set 240P test if OOM occurs
   ```

4. The recovered frames sequence and GIF video will be saved in ```outputs/``` folder.

## Citation

If you find the code useful, please kindly consider citing our paper

```
@article{liu2025towards,
  title={Towards Blind Bitstream-corrupted Video Recovery via a Visual Foundation Model-driven Framework},
  author={Liu, Tianyi and Wu, Kejun and Cai, Chen and Wang, Yi and Yap, Kim-Hui and Chau, Lap-Pui},
  journal={arXiv preprint arXiv:2507.22481},
  year={2025}
}
```

## Acknowledgements

This work is built upon [BSCV](https://github.com/LIUTIGHE/BSCV-Dataset), [SAM-2](https://github.com/facebookresearch/sam2), and [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary).


