# B2SCVR
[MM'25] The official implementation of the paper "Towards Blind Bitstream-corrupted Video Recovery: A Visual Foundation Model-driven Framework"

> [Tianyi Liu]()<sup>1</sup>, [Kejun Wu]()<sup>2</sup>, [Chen Cai]()<sup>1</sup>, [Yi Wang]()<sup>3</sup>, [Kim-Hui Yap]()<sup>1</sup>, and [Lap-Pui Chau]()<sup>3</sup><br>
> <sup>1</sup>School of EEE, Nanyang Technological University<br>
> <sup>2</sup>School of Electronic Information and Communications, Huazhong University of Science and Technology<br>
> <sup>3</sup>Department of Electrical and Electronic Engineering, The Hong Kong Polytechnic University

### Installation

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

### Quick Test

0. Prepare inputs: a corrupted video bitstream and the first corruption indication (e.g., the first corruption mask in frame 9 of ```inputs/trucks-race_2.h264```).
   
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

4. The recovered frame sequence will be saved in ```outputs/``` folder.

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


