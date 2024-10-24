# Prerequisites
**Please ensure you have prepared the environment and the DAIR-V2X-I or Rope3D dataset.**
# Train and Test
Train CoBEV with 8 GPUs
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
Eval CoBEV with 8 GPUs
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```