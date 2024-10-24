export TORCH_HOME=./$TORCH_HOME
python exps/dair-v2x/cobev_r50_128x128.py --amp_backend native -b 8 --gpus 8
python exps/dair-v2x/cobev_r50_128x128.py --ckpt outputs/cobev_r50_128x128/checkpoints/ -e -b 8 --gpus 8
