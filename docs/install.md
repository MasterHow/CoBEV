# Step-by-step installation instructions

**a.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**b.** Install mmcv-full==1.4.0  mmdet==2.19.0 mmsegmentation==0.20.0 mmdet3d==0.18.1.

**c.** Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```

**d.** Install requirements.
```shell
pip install -r requirements.txt
```

**e.** Install CoBEV.
```shell
python setup.py develop
```

**f.** Install some other packages.
```shell
pip uninstall opencv-python -y
pip install "opencv-python-headless<4.3"
pip install axial_attention
pip install Pillow==8.4.0
apt-get install libgl1-mesa-glx -y
pip install numba==0.53.0
```
