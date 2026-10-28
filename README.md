# Standalone HL RNN bandit task

The standalone high-level RNN bandit task isolates learning and block-switch
adaptation in a two-arm bandit, without vision, navigation, DG/CA3 or low-level
PPO. It reuses the full setup's high-level RNN, trial-history replay and Q/policy
losses, with compatible manager checkpoint loading.

Only PyTorch and NumPy are needed for this standalone task. From the repository root:

```bash
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/seed7 --seed 7
```

See the [standalone task runbook](sf_working_directories/zeynep/bandit/README.md)
for block-switch rules, controls, evaluation metrics and checkpoint transfer.

# quick start


- mamba create -n "yourname" python=3.10.12
	- older versions also work until 3.8
- pip install deepmind_lab-1.0-py3-none-any.whl 
  - the appropriate wheel depends on your environment, try [`magic dmlab wheel`](https://drive.google.com/file/d/1YEXjm06f79KY5LB4NZ0e4xce3j1dcuxf/view?usp=drive_link), or the wheel from sample factory
- pip uninstall numpy
- pip install dm_env
- (fork samplefactory)
- clone samplefactory
- cd samplefactory
- pip install -e .\[dev,mujoco,atari,vizdoom\] -> arguments depend on which environment, we do not need any

when encountering numpy.ndarray error, `pip uninstall numpy` and `pip install -e .` again

patch deepmindlab (transparent reward, custom maps)
```
cd deepmindlab_patch
chmod +x patch_deepmindlab.sh
./patch_deepmindlab.sh
```

# update for fixes
- pkg_resources error: setuptool needs to be an older version, e.g. 65.5.0
- torchvision: pip install torchvision
- libosmesa
  - on NEMO2 Cluster: `module load lib/sdl2/2.28.2-gcccore-12.3.0 vis/mesa/24.1.3-gcccore-13.3.0`
  - with sudo: `sudo apt-get install libosmesa6  libosmesa6-dev`

# Citation

This project is accepted at ICLR 2026

https://openreview.net/forum?id=li1vfqDzRD


code related to the paper is in the folder [sf_xxl](sf_xxl)

