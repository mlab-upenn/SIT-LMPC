# SIT-LMPC 

SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks

[[website]](https://sites.google.com/view/sit-lmpc/)[[Paper]](https://ieeexplore.ieee.org/document/11260933)

## Installation

0. Update Nvidia driver (if necessary).
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-570
```

1. Setup environment (x86 Ubuntu):
```
# Install poetry (if necessary):
curl -sSL https://install.python-poetry.org | python3 -

# Install python3.12 (if necessary)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12-dev python3-virtualenv

git clone -b sit_lmpc git@github.com:zzangupenn/SIT_LMPC.git
cd SIT_LMPC
virtualenv -p python3.12 sit_env
source sit_env/bin/activate
pip install -e .
```

2. Install utilsuite & f1tenth_gym (Directly install in SIT_LMPC folder)
```
git clone -b sit_lmpc https://github.com/zzangupenn/utilitySuite.git
pip install -e ./utilitySuite
git clone -b sit_lmpc https://github.com/zzangupenn/f1tenth_gym.git
pip install -e ./f1tenth_gym
```

## Race Car Example
```
cd sit_lmpc
python run_racecar.py
```
`run_racecar.py` runs the simulated race car experiment in the paper. We use JAX as our GPU library. The config uses the parameters in the paper and it's configurable in `config/exp_config.yaml`. The script initializes the gym environment, first records the initial safe set with centerline following with MPPI, then starts SIT-LMPC for task opimization.

Feel free to email us if you have questions. Please cite us if our work can be a help for you.
```
@article{zang2025sit,
  title={SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks},
  author={Zang, Zirui and Amine, Ahmad and Kokolakis, Nick-Marios T and Nghiem, Truong X and Rosolia, Ugo and Mangharam, Rahul},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={1},
  pages={986--993},
  year={2025},
  publisher={IEEE}
}
```

