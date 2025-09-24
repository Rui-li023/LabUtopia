<div align="center">
  <img src="images/logo.png" alt="Logo" width="180">
</div>

# [NeurIPS 2025] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents

<div align="center">

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red.svg)](https://arxiv.org/pdf/2505.22634v1.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2505.22634-b31b1b.svg)](https://arxiv.org/abs/2505.22634)
[![Website](https://img.shields.io/badge/🌐_Website-LabUtopia-blue.svg)](https://rui-li023.github.io/labutopia-site/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/Ruinwalker/Labutopia-Dataset)

</div>

<div align="center">
  <img src="images/teaser.png" alt="LabUtopia Teaser" width="80%">
</div>

[中文版 README](README_CN.md) | [English README](README.md)

## System Requirements
- NVIDIA GPU with CUDA support (RTX series recommended, Isaac Sim does not support A100/A800)
- Ubuntu 22.04 (tested version)
- conda
- Python 3.10
- Isaac Sim 4.2

## 🛠️ Installation

### 1. Code Download

Download the code and pull scene assets:

```bash
git clone https://github.com/Rui-li023/LabUtopia.git
sudo apt install git-lfs 
git lfs pull
```

### 2. Environment Creation
Create and activate a new conda environment:
```bash
conda create -n labutopia python=3.10 -y
conda activate labutopia
```

### 3. Dependencies Installation
Install required packages:
```bash
# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install Isaac Sim
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

# Install other dependencies
pip install -r requirements.txt

# Run script to setup .vscode/settings.json
python setup_vscode_settings.py
```

## Code Structure

```
LabSim/
├── assets/                # Resource files directory
│   ├── chemistry_lab/     # Chemistry lab scene resources
│   ├── fetch/             # Fetch robot resources
│   ├── navigation/        # Navigation task resources
│   └── robots/            # Robot model resources
├── config/                # Configuration files directory
│   ├── level1_*.yaml    # Level 1 basic task configs
│   ├── level2_*.yaml    # Level 2 combined task configs
│   ├── level3_*.yaml    # Level 3 generalization task configs
│   └── level4_*.yaml    # Level 4 long sequence task configs
├── controllers/         # Controller implementations
│   ├── atomic_actions/  # Basic action controllers
│   ├── inference_engines/ # Inference engine implementations
│   └── robot_controllers/ # Robot controllers
├── data_collectors/     # Data collector implementations
├── factories/           # Factory class implementations
├── policy/            # Policy model implementations
├── tasks/             # Task definition implementations
├── tests/            # Test code
└── utils/            # Utility functions
```

### Design Philosophy
1. Modular structure for better code organization and maintainability
2. Scene state and observation data acquisition (including camera images, robot states, and scene object states) are handled in tasks
3. Robot control and task success condition checking are handled in controllers

## Usage

### Data Collection

Data collection is the first step in training models. LabUtopia supports data collection for multiple task types, and you can also download pre-collected data from our HuggingFace repository.

#### 1. Select Configuration File
There are multiple pre-configured task files in the `config` folder:

**Level 1 Basic Tasks:**
- `level1_pick.yaml` - Pick tasks
- `level1_place.yaml` - Place tasks  
- `level1_open_door.yaml` - Open door tasks
- `level1_open_drawer.yaml` - Open drawer tasks
- `level1_close_door.yaml` - Close door tasks
- `level1_close_drawer.yaml` - Close drawer tasks
- `level1_pour.yaml` - Pour tasks
- `level1_press.yaml` - Press tasks
- `level1_shake.yaml` - Shake tasks
- `level1_stir.yaml` - Stir tasks

**Level 2 Combined Tasks:**
- `level2_ShakeBeaker.yaml` - Shake beaker
- `level2_StirGlassrod.yaml` - Stir with glass rod
- `level2_PourLiquid.yaml` - Pour liquid
- `level2_TransportBeaker.yaml` - Transport beaker
- `level2_Heat_Liquid.yaml` - Heat liquid
- `level2_openclose.yaml` - Open and close tasks

**Level 3 Generalization Tasks:**
- `level3_PourLiquid.yaml` - Complex pour tasks
- `level3_Heat_Liquid.yaml` - Complex heating tasks
- `level3_TrabsportBeaker.yaml` - Complex transport tasks
- `level3_open.yaml` - Complex open tasks
- `level3_pick.yaml` - Complex pick tasks
- `level3_press.yaml` - Complex press tasks

**Level 4 Long Sequence Tasks:**
- `level4_CleanBeaker.yaml` - Clean beaker
- `level4_DeviceOperation.yaml` - Device operation

#### 2. Modify Configuration Parameters

Each configuration file contains the following main parameters that need to be adjusted according to requirements:

```yaml
# Basic configuration
name: level1_pick                    # Task name
task_type: "pick"                   # Task type
controller_type: "pick"             # Controller type
mode: "collect"                     # Mode: collect or infer

# Scene configuration
usd_path: "assets/chemistry_lab/pick_task/scene.usd"  # Scene file path

# Task parameters
task:
  max_steps: 1000                   # Maximum steps
  obj_paths:                        # Target object configuration
    - path: "/World/conical_bottle02"
      position_range:               # Object position range
        x: [0.22, 0.32]
        y: [-0.07, 0.03]
        z: [0.80, 0.80]

# Data collection parameters
max_episodes: 100                   # Maximum collection episodes

# Camera configuration
cameras_names: ["camera_1", "camera_2"]
cameras:
  - prim_path: "/World/Camera1"
    name: "camera_1"
    translation: [2, 0, 2]         # Camera position
    resolution: [256, 256]         # Resolution
    focal_length: 6                 # Focal length
    orientation: [0.61237, 0.35355, 0.35355, 0.61237]  # Orientation
    image_type: "rgb"              # Image type, rgb depth, point. You can use rgb+depth to get more than one type of image.

# Robot configuration
robot:
  type: "franka"                   # Robot type
  position: [-0.4, -0, 0.71]      # Robot position

# Data collector configuration
collector:
  type: "default"                  # Collector type
  compression: null                # Compression settings
```

#### 3. Run Data Collection

After selecting the configuration file, run:
```bash
# Use default configuration
python main.py --config-name level1_pick
```

Data will be saved in the `outputs/collect/date/time_taskname/` directory.

### Training

The training process uses collected data to train robot policy models.

#### 1. Select Training Configuration

There are multiple training configurations in the `policy/config/` folder:

- `train_diffusion_unet_image_workspace.yaml` - Diffusion model training (recommended)
- `train_act_image_workspace.yaml` - ACT model training

#### 2. Modify Training Parameters

Main parameters that need to be adjusted:

```yaml
# Model configuration
policy:
  _target_: policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
  shape_meta: ${shape_meta}        # Data shape metadata
  
  # Noise scheduler configuration
  noise_scheduler:
    num_train_timesteps: 100       # Training timesteps
    beta_start: 0.0001             # Beta start value
    beta_end: 0.02                 # Beta end value
    beta_schedule: squaredcos_cap_v2  # Beta schedule strategy

  # Observation encoder configuration
  obs_encoder:
    _target_: policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    rgb_model:
      _target_: policy.model.vision.model_getter.get_resnet
      name: resnet18               # Backbone network
    resize_shape: [256, 256]      # Resize shape
    random_crop: False              # Random crop

# Training parameters
training:
  device: "cuda:0"                # Training device
  seed: 42                        # Random seed
  num_epochs: 8000                # Training epochs
  lr: 1.0e-4                      # Learning rate
  batch_size: 64                  # Batch size
  gradient_accumulate_every: 1     # Gradient accumulation steps
  
  # Checkpoint saving
  checkpoint_every: 30             # Save every 30 epochs
  val_every: 10                   # Validate every 10 epochs

# Data loader configuration
dataloader:
  batch_size: 64                  # Batch size
  num_workers: 4                  # Number of workers
  shuffle: True                   # Whether to shuffle data

# Optimizer configuration
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4                      # Learning rate
  betas: [0.95, 0.999]           # Adam parameters
  weight_decay: 1.0e-6           # Weight decay
```

#### 3. Specify Dataset Location
Modify the corresponding configuration file in the `policy/config/task` folder, change the `dataset_path` parameter to your dataset folder location.

#### 4. Run Training

```bash
# Use diffusion model training
python train.py --config-name=train_diffusion_unet_image_workspace

# Use ACT model training
python train.py --config-name=train_act_image_workspace
```

Training logs and models will be saved in the `outputs/train/date/time_modelname_taskname/` directory.

### Inference

Use trained models for inference testing.

#### 1. Modify Configuration File

Change the mode from `collect` to `infer` in the configuration file and add inference-related configurations:

```yaml
# Basic configuration
mode: "infer"                     # Change to inference mode

# Inference configuration
infer:
  obs_names: {"camera_1_rgb": 'camera_1_rgb', "camera_2_rgb": 'camera_2_rgb'}
  
  # Local inference configuration
  policy_model_path: "outputs/train/2025.03.25/12.43.59_train_act_image_pick_pick_data/checkpoints/latest.ckpt"
  policy_config_path: "outputs/train/2025.03.25/12.43.59_train_act_image_pick_pick_data/.hydra/config.yaml"
  normalizer_path: "outputs/train/2025.03.25/12.43.59_train_act_image_pick_pick_data/checkpoints/normalize.ckpt"
  
  # Remote inference configuration (optional)
  type: "remote"                  # Use remote inference
  host: "101.126.156.90"         # Server address
  port: 56434                     # Server port
  n_obs_steps: 1                  # Observation steps
  timeout: 30                     # Timeout
  max_retries: 3                  # Maximum retries

max_episodes: 50                  # Inference episodes
```

#### 2. Run Inference

```bash
# Use local model inference
python main.py --config-name level1_pick

# Use remote inference
python main.py --config-name level3_PourLiquid
```

Inference results will be saved in the `outputs/infer/date/time_taskname/` directory.

## Use OpenPI

### Installation

Download our modified OpenPI code:

```
git clone https://github.com/Rui-li023/openpi.git
```

### Data Conversion

Convert LabUtopia format data to LeRobot format dataset:

```
python scripts/convert_labsim_data_to_lerobot.py --data_dir outputs/collect/xxx/xxx/dataset --num_processes 8 --fps 60 --repo_name labutopia/level3-pick
```

### Remote Inference

LabUtopia supports using remote servers for model inference.

#### Installation

```
cd openpi/packages/openpi-client
pip install -e . 
```

#### Configuration
Configure the remote inference engine in your config file:

```yaml
infer:
  engine: remote  # Use remote inference engine
  host: "0.0.0.0"  # OpenPI server host
  port: 8080  # OpenPI server port (optional)
  n_obs_steps: 3  # Observation steps
```

#### Usage
The OpenPI client provides simplified WebSocket communication with the remote server:

1. **Initialize**: The client automatically connects to the OpenPI server using WebSocket
2. **Inference**: Sends observation data (images, poses) to the server and receives action predictions
3. **Data Format**: Automatically handles image format conversion and pose data serialization
4. **Error Handling**: Includes fallback mechanisms for failed predictions

#### Server Response Format
The OpenPI server should return actions in one of these formats:
- `{"action": [action_array]}`
- `{"actions": [action_array]}`
- Any dictionary with a key containing "action"

## 📚 Citation

```bibtex
@article{li2025labutopia,
  author    = {Li, Rui and Hu, Zixuan and Qu, Wenxi and Zhang, Jinouwen and Yin, Zhenfei and Zhang, Sha and Huang, Xuantuo and Wang, Hanqing and Wang, Tai and Pang, Jiangmiao and Ouyang, Wanli and Bai, Lei and Zuo, Wangmeng and Duan, Ling-Yu and Zhou, Dongzhan and Tang, Shixiang},
  title     = {LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents},
  journal   = {arXiv preprint arXiv:2505.22634},
  year      = {2025},
}
```

## 📄 License
This repository contains both source code and data assets:

- **Code**  
  Released under the [MIT License](./LICENSE).  

- **Data Assets**  
  Released under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).  
  Free to use and modify for research and educational purposes **only**.  
