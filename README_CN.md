<div align="center">
  <img src="images/logo.png" alt="Logo" width="180">
</div>

# [NeurIPS 2025] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents

<div align="center">

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red.svg)](https://arxiv.org/pdf/2505.22634v1.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2505.22634-b31b1b.svg)](https://arxiv.org/abs/2505.22634)
[![Website](https://img.shields.io/badge/🌐_Website-LabUtopia-blue.svg)](https://rui-li023.github.io/labutopia-site/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset)

</div>


<div align="center">
  <img src="images/teaser.png" alt="LabUtopia Teaser" width="80%">
</div>


## 系统要求
- 支持CUDA的RTX系列NVIDIA GPU（Isaac Sim 不支持A100/A800）
- Ubuntu 2204（经过我们测试的系统版本）
- conda
- Python 3.10
- Isaac Sim 4.2

## 🛠️ 安装

### 1. 代码下载

下载代码并拉去场景资产

```bash
git clone https://github.com/Rui-li023/LabUtopia.git
sudo apt install git-lfs 
git lfs pull
```

### 2. 环境创建
创建并激活新的conda环境：
```bash
conda create -n labutopia python=3.10 -y
conda activate labutopia
```

### 3. 依赖安装
安装所需包：
```bash
# 安装PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装Isaac Sim
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

# 安装其他依赖
pip install -r requirements.txt

# 运行脚本设置.vscode/settings.json
python setup_vscode_settings.py
```

## 代码结构

```
LabSim/
├── assets/                # 资源文件目录
│   ├── chemistry_lab/     # 化学实验室场景资源
│   ├── fetch/             # Fetch机器人相关资源
│   ├── navigation/        # 导航任务相关资源
│   └── robots/            # 机器人模型资源
├── config/                # 配置文件目录
│   ├── level1_*.yaml    # Level 1基础任务配置
│   ├── level2_*.yaml    # Level 2组合任务配置
│   ├── level3_*.yaml    # Level 3泛化性任务配置
│   └── level4_*.yaml    # Level 4长序列任务配置
├── controllers/         # 控制器实现
│   ├── atomic_actions/  # 基础动作控制器
│   ├── inference_engines/ # 推理引擎实现
│   └── robot_controllers/ # 机器人控制器
├── data_collectors/     # 数据收集器实现
├── factories/           # 工厂类实现
├── policy/            # 策略模型实现
├── tasks/             # 任务定义实现
├── tests/            # 测试代码
└── utils/            # 工具函数
```

### 设计思路
1. 使用模块化结构
2. 在task中完成场景状态和观测数据的获取，包括相机图像、机器人状态和场景物体状态
3. 在controller中完成机器人控制和任务成功条件判断

## 使用方法

### 数据收集

收集训练数据是训练模型的第一步，Labutopia 支持多种任务类型的数据收集。

#### 1. 选择配置文件
在`config`文件夹中有多种预配置的任务文件：

**Level 1 基础任务：**
- `level1_pick.yaml` - 抓取任务
- `level1_place.yaml` - 放置任务  
- `level1_open_door.yaml` - 开门任务
- `level1_open_drawer.yaml` - 开抽屉任务
- `level1_close_door.yaml` - 关门任务
- `level1_close_drawer.yaml` - 关抽屉任务
- `level1_pour.yaml` - 倾倒任务
- `level1_press.yaml` - 按压任务
- `level1_shake.yaml` - 摇晃任务
- `level1_stir.yaml` - 搅拌任务

**Level 2 组合任务：**
- `level2_ShakeBeaker.yaml` - 摇晃烧杯
- `level2_StirGlassrod.yaml` - 玻璃棒搅拌
- `level2_PourLiquid.yaml` - 倾倒液体
- `level2_TransportBeaker.yaml` - 运输烧杯
- `level2_Heat_Liquid.yaml` - 加热液体
- `level2_openclose.yaml` - 开关任务


**Level 3 泛化性任务：**
- `level3_PourLiquid.yaml` - 复杂倾倒任务
- `level3_Heat_Liquid.yaml` - 复杂加热任务
- `level3_TrabsportBeaker.yaml` - 复杂运输任务
- `level3_open.yaml` - 复杂开启任务
- `level3_pick.yaml` - 复杂抓取任务
- `level3_press.yaml` - 复杂按压任务

**Level 4 长序列任务：**
- `level4_CleanBeaker.yaml` - 清洗烧杯
- `level4_DeviceOperation.yaml` - 设备操作

#### 2. 修改配置参数

每个配置文件包含以下主要参数需要根据需求调整：

```yaml
# 基本配置
name: level1_pick             # 任务名称
task_type: "pick"             # 任务类型，用于在工厂类中创建
controller_type: "pick"       # 控制器类型，用于在工厂类中创建
mode: "collect"               # 模式：infer or collect

# 场景配置
usd_path: "assets/chemistry_lab/pick_task/scene.usd" 

# 任务参数
task:
  max_steps: 1000                   # 最大步数
  obj_paths:                        # 目标物体配置
    - path: "/World/conical_bottle02"
      position_range:               # 物体位置范围
        x: [0.22, 0.32]
        y: [-0.07, 0.03]
        z: [0.80, 0.80]

# 数据收集参数
max_episodes: 100                   # 最大收集轮数

# 相机配置
cameras_names: ["camera_1", "camera_2"]
cameras:
  - prim_path: "/World/Camera1"
    name: "camera_1"
    translation: [2, 0, 2]         # 相机位置
    resolution: [256, 256]         # 分辨率
    focal_length: 6                 # 焦距
    orientation: [0.61237, 0.35355, 0.35355, 0.61237]  # 方向
    image_type: "rgb"              # 图像类型

# 机器人配置
robot:
  type: "franka"                  # 机器人类型，目前只支持franka
  position: [-0.4, -0, 0.71]      # 机器人位置

# 数据收集器配置
collector:
  type: "default"                  # 收集器类型
  compression: null                # 压缩设置
```

#### 3. 运行数据收集

选择配置文件后运行：
```bash
# 使用默认配置
python main.py --config-name level1_pick
```

数据将保存在 `outputs/collect/日期/时间_任务名/` 目录下。

### 训练

训练过程使用收集到的数据来训练机器人策略模型。

#### 1. 选择训练配置

在`policy/config/`文件夹中有多种训练配置：

- `train_diffusion_unet_image_workspace.yaml` - 扩散模型训练
- `train_act_image_workspace.yaml` - ACT模型训练

#### 2. 修改训练参数

主要需要调整的参数：

```yaml
# 模型配置
policy:
  _target_: policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
  shape_meta: ${shape_meta}        # 数据形状元信息
  
  # 噪声调度器配置
  noise_scheduler:
    num_train_timesteps: 100       # 训练时间步数
    beta_start: 0.0001             # β起始值
    beta_end: 0.02                 # β结束值
    beta_schedule: squaredcos_cap_v2  # β调度策略

  # 观察编码器配置
  obs_encoder:
    _target_: policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    rgb_model:
      _target_: policy.model.vision.model_getter.get_resnet
      name: resnet18               # 骨干网络
    resize_shape: [256, 256]      # 调整大小
    random_crop: False              # 随机裁剪

# 训练参数
training:
  device: "cuda:0"                # 训练设备
  seed: 42                        # 随机种子
  num_epochs: 8000                # 训练轮数
  lr: 1.0e-4                      # 学习率
  batch_size: 64                  # 批次大小
  gradient_accumulate_every: 1     # 梯度累积步数
  
  # 检查点保存
  checkpoint_every: 30             # 每30轮保存一次
  val_every: 10                   # 每10轮验证一次

# 数据加载器配置
dataloader:
  batch_size: 64                  # 批次大小
  num_workers: 4                  # 工作进程数
  shuffle: True                   # 是否打乱数据

# 优化器配置
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4                      # 学习率
  betas: [0.95, 0.999]           # Adam参数
  weight_decay: 1.0e-6           # 权重衰减
```

#### 3. 指定数据集的位置
修改`policy/config/task`文件夹下对应的配置文件，修改参数 `dataset_path` 为你的数据集所在的文件夹

#### 4. 运行训练

```bash
# 使用扩散模型训练
python train.py --config-name=train_diffusion_unet_image_workspace

# 使用ACT模型训练
python train.py --config-name=train_act_image_workspace
```

训练日志和模型将保存在 `outputs/train/日期/时间_模型名_任务名/` 目录下。

### 推理

使用训练好的模型进行推理测试。

#### 1. 修改配置文件

将配置文件中的模式从 `collect` 改为 `infer`，并添加推理相关配置：

```yaml
# 基本配置
mode: "infer"                     # 改为推理模式：remote or local

# 推理配置
infer:
  obs_names: {"camera_1_rgb": 'camera_1_rgb', "camera_2_rgb": 'camera_2_rgb'}
  
  # 本地推理配置
  policy_model_path: "outputs/train/2025.03.25/12.43.59_train_act_image_pick_pick_data/checkpoints/latest.ckpt"
  policy_config_path: "outputs/train/2025.03.25/12.43.59_train_act_image_pick_pick_data/.hydra/config.yaml"
  normalizer_path: "outputs/train/2025.03.25/12.43.59_train_act_image_pick_pick_data/checkpoints/normalize.ckpt"
  
  # 远程推理配置（可选）
  type: "remote"                  # 使用远程推理
  host: "101.126.156.90"         # 服务器地址
  port: 56434                     # 服务器端口
  n_obs_steps: 1                  # 观察步数
  timeout: 30                     # 超时时间
  max_retries: 3                  # 最大重试次数

max_episodes: 50                  # 推理数据集数
```

#### 2. 运行推理

```bash
# 使用本地模型推理
python main.py --config-name level1_pick

# 使用远程推理
python main.py --config-name level3_PourLiquid
```

推理结果将保存在 `outputs/infer/日期/时间_任务名/` 目录下。


## 使用OpenPI

### 安装

下载我们修改过后的OpenPI代码，并参考其`Readme`安装环境并下载预训练权重

```
git clone https://github.com/Rui-li023/openpi.git
```

### 数据转换

需要将labutopia格式的数据转换为LeRobot格式的数据，下面命令会自动在`$HF_HOME/lerobot/{repo_name}`下生成需要的lerobot格式数据集

```
python scripts/convert_labsim_data_to_lerobot.py --data_dir outputs/collect/xxx/xxx/dataset --num_processes 8 --fps 60 --repo_name labutopia/level3-pick
```

### 远程推理

Labutopia 支持使用openpi格式的远程服务器进行模型推理

#### 安装

```
cd packages/openpi-client
pip install -e . 
```

#### 配置
在配置文件中配置远程推理引擎：

```yaml
infer:
  engine: remote  # 使用远程推理引擎
  host: "0.0.0.0" # OpenPI服务器主机
  port: 8080      # OpenPI服务器端口（可选）
  n_obs_steps: 1  # Obs步数
```

#### 使用方法
OpenPI客户端提供简化的WebSocket与远程服务器通信：

1. **初始化**：客户端自动使用WebSocket连接到OpenPI服务器
2. **推理**：向服务器发送观察数据（图像、姿态）并接收动作预测
3. **数据格式**：自动处理图像格式转换和姿态数据序列化
4. **错误处理**：包含预测失败的回退机制

#### 服务器响应格式
OpenPI服务器应返回以下格式之一的动作：
- `{"action": [action_array]}`
- `{"actions": [action_array]}`
- 任何包含"action"键的字典

## 📚 引用

```bibtex
@article{li2025labutopia,
  author    = {Li, Rui and Hu, Zixuan and Qu, Wenxi and Zhang, Jinouwen and Yin, Zhenfei and Zhang, Sha and Huang, Xuantuo and Wang, Hanqing and Wang, Tai and Pang, Jiangmiao and Ouyang, Wanli and Bai, Lei and Zuo, Wangmeng and Duan, Ling-Yu and Zhou, Dongzhan and Tang, Shixiang},
  title     = {LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents},
  journal   = {arXiv preprint arXiv:2505.22634},
  year      = {2025},
}
```

## 📄 许可

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.