# AGENTS.md

本文件面向在本仓库中工作的 coding agents（如 Codex、Claude Code、Cursor Agent 等）。
目标是帮助代理在尽量少踩坑的前提下，完成代码修改、验证和交付。

## 1. 项目概览

- 项目名：LabUtopia
- 类型：基于 NVIDIA Isaac Sim 的高保真实验室仿真与具身智能 benchmark
- 主要语言：Python 3.11
- 关键依赖：Isaac Sim 5.1、PyTorch、Hydra / OmegaConf、h5py、loguru
- 运行环境：README 标注为 Ubuntu 24.04 + NVIDIA RTX GPU + CUDA

高层入口：

- `main.py`：仿真主入口，负责 `config -> factory -> task/controller -> simulation loop`
- `tests/test_single_config.py`：单个配置的烟雾测试
- `tests/test_config_files.py`：批量配置测试

训练不在本仓库树内进行：`policy/` 下是四个 VLA 子模块（Isaac-GR00T、lerobot、
lingbot-vla、openpi），训练在各子模块内部完成。历史遗留的树内训练入口
`train.py` / `train-muilt.py`（Diffusion UNet / ACT）已于 2026-07-10 删除。

## 2. 目录速览

- `config/`：任务 YAML 配置，按 Level 1-5 组织
- `tasks/`：任务环境，负责场景、物体、相机、观测
- `controllers/`：控制逻辑，负责动作生成、成功判定、数据采集/推理/回放调度
- `factories/`：task/controller/robot/collector 注册与构造
- `robots/`：Franka、Ridgebase、Piper 等机器人定义
- `data_collectors/`：HDF5 采集逻辑
- `policy/`：策略训练（git 子模块：Isaac-GR00T、lerobot、lingbot-vla、openpi）
- `packages/openpi-client/`：内置的远程推理 client 子包
- `assets/`：USD 场景和机器人资源，体积大且容易误伤
- `utils/`：`ObjectUtils`、相机处理、回放加载、A* 等工具

## 3. 核心架构约束

### 3.1 Task / Controller 分工

- `Task` 负责场景状态和观测数据。
- `Controller` 负责机器人动作和成功条件判断。
- 两者由 `main.py` 独立创建，再在运行时拼接。

不要把场景初始化逻辑塞进 controller，也不要把动作决策逻辑塞进 task，除非你在做明确的架构调整。

### 3.2 Task 基类约束

所有任务最终继承自 `tasks/base_task.py` 中的 `BaseTask`。

最低要求：

- 实现 `step(self) -> Optional[Dict[str, Any]]`
- 需要支持确定性回放时，实现或扩展 `reset_with_init_state`

可直接复用的基础能力：

- `setup_cameras()`
- `setup_objects()`
- `setup_materials()`
- `setup_lighting()`
- `get_basic_state_info(...)`
- `reset()` / `reset_with_init_state(...)`

状态字典里会自动包含 `init_state`，用于 replay。

### 3.3 Controller 基类约束

所有控制器最终继承自 `controllers/base_controller.py` 中的 `BaseController`。

子类必须实现：

- `_step_collect(self, state) -> tuple[Any, bool, bool]`
- `_step_infer(self, state) -> tuple[Any, bool, bool]`
- `_check_success(self) -> bool`

注意：

- `_step_replay()` 已在 `BaseController` 中实现，通常不要覆写
- `BaseController.step()` 会先把最新观测写入 `self.state`
- 成功判定通常依赖 `self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS`
- 当前默认 `REQUIRED_SUCCESS_STEPS = 60`
- `_last_failure_reason` 在本仓库里应始终是 `str`，无错误时用 `""`，不要写成 `None`

### 3.4 Factory 注册模式

工厂均采用显式注册，不是自动发现：

- `factories/task_factory.py`
- `factories/controller_factory.py`
- `factories/robot_factory.py`
- `factories/collector_factory.py`

新增 task/controller/robot/collector 时，除了写类本身，还必须注册到对应 factory。

配置里的键必须是已注册的 key，而且区分大小写。

重要：`task_type` 和 `controller_type` 不一定相同。
例如：

- `level1_open_door.yaml` 使用 `task_type: "open_close"` + `controller_type: "open"`
- `level1_close_door.yaml` 使用 `task_type: "open_close"` + `controller_type: "close"`

不要想当然把二者改成同名。

### 3.5 Singleton / 配置约束

- `ObjectUtils` 必须通过 `ObjectUtils.get_instance()` 获取，不要直接实例化
- 注册 OmegaConf resolver 之前，先用 `OmegaConf.has_resolver(...)` 判断
- 日志统一使用 `loguru.logger`

## 4. 修改建议

### 4.1 优先修改哪里

- 改任务逻辑：优先看 `tasks/`
- 改动作流程或成功判定：优先看 `controllers/`
- 新增任务类型：通常要同时改 `tasks/`、`controllers/`、`factories/`、`config/`
- 改训练逻辑：在 `policy/` 下对应的 VLA 子模块内改（注意子模块有独立的 git 仓库）
- 改远程推理：看 `controllers/inference_engines/` 和 `packages/openpi-client/`

### 4.2 尽量不要碰哪里

- `assets/`：大部分是 USD/材质/模型资源，除非用户明确要求，不要编辑
- `outputs/`、测试输出、数据集目录：不要把生成物混进提交
- `packages/openpi-client/`：这是内置子包，只在远程推理链路相关变更时再改

### 4.3 命名与风格

- task 文件：`{action}_task.py`
- controller 文件：`{action}_controller.py`
- task 类：`{Action}Task`
- controller 类：`{Action}TaskController` 或项目里已有的对应命名
- camera 观测键：`{camera_name}_{image_type}`
- episode 文件：`episode_{NNNN}.h5`

代码风格按仓库现有配置执行：

- 顶层 `pyproject.toml` 使用 `ruff` 和 `mypy`
- 公共方法尽量补齐类型标注
- import 分组为：stdlib -> third-party -> first-party
- 不要把 import 塞进函数体，除非有非常明确的延迟导入需求

## 5. 常用工作流

### 5.1 运行仿真

建议总是显式指定配置名：

```bash
python main.py --config-name level1_pick
python main.py --config-name level4_clean_beaker
python main.py --config-name level5_navigation --no-video
```

说明：

- `mode` 由 YAML 配置控制，常见值为 `collect` / `infer` / `replay`
- 输出目录通常由配置中的 `hydra.run.dir` 和 `multi_run.run_dir` 决定
- `--backend gpu` 会切换到 GPU 物理设置；默认是 `numpy`

### 5.2 训练策略

先用 `scripts/lerobot_export/` 把采集数据导出为 LeRobot 格式，再进入 `policy/`
下对应子模块（Isaac-GR00T / lerobot / lingbot-vla / openpi）按其 README 训练：

```bash
python -m scripts.lerobot_export.cli --src <run_dir> --dst <out_dir> --version v2.1
```

### 5.3 轻量验证

静态检查：

```bash
ruff check .
ruff format --check .
mypy .
```

说明：

- 顶层 `ruff` / `mypy` 默认排除了 `packages/`、`assets/`、`outputs/`
- 如果你改了 `packages/openpi-client/`，请额外对该子包做定向检查

仿真相关烟雾测试（需要 Isaac Sim 环境）：

```bash
python tests/test_single_config.py level1_pick
python tests/test_single_config.py level1_open_door
python tests/test_config_files.py
```

这两个测试脚本会把配置临时改为：

- `mode: collect`
- `max_episodes: 8`
- `collector.type: mock`

因此它们是比真实采集更轻的验证方式。

## 6. 新增能力时的最小变更清单

### 6.1 新增任务

至少检查以下几点：

1. 在 `tasks/` 中新增任务类
2. 在 `controllers/` 中新增或复用控制器
3. 在对应 factory 中注册
4. 新增 `config/*.yaml`
5. 确认 `task_type` / `controller_type` 与 factory 注册键一致
6. 至少做一次 `test_single_config.py` 或最小手动运行

### 6.2 新增配置

至少确认：

- `usd_path` 存在
- `robot.type` 已注册
- `collector.type` 已注册
- `task.max_steps` 合理
- 相机配置里的 `prim_path`、`resolution`、`image_type` 正确
- 如果依赖 replay / infer，相关路径或服务参数已经设置

## 7. 已知陷阱

- `main.py` 的 `--config-name` 默认值为 `level3_heat_liquid`；运行时建议显式传 `--config-name` 指定任务
- `main.py` 虽然解析了 `--headless`，但当前 `SimulationApp` 初始化仍写死为 `"headless": False`；不要假设这个参数已经生效
- registry key（`task_type` / `controller_type`）已统一为 `snake_case`（如 `open_transport_pour`、`liquid_mixing`）；改配置时要按源码中的注册值填写
- 工作区可能已经有用户未提交改动；除非用户明确要求，不要回退他人的变更

## 8. 提交前检查

在结束修改前，尽量完成以下事项：

1. 只改与任务直接相关的文件
2. 避免误改 `assets/` 和生成物目录
3. 跑至少一轮与改动匹配的最小验证
4. 若未能验证，明确说明原因（如本机无 Isaac Sim / 无 GPU / 无数据）
5. 在说明中写清楚是否改了 config、factory 注册项或运行命令
