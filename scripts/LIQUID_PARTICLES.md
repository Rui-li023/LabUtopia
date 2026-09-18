# 通用液体粒子工具

适用于在不同世界位置、不同坐标系或已校准容器中创建 PhysX PBD 液体。
没有 task/controller/factory 依赖，不修改原始模型、材质或容器碰撞；倒液成功判定仍归各任务 controller。

## 命令行

在 Isaac Sim 5.1 环境中运行，输出目录必须是新目录：

```bash
python -m scripts.generate_liquid \
  --config experiments/config/liquid/particle_positions.yaml \
  --output outputs/liquid_positions_demo --headless
```

示例在三个位置分别创建盒状、球状、柱状液体，输出初始场景、液体初态、初始/结束预览图和恢复验证结果。
这是液体工具演示，不是 benchmark 采集或成功率测试。

配置中的每个 `volumes` 条目使用独立的 `root_path`，可以指定：

- `shape: box/sphere/cylinder`，以及 `center`、`size` 或 `radius/height`。
- `frame_matrix`：USD 行向量 4×4 矩阵，平移在最后一行；默认单位阵，即世界坐标。
- `frame_path`：已有场景内的坐标系 prim。可用 `scene_usd` 引用现有场景，不改源文件。
- `velocity`：初始世界速度，单位 m/s。生成后只由物理仿真推动，不持续跟随坐标系。
- `parameters`：间距、密度、颜色及 PBD 材料参数。全局默认可被每个条目覆盖。
- `shape: container`：同时提供 `interior` 的 `center_x/center_z/radius/bottom/top/lower_profile`、
  `fill_height` 和容器的 `frame_path`。内壁为局部 Y 轴旋转体，必须来自实际测量。

`center` 和尺寸在指定局部坐标系内；没有坐标系时为世界米制坐标。粒子间距始终按世界米制计算。
柱状区域默认沿局部 Z 轴；通过坐标系可以改变方向。不支持剪切矩阵。
`experiments/config/liquid/particle_frames.yaml` 是已有场景坐标系的示例：
分别使用一个非均匀缩放坐标系和一个旋转坐标系，液体生成后不会重复继承父级变换。

## 在其他任务中复用

```python
# sample_volume 无 Isaac 依赖，可以在普通 Python 中预览/测试粒子布局。
from utils.liquid_particles import ParticleFluidParameters, sample_volume

# 以下导入和操作必须在 SimulationApp 创建之后。
from utils.particle_fluid import ParticleFluid, configure_particle_world

configure_particle_world(world, stage)  # 必须在相机与物理初始化前，仅需配置一次
points = sample_volume("box", center=[0.2, -0.1, 0.95], size=[0.04, 0.04, 0.03], spacing=0.004)
fluid = ParticleFluid(
    stage, scene_path=world.get_physics_context().prim_path,
    root_path="/MyLiquids/SourceA", positions=points,
    parameters=ParticleFluidParameters(spacing=0.004),
)
initial_state = fluid.snapshot()
# 每次 episode reset 时恢复；不要在 controller 中搬运粒子来制造倒液成功。
fluid.restore(initial_state)
```

另一个位置使用另一个 `root_path`；也可以把多次 `sample_volume` 的点合并，创建一个液体系统。
独立系统适合互不接触的液体；需要相互作用或混合的液体应合并到同一个系统中。
工具拒绝覆盖已有 prim、无效或空坐标、非有限参数、超大粒子分配和不匹配的恢复参数。
每次分配的候选网格及粒子状态上限为 250,000；不宣称任意分辨率都有可靠物理效果。

## 真实容器与回放

`sample_container` 接受显式内壁轮廓及 USD 变换；现有 `PourLiquid` 复用此函数与 `ParticleFluid`。
当前 benchmark 的烧杯和锥形瓶保留原采样顺序、原碰撞标定和成功门槛。
**在容器里生成点不等于已修复容器碰撞**：实心凸包仍会排斥液体，薄壁漏洞仍会泄漏。
新容器需要独立检查内壁、底部、瓶口、粒子分辨率及静置保液，再进行动作和严格回放测试。

`PourLiquid.configure_world(world, stage, cfg)` 支持 `liquid.physics_dt`，默认 `1/480` 秒，
控制和渲染仍为 60 Hz。锥形瓶任务使用 `1/960` 秒做接触稳定性修正，即每次动作 16 个物理子步。
必须在初始化前配置；禁止运行中改步长。改变步长后需要新采集，旧液体快照会被严格 replay 拒绝。
增加子步会增加计算时间，不代表已达到随机化成功率要求。

## 机械臂显示

液体任务在初始化前调用 `prepare_robot_visuals`，只在运行时层取消机器人局部实例化。
不编辑原机器人 USD，不改变可视网格、材质、碰撞或关节参数；不允许在运行/暂停状态热切换。
新数据的 `init_state.extra.robot_rendering` 记录该表示方式。含机器人的独立工具配置可显式提供
`robot_roots`；普通非液体任务不会自动改变机器人表示。

旧数据中缺失手臂的 RGB 不能因为物理 episode 曾成功就视为合格；需要重新生成并验收。

锥形瓶任务另行显式启用 `task.robot_inertia_mode: collision_geometry`，修正原 Panda 资源中
异常偏小的惯量。该选项与显示修复独立：保留原质量、重心和网格，仅在运行时从既有碰撞几何
推算惯量。这是均匀密度近似，不是厂家实测值。其他任务默认 `asset`，不改惯量；
新数据记录 `init_state.extra.robot_mass_properties`，不允许跨惯量版本 replay。

该任务同时设置 `task.gripper_max_joint_velocity: 0.05`（m/s，每指），修正资源中主动指
0.2 m/s、被动指 10 m/s 的不对称限速。该限制通过物理关节约束生效，不直接裁剪物体姿态或
给被动指增加驱动。液体初态记录该参数，不允许跨限速版本 replay。

## 重新采集与严格回放

锥形瓶的批测入口逐个 seed 新采集，每个 seed 仅尝试一次；成功后才回放该 seed 的新 HDF5。
原相机不变，失败不会自动重采或从分母消失。运行前选择新的输出目录：

```bash
python -m experiments.liquid.evaluate_conical_pairs \
  --output outputs/conical_pairs --seeds 901 902 903
```

默认只测 `picking2 → pouring → transporting2`，不能计作完整 Level 4；
加 `--scope full_level4` 才测完整任务。`results.json` 分别统计采集成功率、
成功采集后的回放成功率，以及以所有尝试为分母的配对成功率。
回放必须消耗全部高精度动作；不补动作，不绑定瓶子。运行中修改受跟踪的源文件会中止批测，避免混合版本。
这是脚本 controller 的验证，不是训练后的 VLA policy 测试。
