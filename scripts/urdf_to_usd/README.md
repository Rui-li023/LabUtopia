# URDF → USD 机械臂导入管线

把 ROS `*_description` 包里的机械臂批量转成 LabUtopia 能用的 USD 资产，并**可验证地**确认
结构、材质、尺度、物理都正确。

## 组成

| 文件 | 作用 |
|---|---|
| `robots.yaml` | 清单：每支臂的仓库 / 包 / 入口 / xacro 参数 / datasheet 臂展 |
| `prepare_urdf.py` | xacro 展开（或平坦 URDF 直通）+ `package://` 改写。**跑在独立 xacro venv** |
| `pipeline.py` | 转换与校验的纯逻辑，无 CLI 无 app 引导。假定 SimulationApp 已启动 |
| `convert.py` | 单支臂转换 CLI |
| `inspect_usd.py` | 单支臂校验 CLI |
| `batch.py` | 清单驱动的批量：**只启动一次 Isaac Sim** 跑完全部 |
| `contact_sheet.py` | 把所有预览图拼成一张总览（纯 Pillow，不需要 Isaac） |

## 快速开始

```bash
# 0) 一次性：装 xacro 到独立 venv（不要污染 isaacsim5.1 环境）
python3 -m venv /tmp/xacro_venv && /tmp/xacro_venv/bin/pip install xacro rospkg pyyaml

# 1) 拉源码（不进 git，third_party/ 已 gitignore）
mkdir -p third_party/urdf && cd third_party/urdf
git clone --depth 1 --branch melodic-devel https://github.com/ros-industrial/universal_robot.git
# ...其余仓库见 robots.yaml 的 repos 段
cd -

# 2) 全量转换 + 校验 + 渲染（isaacsim5.1 环境的解释器）
python scripts/urdf_to_usd/batch.py

# 只跑部分 / 复用已展开的 URDF / 跳过渲染
python scripts/urdf_to_usd/batch.py --only ur5e panda --skip-prepare --no-render

# 3) 总览图
python scripts/urdf_to_usd/contact_sheet.py
```

`batch.py` 用 isaacsim5.1 解释器跑，它会自己 shell out 到 xacro venv 做展开。

## 加一支新臂

在 `robots.yaml` 的 `robots:` 里加一条：

```yaml
  - name: my_arm
    vendor: SomeVendor
    repo: some_repo            # 需在 repos: 段声明，并已 clone 到 third_party/urdf/<repo>
    package: myarm_description # 靠 package.xml 定位，不需要 ROS 环境
    source: urdf/my_arm.xacro  # .xacro 会展开；.urdf 直接用
    args:                      # 可选：xacro $(arg)
      dof: "6"
    visual_rpy: [-1.5708, 0, 0] # 可选：修正 visual 网格自身的坐标轴
    reach_m: 0.85              # 仅供参考对照，不参与断言
    visual_meshes: dae         # dae/obj → 材质保留；stl → 必然平色
```


## 两个来源：官方资产优先，自转补缺

NVIDIA 用**同一个 URDF importer** 生成官方 Isaac 资产。实测对拍（`ur5e` / `xarm6`）：

| | 官方 | 自转 |
|---|---|---|
| 总面数 | 129347 / 100510 | **完全相同** |
| 材质名 | Black/JointGrey/URBlue/... | **完全相同** |
| 着色器 | `OmniPBR.mdl` | **相同** |
| 目录结构 | `x.usd` + `configuration/x_{base,physics,sensor}.usd` | **相同** |
| 材质 prim 数 | 5（去重） | 27（每 link 一份副本） |

同一套渲染代码下两者视觉无法区分。**官方版仍优先**：材质已去重，且多带一个
`*_robot_schema.usd` 语义层，由 NVIDIA 维护。

```bash
python scripts/urdf_to_usd/fetch_official.py --list    # 看服务器上有什么
python scripts/urdf_to_usd/fetch_official.py           # 按 robots.yaml 的 official: 段下载
```

产物落在 `assets/robots/official/<name>/`，该目录是可重建的下载缓存，不进 Git。
完整清单目前包含 24 个型号（约 325 MB）：
UR 全系 9 款、Franka Panda/FR3/Emika、Kinova Gen3/Jaco2、UFACTORY xarm6/7/lite6/uf850、
Flexiv Rizon4、FestoCobot、Clearpath RidgebackFranka/RidgebackUr、Robotiq 2F-85/2F-140。

**官方没有的**（需自转）：KUKA iiwa 系列、Fanuc LR Mate 200iD、Doosan、WidowX、
ARX 全系、Split ALOHA、AgileX piper 机械臂。

### 双臂合成

官方只有单臂 FR3。`compose_dual.py` 用 USD 引用把同一个资产放两次组成双臂，
文件只有几 KB 且改源自动同步：

```bash
python scripts/urdf_to_usd/compose_dual.py \
    --source assets/robots/official/franka_fr3/fr3.usd \
    --out assets/robots/fr3_duo.usd --separation 0.9
```

**注意：两臂相对位姿是假设值**（并排、间距 0.9 m、同朝 +X）。真实双臂平台构型各异，
用于实际任务前需按硬件调整 `--separation` / `--yaw-deg` / `--mount-height`。

## 选型：材质保真只取决于源网格格式

`ImportConfig` **没有任何材质相关选项**（全部字段：`distance_scale / merge_fixed_joints /
fix_base / make_default_prim / create_physics_scene / import_inertia_tensor / self_collision /
default_drive_type / default_drive_strength / convex_decomp / collision_from_visuals /
parse_mimic / replace_cylinders_with_capsules / override_joint_dynamics / density`）。
材质完全继承自源 **visual** 网格：

| 格式 | 材质 |
|---|---|
| DAE (COLLADA) | 带材质/贴图 → 保留 |
| OBJ + MTL | 带材质 → 保留 |
| STL | **无材质** → 只剩 URDF `<color rgba>`，纯平色 |

**选仓库比调参数重要。** 想要材质，先确认 visual 是 DAE/OBJ。
若只有 STL，回退方案是先用 assimp/trimesh 转 OBJ+MTL（NVIDIA 对自带的 ur10 就是这么做的）。

注意：同一厂商不同入口的网格格式可能不同。xArm 仓库里有 42 个 DAE，但
`xarm_device.urdf.xacro` 这条路径引用的全是 STL；Kinova ros_kortex 的 `gen3.xacro` 同理。
以 `prepare_urdf.py` 打印的 `mesh references by extension` 为准，别信仓库总量。

## 校验做了什么

`pipeline.verify()` 六项，全过才算 OK：

1. **articulation** — 有且仅有一个 prim 带 `UsdPhysics.ArticulationRootAPI`
2. **joint set** — 可动关节数与 URDF 一致（含 `continuous`）
3. **joint limits** — 逐个与 URDF 比对（USD 是**度**，URDF 是**弧度**，换算后比）
4. **materials** — 每个 visual mesh 都绑到材质；**按漫反射颜色去重**的种类数达标
   （DAE/OBJ 源要求 ≥2，STL 源要求 ≥1）
5. **scale** — 世界包围盒最大跨度落在合理带内（0.15–3.5 m），防单位陷阱
6. **physics** — 真正加载成 articulation，DOF 数对，位置驱动能响应

## 踩过的坑

1. **PyPI 的 `xacro` 是 ROS2 版**，`$(find pkg)` 走 `ament_index_python` 而非 `ROS_PACKAGE_PATH`，
   而 `ament_index_python` 不在 PyPI 上。`prepare_urdf.py` 往 `sys.modules` 注入 stub 解决
   （xacro 那处是函数内局部 import，所以注入有效）。
2. **importer 不解析 `package://`**。NVIDIA 自带的 ur10 示例用的是相对路径 `../meshes/*.obj`。
   `prepare_urdf.py` 负责改写成相对路径。
3. **`default_drive_type` 要枚举不是 int**，传 `1` 会 `TypeError`。
   用 `_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION`。
4. **Kit 会接管 stdout**，`print` 会被吞。所有进度输出走 stderr。
5. **产物是分层的**：`<name>.usd`（约 1 KB 薄壳）+ `configuration/<name>_{base,physics,robot,sensor}.usd`。
   **两者必须一起走**，只拷入口文件会得到空壳。这与仓库里其它自包含单文件 USD
   （Franka.usd / piper.usd）的惯例不同。
6. **遍历必须带 instance proxy**：importer 把每个 link 的 `visuals`/`collisions` 标成
   `instanceable=True`，普通 `stage.Traverse()` 到此为止，一个 mesh 都看不到。
   要用 `Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())`。
7. **材质常按 GeomSubset 绑**：一个 DAE mesh 上有多种材质时，材质绑在 `UsdGeom.Subset`
   子 prim 上，直接问 mesh 要 `ComputeBoundMaterial()` 返回空。必须同时查 subset。
8. **材质要按颜色去重，不能按名字**。Kinova 的 DAE 把每个材质都命名成 `Material_001`，
   按名字去重会把 32 个真材质塌成 1 个，误报"材质丢失"（实际有 3 种不同灰度）。
9. **`World` 是单例**。批量循环里第二次 `World(...)` 拿到的是已被 PhysX 失效的实例，
   下一个 `SingleArticulation` 会死在 `'NoneType' object has no attribute 'link_names'`。
   每轮必须 `create_new_stage()` + `World.clear_instance()`。
10. **`continuous` 关节没有 lower/upper**。只按 `revolute|prismatic` + 有 `<limit>` 来数关节
    会漏掉它们（Kinova 大量使用），导致"USD 关节比 URDF 多"的误报。它们是 DOF，要计数但不比限位。
11. **mimic 从动关节的限位会被 importer 重算**，不是照抄 URDF。实测 widowx_vx300s：
    主动关节 `left_finger` 原样通过，从动 `right_finger` 被改写
    `[-0.057,-0.021]` → `[-0.0642,-0.0138]`。校验时跳过其限位比对。
    **需要精确夹爪限位的话要手工复核。**
12. **USD 角度限位是「度」，URDF 是「弧度」**，比较前要换算。
13. **别拿「高度」判尺度**：机械臂 URDF 零位通常是水平伸直而非立姿。用最大跨度对臂展。
14. 转换时会刷 `Unresolved reference prim path .../visuals/<link>` 警告，
    对应的是 URDF 里**没有 `<visual>` 的纯坐标系 link**（UR5e 是 `base_link` / `base` /
    `flange` / `tool0`）。importer 给每个 link 无脑建 `visuals` scope，空的就报。**无害。**
15. **Doosan 仓库的 `dsr_description/urdf/*.urdf` 是过时的**——引用 `meshes/m1013/`，
    但仓库已改成配色变体目录 `m1013_white/` / `m1013_blue/`。要用
    `xacro/m1013.urdf.xacro` 并传 `--arg color:=white`。
16. **Replicator 默认相机的近裁剪面在 ~1 m**。台面尺度的臂完全落在里面：把 0.55 m 的臂
    贴近取景会让相机停在 0.75 m 处，整个机器人被近裁剪掉，渲出一张纯灰空图；
    刚好卡在 1 m 附近则出现局部黑块。必须显式传
    `rep.create.camera(..., clipping_range=(0.01, 1000.0))`，而不是靠退远规避。


### 预览渲染的坑（都会让"资产其实没问题"看起来像有问题）

17. **URDF 零位不是好的展示姿态**，而且零位下"看不出是机械臂"往往不代表几何有错。
    实测：Kinova Gen3 零位 bbox 是 `(0.093, 0.128, 1.19)`——一根 1.19 m 高、9 cm 粗的竖杆；
    xArm 零位完全折叠回自身；Fanuc LR Mate 小臂正好指向固定相机被透视压成一坨。
    **判定几何对错的正确方法是摆个非零姿态再看**，三者摆开后都是标准机械臂。
    因此预览统一先摆 `SHOWROOM_POSE_RAD` 再渲染。
18. **固定相机方位角会压扁一部分臂**。按 bbox 的水平长边自动选方位
    （x 长就沿 −y 看，y 长就沿 +x 看），让最长方向横铺在画面里。
19. **摆姿要同时改状态和驱动目标**。只 `set_joint_positions()` 的话，位置驱动的目标仍是 0，
    随后的物理步会把手臂拉回零位。必须同时 `apply_action(ArticulationAction(joint_positions=...))`。
20. **摆姿后不要调 `world.stop()`**。`stop()` 会恢复 play 之前的初始状态（等于撤销姿态），
    而且会拆掉 articulation view，之后 `get_joint_positions()` 返回 `None`。
21. **通用展示姿态会被关节限位截断**。xArm 的 joint3 上限约 0.19 rad，通用模式给 +0.5 会被
    clamp 回折叠态。这类臂在 `robots.yaml` 里用可选的 `preview_pose` 覆盖。
22. **背景要用深色**。多数臂是白/浅灰本体，浅色背景下轮廓会糊掉（Gen3 只有 1 种颜色时尤其明显）。
23. **RTX 要累积够帧数**。50 帧不够，大面积平色表面会留下彩色噪点（Fanuc 那身黄最明显），
    现用 150 帧。

## 已转换成果（2026-08-13，20/20 全部通过校验）

`colours` = 视觉网格上去重后的漫反射颜色数（贴图驱动的材质按贴图区分）。

| 机器人 | 厂商 | DOF | 源网格 | colours | span m | MB |
|---|---|---|---|---|---|---|
| `arx_lift2s` | ARX | 17 | glb | 10 | 0.97 | 46.5 |
| `arx_r5` | ARX | 8 | glb | 7 | 0.53 | 4.5 |
| `arx_x5` | ARX | 8 | glb | 6 | 0.53 | 10.9 |
| `piper_agilex` | AgileX | 8 | dae | 9 | 0.57 | 28.7 |
| `split_aloha` | AgileX | 42 | stl | 18 | 1.15 | 32.2 |
| `doosan_m0609` | Doosan | 6 | dae | 4 | 1.04 | 4.1 |
| `doosan_m1013` | Doosan | 6 | dae | 3 | 1.45 | 7.1 |
| `fanuc_lrmate200id` | Fanuc | 6 | stl | 3 | 0.76 | 1.9 |
| `fr3` | Franka Robotics | 7 | dae | 8 | 1.11 | 7.1 |
| `panda` | Franka Robotics | 7 | dae | 8 | 1.11 | 7.1 |
| `iiwa14` | KUKA | 7 | stl | 2 | 1.31 | 10.1 |
| `iiwa7` | KUKA | 7 | stl | 2 | 1.27 | 17.3 |
| `gen3_7dof` | Kinova | 7 | stl | 1 | 1.19 | 6.0 |
| `jaco2_j2n6s300` | Kinova | 12 | dae | 3 | 0.55 | 5.4 |
| `widowx_vx300s` | Trossen / Interbotix | 9 | stl | 3 | 0.76 | 1.9 |
| `xarm6` | UFACTORY | 6 | stl | 2 | 0.59 | 3.6 |
| `xarm7` | UFACTORY | 7 | stl | 2 | 0.60 | 2.9 |
| `ur10e` | Universal Robots | 6 | dae | 4 | 1.33 | 8.1 |
| `ur3e` | Universal Robots | 6 | dae | 4 | 0.55 | 7.1 |
| `ur5e` | Universal Robots | 6 | dae | 4 | 0.93 | 6.1 |

合计约 219 MB。官方资产可用 `fetch_official.py` 按需重建；只有被运行时配置直接引用的
转换产物才作为 Git LFS 资产提交。

`robots/<name>/` 下提交的 URDF 是 Lula/RMPFlow 使用的纯运动学模型，不包含 visual、
collision 或 mesh 引用。完整 URDF 只存在于可重新拉取的 `third_party/` 转换缓存中；渲染和
物理几何由 `assets/robots/` 下的 USD 提供，RMPFlow 碰撞体由 `robot_descriptor.yaml` 提供。


### 多来源与材质来源的坑

24. **`glb`/`gltf` 也带材质**，判断"源是否带材质"时不能只认 dae/obj。
    ARX 全系用 GLB，漏判会把材质门槛降到 1，掩盖真实的材质丢失。
25. **不同来源的着色器颜色输入字段名不同**：UsdPreviewSurface 是 `diffuseColor`，
    OmniPBR 是 `diffuse_color_constant`，**glTF/GLB 导入是 `gltf/pbr.mdl` 的
    `base_color_factor`**。只探前两个会把 ARX 的 7 种材质误报成 1 种。
26. **URDF 里 `<visual>` 内的 `<material>` 会覆盖网格自带材质**。AgileX piper 给每个
    visual 挂了一个平色，把 DAE 里各 13–15 种材质全盖掉。
    用 `strip_visual_materials: true` 剥离后恢复到 9 种颜色 / 94 个材质名。
27. **剥离 XML 标签不能用正则**。`<material name=""><color rgba=".."/></material>` 里
    非贪婪匹配会停在内层 `/>`，留下孤儿 `</material>` 把文件改坏。用 ElementTree。
28. **有些仓库把非包目录当包引用**：mobile_aloha 的 URDF 写
    `package://tracer/tracer_description/meshes/...`，而 `tracer/` 下没有 package.xml。
    `find_package_roots` 用目录名做回退解析。
29. **一个机器人的描述可能横跨两个仓库**：ARX 的 xacro 在 robot-descriptions-arx，
    网格在 robot-descriptions-common 的 `component_models`。用 `--extra-repo` / `extra_repos`。
30. **顶层入口不一定是型号同名文件**。`arx5_description/xacro/X5.xacro` 是**宏定义**，
    展开出 0 link；真正的入口是 `robot.xacro` + `--arg type:=x5`。
    展开后务必检查 link/joint 数，别只看命令成功。
31. **`--only` 不能覆盖整个 summary**，否则未跑到的机器人记录会丢，联络表图注随之缺失。
    `batch.py` 改为合并写入。
32. **GLB 的 up-axis 不一定与 URDF 一致**。ARX 的 GLB 是 Y-up，URDF/Isaac Sim 是 Z-up；
    importer 不会根据 GLB 元数据自动校正，结果是关节运动学正确但视觉连杆像爆炸图一样错位。
    在清单中设置 `visual_rpy: [-1.5707963267948966, 0, 0]`，转换前会把修正与每个既有
    `<visual><origin>` 做刚体旋转合成；collision 不受影响。

## 运行时接入状态

转换管线之外，仓库现已提供 22 个新机器人适配器及对应的 Level-1 pick 配置：机器人类声明
关节/夹爪/TCP，RMPFlow 使用各型号自己的 Lula 描述，`trajectory_controller.py` 从机器人实例读取
motion config，factory 采用显式注册。带裸法兰的工业臂使用与机械臂合成到同一 articulation 的
Robotiq 2F-85 资产。

“USD 校验通过”和“任务可采集”是两个不同门槛：前者只证明 articulation、关节、材质、尺度和
物理响应正确；后者还要求三段 pick 轨迹可达、夹爪接触稳定且成功条件连续满足。配置中
`collector.type: default` 的型号可进行真实采集，`collector.type: mock` 的型号目前只作为加载/渲染
烟雾配置，不应计入 benchmark 成功率。可用以下命令逐个生成单 episode 结果清单：

```bash
PYTHON_BIN=/path/to/isaac-python bash scripts/render_all_arm_picks.sh
```
