# TCR-Continual-Training

π0.5 / LIBERO 继续训练实验（Table 6）的独立运行包。
五组统一重新训练：**Base、Model Soups、RegMean++、FeatCal、TCR**。
没有单专家组；不重新融合；不训练 FastWAM/OpenVLA；不包含历史实验目录。

**训练服务器拿到各步 checkpoint 后，按 [Table 6 评测执行单](docs/EVALUATION.md) 运行。**
每个点测四套件 400 回合；该文档包含可复制命令、并行方式、指标定义和回传清单。

## 当前状态

代码已具备独立训练、评测与汇总入口；本地验证范围见 [验证记录](docs/VALIDATION.md)。初始化权重上传目标为
`https://modelscope.cn/models/Velixx/TCR` 下的
`pi05-continued-training/initializations-v1/`。
**只有上传结束且远端 SHA256 核验通过，才能认为模型发布完整。**
GitHub：`https://github.com/honglei-lab/TCR-Continual-Training`。
本包不宣称已经验证 64 卡训练，也不会自动启动训练。

## 1. 实验口径

`configs/table6.json` 是统一配置。当前给出可修改的建议预算：

| 项目 | 所有组共同设置 |
|---|---|
| 数据 | `lerobot/libero`，Spatial/Object/Goal/Long，40 任务、1693 条示范 |
| 训练预算 | 5000 更新，warmup 500，cosine 到 5000；正式开跑前确认预算 |
| 全局 batch | 128；64 卡时每卡 2，梯度累积 1 |
| 参数 | 新建 LoRA r64/alpha64 + action_in/out、time_mlp_in/out 四个完整模块 |
| 学习率 | 3e-4，AdamW，沿用 π0.5 配置中的其他 optimizer 参数 |
| 随机种子 | 1000/1001/1002，五组配对；LoRA 初始化固定 3407 |
| 存储 / 评测 | 0、500、1000、…、5000 更新；每任务 10 回合，共 400 回合/评测点 |

5000 更新是本包的**建议值，不是用户已确定的预算，也不是已经完成的结果**。
若需其他预算，在所有组开跑前统一修改同一份 JSON，并保留版本。
完整设置为 5×3×11×400=66000 个评测回合；训练提速不代表仿真评测同步提速。
时间紧可以在开跑前统一减少 eval_grid，但必须包含 0 和 B，不能看结果后删点。
`threshold_percent` 尚未选定；不自动假造阈值或 Tτ 数字。

Base 是四个专家共同使用的 theta0（已有 200 步 LIBERO 适配），不是原始官方权重。
旧 15k 联合训练权重只作参考，不替代本次重新训练的 Base。
TCR 使用已选定的 c_e，三个训练 seed 都从同一份 c_e 开始。
见 [来源说明](docs/PROVENANCE.md)。

## 2. 环境

Linux x86_64，Python 3.12，支持 CUDA 12.8 的驱动，64 卡示例为 8 节点×8 卡。
GPU 型号和节点网络尚未确认，不能承诺 64 卡线性加速。
先安装 uv，再运行：

```bash
bash scripts/install.sh
source .venv/bin/activate
python scripts/doctor.py
python -m pytest tests -q
```

安装脚本拒绝覆盖已有 `.venv`。关键依赖锁定在 `requirements-constraints.txt`，
不是完整跨平台 lockfile。torchcodec 0.3.0 需要兼容的 FFmpeg 共享库；本机原实验使用
FFmpeg 6。若 doctor 导入 torchcodec 失败，先配置 FFmpeg 库目录至 `LD_LIBRARY_PATH`。
不要跳过 doctor 后直接启动 64 卡作业。
LeRobot 依赖源码已放在 `vendor/lerobot`，无需访问原研究仓库。

## 3. 下载初始化与数据

权重发布完成后（五份模型合计约 46.94 GB，缓存与目标双份存储预留约 100 GB）：

```bash
python scripts/download_initializations.py
python scripts/prepare_data.py --root /data/libero --download
```

公开权重下载不需要写 token；上传才需要。
已有完整 LeRobot 数据可用 `python scripts/prepare_data.py --root /data/libero` 核验并生成清单。
数据脚本固定下载 revision，并检查 1693 episodes / 40 tasks；不下载 LIBERO-90，
不进行再采集。各节点必须使用相同数据文件和 DATASET_RECEIPT.json。
训练按数据帧采样，不宣称逐任务严格等量采样；五组使用完全相同数据和采样设置。

## 4. 先检查命令，再训练

所有启动器默认只打印命令，只有加 `--execute` 才执行。
64 卡、8 节点示例，在**每个节点分别运行**；修改 `--node-rank` 为 0…7，
`--master-addr` 设为节点 0 可达地址。代码、checkpoint、数据路径在各节点保持相同。

```bash
python scripts/train.py \
  --arm base --seed 1000 \
  --dataset-root /data/libero \
  --nodes 8 --gpus-per-node 8 --node-rank 0 \
  --master-addr 10.0.0.1 --master-port 29500
```

先加 `--smoke --execute` 跑两步，验证有限 loss、保存 checkpoint、重新加载、
以及至少一次原生 LIBERO rollout；这是接收服务器的验收步骤，不是本地已完成的测试。
验收通过后移除 `--smoke`，添加 `--execute` 正式训练。
将 `--arm` 换为 soup/regmeanpp/featcal/tcr，再分别运行另外两个 seed。
调度器应为每次作业分配相同拓扑；本包不自行抢卡、杀进程或提交集群作业。

禁止自动从旧 optimizer 恢复：每组从初始 dense checkpoint 新建 adapter 和 optimizer。
新作业拒绝覆盖已有输出。中断恢复需先人工核对同一配置、拓扑、checkpoint 的全部
training_state 文件，再使用 LeRobot 的显式 resume；本包不自动猜测恢复点。

**不要把每卡 batch 保持为旧值 32 再直接上 64 卡**，那会把全局 batch 改为 2048。
本包固定梯度累积为 1：当前 vendored 上游按训练循环计 step，多步累积不能直接等同
optimizer update 预算，因此不开放该参数。所有正式组必须使用一致拓扑。

## 5. 原生 LIBERO 评测

需要官方 clean LIBERO 的 `bddl_files/`、`init_files/`、完整 `assets/`。
hf-libero 0.1.4 安装包提供环境代码、BDDL 与 init files；资产可用以下命令固定版本下载：

```bash
python scripts/setup_libero.py --download-assets
```

已有完整资源则使用下面的指定路径命令（二选一，不要重复覆盖 config.yaml）。
不要混入 LIBERO-PRO 的资产或配置。

```bash
python scripts/setup_libero.py \
  --benchmark-root /data/LIBERO/libero/libero \
  --assets /data/LIBERO/libero/libero/assets
```

配置完成后：

```bash
export LIBERO_CONFIG_PATH="$PWD/local/libero-config"
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint checkpoints/initializations-v1/base \
  --output outputs/eval/base/seed1000/step000000
```

核对打印命令后加 `--execute`。step>0 时 checkpoint 换为
`outputs/formal/base/seed1000/train/checkpoints/000500/pretrained_model` 等。
训练 checkpoint 不一定附带 tokenizer；可加 `--tokenizer checkpoints/initializations-v1/base/tokenizer`
显式指定对应初始化的 tokenizer。初始模型是 dense；训练输出是 **adapter + 四个完整接口模块**，评测需原始初始化仍位于
adapter_config.json 所记录的位置。转移服务器时也需迁移对应 dense 初始化并修正
副本中的 base_model_name_or_path；不能把 adapter 当成完整 π0.5 权重。

每个评测点重新创建 clean LIBERO 环境，hard_reset=true，固定 batch=1、seed、
原生 initial-state 顺序；不要混用不同并行环境数量。并非正文主表的原始回合复现。
S(0) 必须重新测，不能直接填旧表 77.83% 或被选中的 TCR 历史 80.50%。
同一 arm 的 S(0) 没有训练 seed 差异，若复用同一原始记录，不能算三次独立评测。

```bash
python scripts/summarize.py \
  --results outputs/eval \
  --output outputs/table6_summary.json
```

目录必须为 `arm/seed1000/step000000/libero_spatial/eval_info.json` 等。
脚本要求五组全部 seed 与全部评测点完整，并核对 evaluation_receipt.json 中的统一配置；缺失会报错，不插值或补零。
计算 S(0)、S(B)、梯形 nAUC、各 suite 曲线；Tτ 未达到记 `>B`，不删失败 seed。

## 6. 需要传回哪些文件

见 [传输和上传说明](docs/TRANSFER.md)。至少保留配置、启动记录、训练日志、原始
eval_info.json、所有评测节点 adapter/processor/tokenizer；续跑还需完整 training_state。
权重不要进 Git。代码、配置、文档、tests、vendor 源码进 Git；local/、outputs/、
checkpoints/、token、数据、视频、虚拟环境不进 Git。
