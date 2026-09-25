# Table 6：训练服务器评测执行单

目标：比较五种初始化在相同 LIBERO 多任务继续训练预算下的起点、终点和学习曲线。
**测 clean LIBERO 闭环任务成功率，不是只测训练 loss 或离线动作 MSE。**
本页同时覆盖 Table 6 和附录的分套件训练前后结果；二者复用同一批评测。

## 1. 每个 checkpoint 测什么

| 套件 | 任务数 | 每任务回合 | 合计 |
|---|---:|---:|---:|
| Spatial (`libero_spatial`) | 10 | 10 | 100 |
| Object (`libero_object`) | 10 | 10 | 100 |
| Goal (`libero_goal`) | 10 | 10 | 100 |
| Long (`libero_10`) | 10 | 10 | 100 |
| 总计 | 40 | — | **400** |

五组是 `base / soup / regmeanpp / featcal / tcr`，均测四个套件。
不要只测某个专家原来的套件；这里每组都是四任务集共同继续训练。
不额外跑 LIBERO-PRO、RoboTwin、真机，不重新融合，也不增加单专家组。

固定：`eval_seed=274001`、`eval.batch_size=1`、`hard_reset=true`、`init_states=true`、
`n_action_steps=10`、原生环境时限。`--eval.n_episodes=10` 是**每任务**十回合。
用本仓库的同一 clean LIBERO 版本、资产、初始状态顺序和评测入口。
不要改用服务器上旧的 PRO 环境或其他 reset bank；不要手动设置 60 秒真机时限。

## 2. 哪些训练步数需要评测

以正式训练启动时冻结的 `configs/table6.json` 为准。仓库建议值是：

```text
training_seeds = 1000, 1001, 1002
eval_grid = 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
B = 5000 optimizer updates
```

- **step 0**：该组原始 dense 初始化，不是最早保存的 500 步 adapter。
- **step > 0**：该组、该训练 seed、该精确步数保存的 `pretrained_model/`。
- 不把 epoch、样本数、64 卡各 rank 的步数之和当作横坐标。
- 若实际训练采用不同 B 或保存间隔，先回传启动配置并统一评测网格；不要直接套用本页示例。
- 已保存的点可以立刻评测，不需要等全部训练结束。先确认 checkpoint 保存完整，不能读取仍在写入的文件。
- 时间紧可先交付所有组的 step 0 和 B，再补预先规定的中间点；中间点未齐时不报告正式 nAUC。
  不因分数高低删点、换“best checkpoint”或只给 TCR 增加评测次数。

默认完整规模是 5 组 × 3 个训练 seed × 11 点 × 400 = **66,000 回合**。
训练 seed 表示独立训练重复，评测 seed 则固定用于配对；不要把二者混为一谈。
step 0 没有训练 seed 差异；本包默认目录要求各 seed 完整。若以后显式复用一次
step-0 测量，必须记录其共同来源，不能伪装成三次独立评测或据此扩大样本量。

## 3. 接收服务器先准备什么

```bash
git pull --ff-only
source .venv/bin/activate
python scripts/doctor.py
python -m pytest tests -q
python scripts/setup_libero.py --download-assets
export LIBERO_CONFIG_PATH="$PWD/local/libero-config"
export MUJOCO_GL=egl
```

已有正确资产和 `local/libero-config/config.yaml` 时跳过 setup，不覆盖旧配置。
安装和手动指定资产路径见 README。评测只需 clean LIBERO 环境、模型及 tokenizer；
闭环评测不再读取训练示范数据。

训练输出是 **LoRA adapter + 四个完整接口模块**，不是独立 dense 模型。
必须保留其 `adapter_config.json` 指向的对应组初始化（例如 TCR adapter 必须加载 TCR
初始化，不能加载 Base 初始化）。换服务器后只在工作副本中修正路径，并核对该初始化
与 `configs/initializations.json` 的模型 SHA256；不能只复制 adapter 后直接推理。

训练保存器不保证每个 checkpoint 都复制 tokenizer。本次评测入口支持显式
`--tokenizer`，并能默认从 adapter 的 base 目录寻找；显式路径也会覆盖 processor
中保存的旧服务器 tokenizer 路径。不会改写权重或归一化统计。

## 4. 直接运行：一个 checkpoint 自动测四套件

在仓库根目录执行。以下 TCR 示例只需按实际目录替换 arm、训练 seed 和训练 step。
`--output` 中的 `seed1000` 是**训练 seed 标签**，不是把评测 seed 改成 1000。

先测训练前：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --config configs/table6.json \
  --checkpoint checkpoints/initializations-v1/tcr \
  --tokenizer checkpoints/initializations-v1/tcr/tokenizer \
  --output outputs/eval/tcr/seed1000/step000000 \
  --execute
```

再测已保存的 500 步（同理替换为 001000、001500、…、005000）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --config configs/table6.json \
  --checkpoint outputs/formal/tcr/seed1000/train/checkpoints/000500/pretrained_model \
  --tokenizer checkpoints/initializations-v1/tcr/tokenizer \
  --output outputs/eval/tcr/seed1000/step000500 \
  --execute
```

去掉 `--execute` 只打印四条命令，不占 GPU。
脚本每个进程顺序评测四套件。可在不同空闲 GPU 上同时评测不同 arm/seed/step，
默认每卡一个评测进程；**不要用 64-rank torchrun 启动同一个评测命令**，否则会重复
评测、争写同一输出。64 卡训练不要求一次评测也占 64 卡。

输出目录已有内容时脚本拒绝覆盖，包括只完成部分套件的情况。不要直接删除失败证据；
先核对已完成结果，必要时用新的 attempt 路径重跑，最终由负责人选择完整、协议一致的
记录归档到正式目录，不能静默覆盖或拼接重复回合。

## 5. 每个点交付什么指标

| 指标 | 定义 |
|---|---|
| 每套件成功率 | 每套件 100 个原始回合的成功比例 |
| 总成功率 S(t) | 四套件等权平均；因回合数相同，也等于总成功数 / 400 |
| S(0)、S(B) | 训练前和统一训练预算终点的成功率，不是历史主表均值或训练中最高值 |
| nAUC | 在预设 step 网格对 S(t) 做梯形积分，再除以 B，结果仍以百分数表示 |
| Tτ | 首个达到共同阈值 τ 的**已测更新步数**，不插值；未达到记 `>B`，起点达到记 0 |

`threshold_percent` 目前为 null，尚未约定阈值。可以先完成评测，但不得自行填一个有利
阈值或报告 Tτ 数字。若在看过曲线后才选阈值，不能称其为预先规定的比较。
汇总器对 S(0)、S(B)、nAUC 输出三个训练 seed 的均值和样本标准差；Tτ 保留各 seed
的值，不能把 `>B` 当作 0 或 B 后直接平均。

附录分套件表直接使用同一数据的 `S_k(0) → S_k(B)` 和差值，无需另跑实验。
训练 loss、离线动作 MSE、视频可作辅助核查，但均不能替代闭环成功率。

## 6. 汇总和回传

必须保留以下目录结构；四个 suite 子目录中都应有原始 `eval_info.json`：

```text
outputs/eval/<arm>/seed<training_seed>/step<6位更新数>/
  evaluation_receipt.json
  libero_spatial/eval_info.json
  libero_object/eval_info.json
  libero_goal/eval_info.json
  libero_10/eval_info.json
```

全部网格完成后：

```bash
python scripts/summarize.py \
  --config configs/table6.json \
  --results outputs/eval \
  --output outputs/table6_summary.json
```

缺任务、缺回合、缺 checkpoint、缺训练 seed 或 receipt 中配置不同，都会拒绝汇总。
不要插值、补零、复制不同组结果或把缺失结果填进论文。汇总文件已存在时换一个新文件名。

**先传这些小文件就能核验结果和填表，无需等几十 GB 权重传完：**

1. 本次代码 commit、实际冻结的配置、`launch_node*.json` 和训练日志。
2. 全部 `outputs/eval/` 下的 `evaluation_receipt.json`、四套件原始 `eval_info.json`、评测日志。
3. 完整后生成的 `outputs/table6_summary.json`；未齐时可先回传原始结果并列出缺项。
4. checkpoint 对照表：arm、训练 seed、更新步数、路径、adapter 和对应 base 身份。

权重随后传回每个规定评测点的完整 `pretrained_model/`（及所依赖的初始化），续训还需
`training_state/`。视频只需少量核查样例，不必批量上传。完整文件要求见
[传输文档](TRANSFER.md)。本仓库的初始化上传器**不会**自动上传训练结果或评测 JSON。

## 给训练服务器执行者的一句话任务

> 按启动时冻结的同一配置，对 Base、Soup、RegMean++、FeatCal、TCR 的每个规定训练
> seed 和保存步数做四套件 clean LIBERO 正式评测，每任务 10 回合、每点 400 回合；
> step 0 测对应初始化，step > 0 加载对应 adapter 和正确 base。先交付所有组起点、
> 终点，再补预设曲线点。保存原始逐回合结果和 receipt，不选最高分、不改评测协议。
> 测完用 summarize.py 汇总，并回传配置、启动记录、日志、原始 JSON 和汇总 JSON。
