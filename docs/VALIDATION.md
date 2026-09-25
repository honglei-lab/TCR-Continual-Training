# 本地验证记录（2026-09-25）

已完成：

- 五组 dense 初始化完整 SHA256 与论文实验身份记录一致。
- 原始五组模型的 tensor names/shapes/dtypes 一致：813 个张量；完整参数文件均为 9,354,050,784 字节。
- 五组发布副本的 processor 与 tokenizer 均在 CPU 成功加载。
- 五组正式 64-rank 训练参数均通过 LeRobot 原生配置解析和 validate()：
  dp_replicate=64，per-rank batch=2，global batch=128，累积=1，LoRA rank=64，1693 episodes。
- 原实验 Python 3.12 环境中的关键依赖版本、torchcodec、π0.5 与训练模块导入检查通过。
  FFmpeg 6 的 lib 路径需要放入 LD_LIBRARY_PATH；未配置时 torchcodec 导入失败。
- vendored LeRobot 构建元数据 dry-run 通过（`pip install --dry-run --no-deps`）。
- Python 编译检查、shell 语法检查、6 项轻量单元测试通过。

没有完成、不能据此声称通过：

- 全新机器从零安装全部依赖（本地使用已存在的原实验环境）。
- 完整模型训练 forward/backward、梯度及 loss 数值检查。
- 64 卡跨节点 NCCL 运行、吞吐或加速比。
- 本包新训练 checkpoint 的保存→重新加载→原生 rollout。
- 新的 Table 6 成功率、学习曲线或统计结果。

正式跑前仍需 README 中的两步 smoke 与 rollout 验收。不会因为脚本存在就把实验记为完成。
ModelScope 上传是独立长任务，只有 `local/upload_verified.json` 存在才表示完整远端核验成功。
