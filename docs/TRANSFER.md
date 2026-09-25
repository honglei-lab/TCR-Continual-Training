# 上传、下载与训练交付

## 初始化发布目录

```
Velixx/TCR/
  pi05-continued-training/initializations-v1/
    manifest.json
    base/      # model.safetensors + config + processors + tokenizer
    soup/
    regmeanpp/
    featcal/
    tcr/
```

每组 dense 模型约 9.35 GB；全部文件约 46.94 GB。
不上传旧 Base 联合训练 adapter，不上传单专家，不修改 ModelScope 根目录文件。
config.json 的过期服务器路径在发布副本中清空；模型参数字节保持不变。

维护者本机的 `local/sources.json` 记录原始路径，已被 Git 忽略。

```bash
python scripts/prepare_initializations.py --sources local/sources.json
python scripts/modelscope_login.py
python scripts/modelscope_upload.py           # 只显示计划
python scripts/modelscope_upload.py --execute # 实际上传、核对远端大小与 SHA256
python scripts/start_upload.py               # 或后台并行上传，最多三轮失败重试
```

登录脚本隐藏输入 token。不要把 token 写入命令行、README、配置或聊天。
公开下载无需写 token；公开仓库仍不能被匿名用户写入。
上传脚本只允许固定子目录：已有同 hash 文件跳过；已有不同 hash 或无法核验的文件
拒绝覆盖。网络中断后重跑；只有已提交完成的文件可跳过，单文件未提交的数据可能需重传。
后台日志位于 `local/modelscope-upload.log`，PID 在 `local/upload_process.json`。
`local/upload_verified.json` 是全量核验完成凭证。没有它不能声称上传完成。
下载脚本在 manifest 尚未发布时会失败，应等发布完成再使用。

## 训练服务器需要带走

1. 本代码仓库（包含 vendor 源码、五组身份清单、四份 episode 划分）。
2. 五组完整初始化目录（也可从 ModelScope 下载）。
3. 同一份 40-task LeRobot LIBERO 数据与 DATASET_RECEIPT.json。
4. clean LIBERO 仿真资产及 BDDL/init files。不要带旧服务器的绝对路径 config.yaml。

## 训练后传回

- 每组每 seed 的 `launch_node*.json`、最终配置、stdout/stderr 日志。
- 每个评测步数的 `pretrained_model/` **整个目录**：adapter、config、processor、tokenizer。
- 每个 checkpoint 对应的完整 `training_state/`（需要继续恢复训练时不可省略）。
- 所有套件的 `eval_info.json`、evaluation_receipt.json，以及最后汇总 JSON。
- 可少量保存视频作核查，不必把所有视频传回。

Adapter 不是独立模型。必须能追溯到 `configs/initializations.json` 中相应 dense
初始化的 SHA256；移动目录后必须修正 adapter 副本的 base_model_name_or_path。
不要修改或覆盖原初始化文件，不要把训练后权重放进 initializations-v1。
后续模型如需发布，应另用 `pi05-continued-training/runs/<run-id>/<arm>/seed<seed>/`。

## 64 卡验收

全局 batch=128、梯度累积=1 时每卡 batch=2；所有五组使用同一 GPU 拓扑。
先做 2 步 smoke，检查 64 个 rank 都加入、loss 有限、保存后 adapter 可重新加载。
再测少量稳态更新的吞吐，确认确有加速，最后才跑正式预算。
本地静态/导入测试不等于这项多机验收通过。
