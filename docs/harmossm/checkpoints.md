# 训练记录与模型文件

[返回目录](README.md) · [项目首页](../../README.md)

输出目录为 `HarmoSSM/checkpoints/<resolved_run_name>/`；可视化位于其 `vis/`。
每个 run 记录：

```text
*_efficientnet_b2_best_primary.pth
*_efficientnet_b2_last.pth
*_efficientnet_b2_epoch10.pth ...
selection.json
metrics.jsonl
options.json
data_snapshot.json
```

每次训练从 CNN/DINO 预训练权重开始，epoch 从 1、global step 从 0 开始。
不支持续训或整模型初始化；`--resume` 和 `--init_checkpoint` 已删除，传入将报未知参数错误。
脚本不再读取 `RESUME` 环境变量，旧命令设置该变量也会启动全新训练。
新 checkpoint 仅保存 `network/meta`，保留 v2 模型配置、阈值选择、epoch/global_step 和审计信息；
不保存任何训练恢复状态。best_primary、last、每 10 epoch 的 periodic 名称与保存时机不变。
新旧 v2 checkpoint 均可用于测试与推理，旧格式仍不支持。
早期 B2-OSCD v2 的 `[2,5,8,11] + raw[1:]` 仅按可证明等价关系解释为 `[5,8,11]`；
显式 `[2,8,11]` 的 0825 checkpoint 仍按原层路由复现。

相关内容：[训练](training.md) · [预测与评价](inference.md)

旧模型名称与内部路径的读取方式见[更名兼容说明](rename.md)。
