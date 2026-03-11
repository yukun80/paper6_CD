# SAR Prompt Flood

面向 `datasets/GF3_Henan` 的 GF3 单极化双时相变化检测/洪水提取工具。

## 当前任务定义

- 输入：`Pre_Zhengzhou_ascending_clip.tif` 与 `Post_Zhengzhou_ascending_clip.tif`
- 输出：全图二值变化/洪水 mask，以及变化置信图
- 数据条件：无标签
- 主流程：预处理 -> 变化先验 -> prompt 生成 -> 规则优化 -> SAM/启发式分割 -> 全图回拼

## 为什么必须先做预处理

这两张 GF3 图虽然尺寸、CRS、分辨率一致，但 `transform` 不完全一致，不能直接逐像素做变化检测。当前预处理会：

- 以 `Pre` 为参考网格重采样 `Post`
- 清洗 `nodata=-3.4028235e38`
- 构造 `diff / log_ratio_like / change_score`
- 按 `512x512`、`128 overlap` 切 tile

## 运行

只做预处理：

```bash
python datasets/script/prepare_gf3_henan_pair.py
```

跑完整 pipeline：

```bash
python -m sar_prompt_flood.run_gf3_pipeline \
  --config-file sar_prompt_flood/config/gf3_henan.json
```

若要启用真实 SAM，请在配置中填写：

- `segmenter.sam_checkpoint`

若未提供 checkpoint，pipeline 会自动只使用启发式分割结果。
