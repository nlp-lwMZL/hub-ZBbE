# 12 语种分类对比实验（本次实际运行）

数据指纹：`4bce2dcaf1b84b9dcda943b0f2f607213d332c0abebe1abd5dd54618cc8dc72d`

| 方法 | 状态 | Test Accuracy | Test Macro-F1 | 训练/选择(s) | 测试推理(s) |
| --- | --- | --- | --- | --- | --- |
| rule | 完成 | 0.9167 | 0.8889 | 0.000 | 0.002 |
| encoder_frozen | 失败，详见 results.json | — | — | — | — |

CNN/Encoder 仅按 Dev Macro-F1 选 epoch；测试集只评测选中的权重。
训练/选择耗时包含验证和最佳权重复制，不含预训练模型加载；首次推理无预热。
未运行的方法不生成分数。多随机种子、数据量变化的结果请保存到不同输出目录。
