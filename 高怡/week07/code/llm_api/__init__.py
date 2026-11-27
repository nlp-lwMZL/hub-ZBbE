"""通过本地 Ollama API 调用 Qwen3 完成事件抽取（免训练、无需 torch）。

复用 ``extract/llm_lora`` 的提示词、解析与对齐逻辑，仅把「模型推理」换成 HTTP 调用。
入口：``python -m extract.llm_api.run``。
"""
