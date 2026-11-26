"""基于 langchain-ollama 的 Ollama 客户端。

为什么用 langchain-ollama 而不是 OpenAI 兼容层
---------------------------------------------
Ollama 的 ``/v1`` 兼容层有两个硬缺陷（实测于 ollama 0.13.5 + qwen3:4b）：

1. 不认 ``think=False``（``extra_body``、``chat_template_kwargs``、``/no_think`` 均无效），
   Qwen3 会把 token 全花在思考上，``content`` 返回空串；
2. 不认 ``num_ctx``，上下文被固定为模型默认 4096，长公告会被静默截断。

``langchain-ollama`` 的 ``ChatOllama`` 走原生 ``/api/chat``，``reasoning=False`` 可真正
关闭思考，``num_ctx`` / ``num_predict`` 也能生效。

解码策略
--------
``temperature`` 固定为 0（贪心），保证与 llm_lora 的 ``do_sample=False`` 等价、结果可复现；
传非 0 会直接报错。
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class OllamaError(RuntimeError):
    """调用本地 Ollama 服务失败。"""


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://127.0.0.1:11434",
        num_predict: int = 256,
        num_ctx: int = 4096,
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 300.0,
        think: bool = False,
        force_json: bool = True,
        retries: int = 3,
    ) -> None:
        if temperature != 0.0:
            raise ValueError(f"temperature 固定为 0（贪心解码），收到 {temperature}")
        self.model = model
        self.host = host.rstrip("/")
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.temperature = 0.0
        self.seed = seed
        self.think = think
        self.retries = max(0, retries)
        self._llm = ChatOllama(
            model=model,
            base_url=self.host,
            temperature=0.0,
            seed=seed,
            num_ctx=num_ctx,
            num_predict=num_predict,
            format="json" if force_json else None,
            reasoning=think,          # False = 关闭 Qwen3 思考
            validate_model_on_init=False,
            client_kwargs={"timeout": timeout},
        )

    # ---------------------------------------------------------------- API
    def list_models(self) -> list[str]:
        from ollama import Client

        try:
            resp = Client(host=self.host).list()
        except Exception as e:  # noqa: BLE001
            raise OllamaError(
                f"无法连接 {self.host}：{e}. 请确认已运行 `ollama serve`，"
                f"且已 `ollama pull {self.model}`"
            ) from e
        models = getattr(resp, "models", None) or resp.get("models") or []
        out = []
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
            if name:
                out.append(name)
        return out

    def chat(self, system: str, user: str,
             demos: list[tuple[str, str]] | None = None) -> tuple[str, dict]:
        """返回 (生成文本, 元信息)。

        ``demos`` 为少样本示例 [(用户文本, 助手标准答案)]，按多轮对话拼在正式提问之前，
        与 llm_lora 的 SFT 格式一致（system → 若干 human/ai → 最终 human）。
        """
        messages = [SystemMessage(content=system)]
        for demo_user, demo_answer in demos or []:
            messages.append(HumanMessage(content=demo_user))
            messages.append(AIMessage(content=demo_answer))
        messages.append(HumanMessage(content=user))

        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                msg = self._llm.invoke(messages)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.retries:
                    # Ollama 的 runner 偶尔会崩（502 / 连接被重置），退避后重试
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise OllamaError(f"{self.model} 调用失败（重试 {self.retries} 次）：{e}") from e
        else:  # pragma: no cover
            raise OllamaError(f"{self.model} 调用失败：{last_err}")
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        usage = msg.usage_metadata or {}
        meta = {
            "eval_count": int(usage.get("output_tokens") or 0),
            "prompt_eval_count": int(usage.get("input_tokens") or 0),
            "wall_sec": round(time.time() - t0, 3),
            "done_reason": "",
        }
        return content, meta
