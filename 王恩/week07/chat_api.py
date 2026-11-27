"""把 OpenAI 兼容聊天接口的回复投影到课件评测格式。"""

import json
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from data_sources import span
from http_client import open_url


@dataclass(frozen=True)
class ApiConfig:
    name: str
    base_url: str
    model: str
    key_env: str
    shots: int

    @property
    def endpoint(self):
        url = self.base_url.rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or (
            parts.username or parts.password or parts.query or parts.fragment
        ):
            raise ValueError("BASE_URL 应为 HTTP(S) 地址，且不含凭据、查询参数或片段")
        return url + "/chat/completions"


def bigrams(text):
    return set(zip(text, text[1:]))


class Prompts:
    def __init__(self, schema, train, max_shots):
        schema_lines = "\n".join(f"{kind}：{'、'.join(roles)}" for kind, roles in schema.items())
        self.system = (
            "从输入文本抽取一个标注事件。类型和角色必须使用下表中的完整名称。"
            "触发词和论元必须是原文的连续片段，原文没有的角色省略。"
            "同一角色有多个片段时分别输出。只输出 JSON，使用以下结构："
            '{"events":[{"type":"类型","trigger":"触发词","args":[{"role":"角色","text":"论元"}]}]}。'
            '无法识别时输出 {"events":[]}。\n'
            f"事件类型及允许的角色\n{schema_lines}"
        )
        self.examples = [(bigrams(row["text"]), row) for row in train] if max_shots else []

    def nearest(self, text, count):
        query = bigrams(text)

        def similarity(example):
            chars, _ = example
            return len(query & chars) / max(1, len(query | chars))

        return [row for _, row in sorted(self.examples, key=similarity, reverse=True)[:count]]

    @staticmethod
    def user(text):
        return {"role": "user", "content": json.dumps({"text": text}, ensure_ascii=False)}

    def messages(self, text, shots):
        examples = self.nearest(text, shots) if shots else []
        messages = [{"role": "system", "content": self.system}]
        for example in examples:
            event = example["event"]
            answer = {"events": [{
                "type": event["type"],
                "trigger": event["trigger"]["text"],
                "args": [{"role": a["role"], "text": a["text"]} for a in event["args"]],
            }]}
            messages.append(self.user(example["text"]))
            messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
        messages.append(self.user(text))
        return messages, [example["uid"] for example in examples]


def parse_reply(raw, text, schema):
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), flags=re.I | re.S)
    candidate = fenced[1] if fenced else raw
    try:
        event_list = json.loads(candidate)["events"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return [], {"status": "invalid_json_or_shape"}
    if not isinstance(event_list, list):
        return [], {"status": "invalid_json_or_shape"}

    stats = {"status": "ok", "wrapped_json": bool(fenced), "unknown_type": 0,
             "unknown_role": 0, "unmatched_spans": 0}

    def align(value):
        start = text.find(value)
        if start < 0:
            stats["unmatched_spans"] += 1
            return {"text": value, "start": -1, "end": -1}
        return span(value, start)

    events = []
    for item in event_list:
        if not isinstance(item, dict):
            return [], {"status": "invalid_json_or_shape"}
        kind, trigger, args = (item.get(key) for key in ("type", "trigger", "args"))
        if not isinstance(kind, str) or not kind or not isinstance(trigger, str) or not isinstance(args, list):
            return [], {"status": "invalid_json_or_shape"}
        stats["unknown_type"] += kind not in schema
        arguments = []
        for arg in args:
            if not isinstance(arg, dict) or not all(
                isinstance(arg.get(key), str) and arg[key] for key in ("role", "text")
            ):
                return [], {"status": "invalid_json_or_shape"}
            stats["unknown_role"] += arg["role"] not in schema.get(kind, ())
            arguments.append(dict(align(arg["text"]), role=arg["role"]))
        events.append({"type": kind, "trigger": align(trigger) if trigger else {}, "args": arguments})
    return events, stats


class ChatExtractor:
    def __init__(self, config, prompts, schema, timeout, max_tokens):
        self.config = config
        self.prompts = prompts
        self.schema = schema
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.url = config.endpoint
        self.key = None
        if config.key_env != "-":
            self.key = os.getenv(config.key_env)
            if not self.key:
                raise ValueError(f"缺少环境变量 {config.key_env}")
        self.fatal_http_error = None

    def predict(self, text):
        if self.fatal_http_error:
            return [], {"status": "api_error", "error": self.fatal_http_error}
        messages, examples = self.prompts.messages(text, self.config.shots)
        payload = {
            "model": self.config.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": 0,
        }
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        if self.key:
            request.add_unredirected_header("Authorization", f"Bearer {self.key}")
        try:
            with open_url(request, timeout=self.timeout) as response:
                result = json.load(response)
            choice = result["choices"][0]
            raw = choice["message"]["content"]
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                raise TypeError("content")
        except HTTPError as exc:
            error = f"HTTP {exc.code}"
            if exc.code in (400, 401, 403, 404):
                self.fatal_http_error = error
            exc.close()
            return [], {"status": "api_error", "error": error}
        except (URLError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            return [], {"status": "api_error", "error": type(exc).__name__}

        events, stats = parse_reply(raw, text, self.schema)
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        response_model, finish_reason = result.get("model"), choice.get("finish_reason")
        stats.update({
            "raw": raw,
            "examples": examples,
            "response_model": response_model if isinstance(response_model, str) else None,
            "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
        return events, stats
