"""L3 · LLM zero-shot / few-shot：不训练参数，靠 prompt 分类。运行: python l3_llm_prompt.py"""
import csv
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DATA_DIR, report


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N_SHOT = 15  # 0 为 zero-shot，15 为每类一条示例
MAX_SAMPLES = None

LABELS = ["100","101","102","103","104","106","107","108","109","110","112","113","114","115","116"]
NAMES = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技",
         "军事", "旅游", "国际", "证券", "农业", "电竞"]

LABEL2NAME = {}
NAME2LABEL = {}
for i in range(len(LABELS)):
    LABEL2NAME[LABELS[i]] = NAMES[i]
    NAME2LABEL[NAMES[i]] = LABELS[i]


def build_prompt(text, examples=None):
    """返回 messages 列表（[{"role": "user", "content": ...}]）"""
    task = ("把下面这条新闻标题分到以下 15 个类别之一："
            + "、".join(NAMES) + "。只回答类别名。")
    content = task
    if examples:
        content += "参考示例："
        for label, example_text in examples:
            content += "\n" + example_text + " -> " + label
    content += "\n标题：" + text
    return [{"role": "user", "content": content}]


def pick_examples(train_rows, n):
    """按类别顺序各抽一条示例，最多返回 n 条。"""
    # 固定示例抽样，不影响其他随机操作。
    rng = random.Random(42)
    by_label = {}
    for label, text in train_rows:
        by_label.setdefault(label, []).append(text)
    examples = []
    for label in LABELS:
        texts = by_label.get(label, [])
        if texts:
            examples.append((LABEL2NAME[label], rng.choice(texts)))
    return examples[:n]


def generate(tokenizer, model, messages):
    """chat 模板拼 prompt -> 模型生成 -> 只保留新生成的部分（去掉 prompt 回显）"""
    # 按模型的对话格式拼消息；tokenize=False 返回文本，add_generation_prompt 添加回答起始标记。
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # return_tensors="pt" 返回张量；to 将输入移动到模型所在的设备。
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # 只需生成中文类名，最多 4 个 token，避免长篇续写拖慢对比。
    # generate 返回输入和续写的 token 编号；max_new_tokens 仅限制新生成的长度。
    # do_sample=False 每步选择最高分 token，decode 再将编号转回文字。
    out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    # 去掉输入部分，只解析新生成的回答。
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_answer(raw_output):
    """从回答中提取类别编号，无法识别时返回 None。"""
    output = raw_output.strip()
    if output in NAME2LABEL:
        return NAME2LABEL[output]
    for name in NAMES:
        if name in output:
            return NAME2LABEL[name]
    return None


def main():
    test_rows = list(csv.reader(open(DATA_DIR / "test.tsv", encoding="utf-8"), delimiter="\t"))[1:][:MAX_SAMPLES]
    train_rows = list(csv.reader(open(DATA_DIR / "train.tsv", encoding="utf-8"), delimiter="\t"))[1:]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 使用库内置 eager 注意力，兼容 MPS，并正确处理自回归模型的因果遮罩。
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    model.eval()

    examples = pick_examples(train_rows, N_SHOT) if N_SHOT > 0 else None

    y_true, y_pred, failed = [], [], 0
    for label, text in test_rows:
        messages = build_prompt(text, examples)
        answer = parse_answer(generate(tokenizer, model, messages))
        if answer is None:
            failed += 1
            answer = "109"  # 解析失败时回退到科技类
        y_true.append(label)
        y_pred.append(answer)

    print(f"\nmodel={MODEL_NAME}  n_shot={N_SHOT}  samples={len(test_rows)}")
    print(f"解析失败: {failed}/{len(test_rows)}")
    return report(y_true, y_pred)


if __name__ == "__main__":
    from common import prepare_quick_data
    prepare_quick_data()
    main()
