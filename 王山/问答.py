# infer.py
import torch
import math
from language_model_fixed import LM   # 从你原来的脚本导入模型类

@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, device, seq_len=64, max_gen_len=120, top_k=8, temp=0.8):
    model.eval()
    pad_idx = char2idx["<PAD>"]
    ids = [char2idx.get(c, pad_idx) for c in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_gen_len):
        if x.size(1) > seq_len:
            x = x[:, -seq_len:]
        logits = model(x)
        next_logit = logits[:, -1, :] / temp

        # top-k 截断
        values, _ = torch.topk(next_logit, top_k)
        cutoff = values[:, -1].unsqueeze(-1)
        next_logit = torch.where(next_logit < cutoff, torch.full_like(next_logit, -1e9), next_logit)

        prob = torch.softmax(next_logit, dim=-1)
        next_id = torch.multinomial(prob, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].cpu().tolist()
    text = "".join([idx2char[i] for i in out_ids if i != pad_idx])
    return text


def main():
    ckpt_path = "sft_best.pt"   # 你的SFT模型文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    saved_args = ckpt["args"]
    pad_idx = char2idx["<PAD>"]

    # 还原模型
    model = LM(
        vocab_size=len(char2idx),
        embed_dim=saved_args["embed_dim"],
        hidden_dim=saved_args["hidden_dim"],
        num_layers=saved_args["num_layers"],
        model_type=saved_args["model"],
        dropout=saved_args["dropout"],
        pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("✅ 模型加载完成，可以提问，输入 q 退出")

    while True:
        q = input("\n请输入问题：")
        if q.strip().lower() == "q":
            break
        # !!!!!!!!! 重点：必须和你SFT训练的input格式一模一样 !!!!!!!!!
        prompt = f"问：{q}\n答："
        ans = generate(model, prompt, char2idx, idx2char, device, seq_len=saved_args["seq_len"], max_gen_len=120)

        print(ans)


if __name__ == "__main__":
    main()
