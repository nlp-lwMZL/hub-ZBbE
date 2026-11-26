"""
LLM 模型层 —— 模型注册 / 下载 / 三种加载模式 / logprob 打分

参照老师的 llm_lora/train_lora.py 与 classify_zero_shot.py，适配三臂 SFT 主对比：
  qwen3-0.6b              SFT 主对比：LoRA SFT vs full SFT（同尺寸、变量唯一）
  qwen3-1.7b              跨尺寸对照：LoRA SFT（与 0.6B 的 LoRA 只差参数量）
  qwen3-4b-instruct-2507  规模探针（zero/few-shot/QLoRA），2507 系纯非思考模型

三种模式对同一底座：
  full   所有参数可训练（T4 上仅 ≤1.7B 可行，fp16 主权重 + 静态 loss scaling
         + 8bit 优化器 + 梯度检查点；fp16 梯度不能用 GradScaler.unscale_）
  lora   底座 fp16 冻结 + 低秩旁路（1.7B 显存不紧张，不需要量化）
  qlora  底座 4bit NF4 冻结 + 低秩旁路（4B 及以上省显存的工具；bitsandbytes 需要 CUDA，
         本机 CPU 冒不了这个路径，会给出明确报错而不是神秘崩溃）

下载（双源，按环境自动选最快的那个）：
  Colab 机器在 Google 机房，直连 huggingface.co 的 CDN 通常比 ModelScope 快几倍；
  国内本机直连 HF 不通、ModelScope 直连可用——所以 auto = Colab 走 HF、本机走 ModelScope。
  HF 侧用 huggingface_hub.snapshot_download（断点续传 + 已有文件自动跳过），
  下载失败自动回退 ModelScope；想强制某个源：--source hf|ms 或环境变量 WEEK6_LLM_SOURCE。
  BERT 底座（iic/nlp_roberta_backbone_base_std）只有 ModelScope 有，不受影响。
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import torch

import paths

# ─────────────────────────────────────────────────────────────────────────────
# 模型注册表：短名 → 双源仓库 id。hybrid_thinking 决定渲染时是否需要
# enable_thinking=False（渲染统一在 dataset.render_messages 里处理，这里只做元信息）
# Qwen 官方在 HF 与 ModelScope 用同一个 "Qwen/..." 仓库名，但显式分开写，
# 以后遇到两边名字不一样的模型不用改这里的调用方。
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY = {
    # 0.6B 现在是正式实验模型（不是只做冒烟），显式登记 HF 优先；
    # auto 模式下即使本该走 MS，也先试 HF，失败仍自动回退 MS（见 ensure_local_llm）
    "qwen3-0.6b":             {"ms_id": "Qwen/Qwen3-0.6B",             "hf_id": "Qwen/Qwen3-0.6B",
                               "prefer_source": "hf", "hybrid_thinking": True},
    "qwen3-1.7b":             {"ms_id": "Qwen/Qwen3-1.7B",             "hf_id": "Qwen/Qwen3-1.7B",
                               "hybrid_thinking": True},
    "qwen3-4b-instruct-2507": {"ms_id": "Qwen/Qwen3-4B-Instruct-2507", "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
                               "hybrid_thinking": False},
}

# 下载白名单：前缀/通配。Qwen 仓库存的是分片 safetensors。
_KEEP_EXACT = {"config.json", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
               "added_tokens.json", "chat_template.jinja"}
_KEEP_SUFFIX = (".safetensors", ".safetensors.index.json")
_SKIP_EXACT = {"README.md", "LICENSE", "optimizer.pt", ".gitattributes"}


def _ms_files(ms_id: str) -> list:
    """ModelScope 仓库文件清单 [{Path, Size, Type}]。模型目录用 Type=='blob'"""
    url = f"https://www.modelscope.cn/api/v1/models/{ms_id}/repo/files?Revision=master"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    return [f for f in data["Data"]["Files"] if f.get("Type") == "blob"]


def _keep(name: str, has_safetensors: bool) -> bool:
    if name in _SKIP_EXACT or name.startswith("."):
        return False
    if name in _KEEP_EXACT or name.endswith(_KEEP_SUFFIX):
        return True
    # 仓库没有 safetensors 时才退回 .bin（Qwen3 各尺寸都有 safetensors，这条是保险）
    if not has_safetensors and name.endswith(".bin") and "pytorch_model" in name:
        return True
    return False


def _download_file(base_url: str, fname: str, dst: Path, log) -> None:
    """带进度 + 断点跳过的单文件下载（与 train_bert 的实现同构）"""
    if dst.exists() and dst.stat().st_size > 0:
        log(f"  已缓存 {fname}（{dst.stat().st_size:,} B），跳过")
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    req = urllib.request.Request(base_url + fname, headers={"User-Agent": "Mozilla/5.0"})
    t0, got, last = time.time(), 0, 0.0
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 5:
                pct = f"（{got / total * 100:.0f}%）" if total else ""
                log(f"  {fname}: {got / (1 << 20):.0f} MB{pct} "
                    f"@ {got / max(1e-9, time.time() - t0) / (1 << 20):.1f} MB/s")
                last = time.time()
    tmp.rename(dst)
    log(f"  完成 {fname}（{dst.stat().st_size:,} B，{time.time() - t0:.0f}s）")


def pick_source(source: str = "auto") -> str:
    """
    下载源选择：hf | ms | auto。
    auto：Colab 上走 HuggingFace（Google 机房直连 HF CDN，通常快几倍），
          国内本机直连 HF 不通、走 ModelScope；
          本机想用 HF 镜像也行：export HF_ENDPOINT=https://hf-mirror.com
          并设 WEEK6_LLM_SOURCE=hf（或训练/评测时传 --source hf）。
    例外：注册表里带 `prefer_source: hf` 的模型（qwen3-0.6b），auto 下也先走 HF，
    失败仍回退 ModelScope——这个例外由 ensure_local_llm 处理，不在本函数里写死。
    """
    s = (source or "auto").strip().lower()
    if s in ("hf", "huggingface"):
        return "hf"
    if s in ("ms", "modelscope"):
        return "ms"
    s = os.environ.get("WEEK6_LLM_SOURCE", "auto").strip().lower()
    if s in ("hf", "ms"):
        return s
    return "hf" if paths.is_colab() else "ms"


def resolve_download_source(info: dict, source: str = "auto") -> str:
    """
    在 pick_source 的「环境默认」之上叠加模型自己的偏好。
    qwen3-0.6b 标了 prefer_source=hf：auto 下先试 HF，失败仍由调用方回退 MS；
    显式 --source 或 WEEK6_LLM_SOURCE 永远优先（本机 HF 不通时用 --source ms 跳过等待）。
    """
    src = pick_source(source)
    env_source = os.environ.get("WEEK6_LLM_SOURCE", "").strip().lower()
    auto_asked = (source or "auto").strip().lower() == "auto" and env_source not in ("hf", "ms")
    if auto_asked and info.get("prefer_source") == "hf" and info.get("hf_id"):
        return "hf"
    return src


def _cache_complete(cache: Path) -> bool:
    """
    缓存完整性粗检：config + 至少一个权重 + tokenizer 文件都在才跳过下载。
    两个下载器的单文件落盘都是「先写 .tmp 再 rename」，目录里不会留半截文件，
    所以这一检查不会把中断的残缺目录误判为完整。
    """
    if not (cache / "config.json").is_file():
        return False
    if not any(cache.glob("*.safetensors")) and not any(cache.glob("pytorch_model*.bin")):
        return False
    return any(cache.glob("tokenizer*")) or (cache / "vocab.json").is_file()


def _hf_download(hf_id: str, cache: Path, log) -> Path:
    """
    HuggingFace snapshot 下载。huggingface_hub 是 transformers 的依赖，无需额外安装。
    local_dir 模式：文件直接落在 cache 下（不进 ~/.cache 深层 blob 目录），
    已存在的文件按元数据跳过——与 ModelScope 路径共用同一个缓存目录，
    从任一源下到一半换源重跑都能续。
    """
    from huggingface_hub import snapshot_download
    log(f"从 HuggingFace 下载 {hf_id} → {cache}（Colab 直连 HF 通常比 ModelScope 快几倍）")
    snapshot_download(
        repo_id=hf_id, local_dir=str(cache),
        allow_patterns=[*sorted(_KEEP_EXACT), "*.safetensors", "*.safetensors.index.json"],
        ignore_patterns=["README.md", "LICENSE", "optimizer.pt", ".gitattributes"],
    )
    return cache


def _find_complete_cache(name: str) -> Path | None:
    """
    在 MODEL_ROOT 与历史位置里找一个完整的缓存目录。
    历史位置 = 旧版默认（代码目录下的 pretrain_models / Drive 的 pretrain_models）：
    换了默认位置不等于要把已经下好的 3~8GB 权重重下一遍。
    """
    for root in [Path(paths.MODEL_ROOT), *paths.legacy_model_roots()]:
        d = Path(root) / name
        if d.is_dir() and _cache_complete(d):
            return d
    return None


def ensure_local_llm(name_or_path: str, logger=None, source: str = "auto") -> Path:
    """
    本地目录直接用；注册表短名/仓库 id 则下载到 MODEL_ROOT 下
    （默认 = 代码目录同级的 models，Colab 上即 /content/models，不进 Drive）。
    source: auto | hf | ms（见 pick_source）；HF 失败自动回退 ModelScope。
    """
    log = logger.info if logger else print
    p = Path(name_or_path)
    if p.exists():
        log(f"使用本地模型目录: {p}")
        return p
    info = dict(REGISTRY.get(name_or_path) or {"ms_id": name_or_path, "hf_id": name_or_path})
    name = name_or_path.split("/")[-1]
    hit = _find_complete_cache(name)
    if hit is not None:
        note = "" if hit.parent == Path(paths.MODEL_ROOT) else \
            f"（旧缓存位置；新默认是 {Path(paths.MODEL_ROOT) / name}）"
        log(f"使用已缓存模型: {hit}（文件齐全，跳过下载）{note}")
        return hit
    cache = Path(paths.MODEL_ROOT) / name
    cache.mkdir(parents=True, exist_ok=True)

    src = resolve_download_source(info, source)
    if info.get("prefer_source") == "hf" and src == "hf" and (source or "auto").strip().lower() == "auto":
        log(f"{name}: auto 原会走 ModelScope，注册表标记 HF 优先 → 先试 HuggingFace，失败自动回退")
    if src == "hf" and info.get("hf_id"):
        try:
            return _hf_download(info["hf_id"], cache, log)
        except Exception as e:                                     # noqa: BLE001
            log(f"!! HuggingFace 下载失败（{type(e).__name__}: {e}），回退 ModelScope")

    base = f"https://www.modelscope.cn/api/v1/models/{info['ms_id']}/repo?Revision=master&FilePath="
    files = _ms_files(info["ms_id"])
    has_sf = any(f["Path"].endswith(".safetensors") for f in files)
    todo = [f["Path"] for f in files if _keep(f["Path"], has_sf)]
    if not todo:
        raise FileNotFoundError(f"{info['ms_id']} 下没有匹配的权重文件，请检查仓库名")
    total_gb = sum(f["Size"] for f in files if f["Path"] in todo) / 1e9
    log(f"从 ModelScope 下载 {info['ms_id']}（{len(todo)} 个文件，约 {total_gb:.1f} GB）→ {cache}")
    for fname in todo:
        _download_file(base, fname, cache / fname, log)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 加载：tokenizer / 三种模式
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(name_or_path: str, logger=None, source: str = "auto"):
    from transformers import AutoTokenizer
    path = ensure_local_llm(name_or_path, logger, source=source)
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, path


def pick_amp(device: str, logger=None):
    """
    T4（Turing，SM 7.5）没有原生 bf16 → fp16 + GradScaler；
    Ampere 及以后用 bf16（不需要 scaler）。CPU 冒烟不开 autocast。
    返回 (autocast_dtype, use_scaler)

    坑（2026 年真的踩了）：torch>=2.4 给 is_bf16_supported() 加了
    including_emulation 参数且**默认 True**——T4 上它返回 True，因为 bf16 可以
    用软件模拟。于是这里错判成 bf16、把 scaler 关掉，实际却在 SM 7.5 上跑
    bf16 仿真内核，既慢又让 fp16 主权重拿着没有 loss scaling 的梯度。
    只认原生 bf16：including_emulation=False。
    """
    log = logger.info if logger else (lambda *a, **k: None)
    if device == "cpu":
        return torch.float32, False
    if torch.cuda.is_available():
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:            # torch<2.4 没有该参数，旧语义就是「原生」
            native_bf16 = torch.cuda.is_bf16_supported()
        if native_bf16:
            return torch.bfloat16, False
        # 走到这里说明是 T4/P100 一类：fp16 必须配 GradScaler
        emu = False
        try:
            emu = torch.cuda.is_bf16_supported()      # 默认含仿真，T4 上会是 True
        except Exception:                             # noqa: BLE001
            pass
        if emu:
            log("!! 检测到只有「仿真 bf16」（T4 等 SM<8.0），已按 fp16+GradScaler 走")
    return torch.float16, True


def describe_device(device: str) -> str:
    """一行说清在哪张卡上跑：型号 + 计算能力 + 总显存（判断能不能全量微调的唯一依据）"""
    if device == "cpu":
        return "CPU（只能冒烟，正式训练需要 CUDA）"
    try:
        p = torch.cuda.get_device_properties(0)
        total = p.total_memory / 2 ** 30
        return (f"{p.name} | SM {p.major}.{p.minor} | 总显存 {total:.1f} GiB "
                f"| 原生bf16={'是' if p.major >= 8 else '否'}")
    except Exception:                                 # noqa: BLE001
        return str(device)


def load_base_model(name_or_path: str, mode: str, device: str, logger=None,
                    source: str = "auto", master_dtype: str = "auto"):
    """
    返回 (model, tok, meta)。mode ∈ full|lora|qlora：
      full      **主权重 fp32**（默认）——fp16 主权重在 lr=2e-5 下更新量 ~1e-5，
                与 fp16 的分辨率同量级，大量更新被舍入成 0：能跑、loss 不动
      lora      底座 fp16 冻结（PEFT 会把 adapter 自动转 fp32，更新精度不受影响）
      qlora     4bit NF4 + double quant（bitsandbytes 仅支持 CUDA，CPU 上明确报错）

    master_dtype: auto|fp32|fp16，只对 full 生效。auto=fp32；显存实在不够时
    可以显式传 fp16 换显存（代价见上），preflight_memory 会算给你看差多少。
    """
    from transformers import AutoModelForCausalLM
    tok, path = load_tokenizer(name_or_path, logger, source=source)
    log = logger.info if logger else print

    if mode == "qlora":
        if device == "cpu":
            raise RuntimeError("QLoRA 需要 CUDA（bitsandbytes 4bit 内核不支持 CPU）。"
                               "本机冒烟请用 --mode lora 或 full；QLoRA 留给 Colab T4。")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError("缺少 bitsandbytes：pip install bitsandbytes>=0.43.0") from e
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(str(path), quantization_config=bnb,
                                                     device_map={"": 0})
        log("底座以 4bit NF4 加载（QLoRA）")
        # numel() 对 bnb 的 Params4bit 也是准的：拿到的是**逻辑**参数量，
        # 显存按 0.55 B/param 折算（estimate_param_footprint 里处理）
        return model, tok, {"path": str(path),
                            "n_params": sum(p.numel() for p in model.parameters())}

    if device != "cuda":
        dtype = torch.float32
    elif mode == "full" and master_dtype != "fp16":
        dtype = torch.float32
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"底座常精度加载：{n_params / 1e6:.0f}M 参数，dtype={str(dtype).split('.')[-1]}"
        + ("（全量微调的主权重必须是 fp32，见 load_base_model 注释）"
           if mode == "full" and dtype == torch.float32 else "")
        + ("（fp16 主权重：省显存但更新精度只有 ~1e-5，loss 可能压着不动）"
           if mode == "full" and dtype == torch.float16 else ""))
    return model, tok, {"path": str(path), "n_params": n_params}


def _disable_torchao_dispatch(log):
    """
    Colab 镜像自带 torchao（如 0.10.0），而新版 peft 要求 >0.16.0。
    peft 的 is_torchao_available() 在「装了但版本不够」时**抛 ImportError 而不是
    返回 False**，而 inject_adapter 对每个 target 都会走一遍 torchao 分支 →
    整个 LoRA 注入直接崩，报错还长得像缺包（其实根本用不到 torchao）。

    我们不使用 torchao 量化 LoRA，所以把它的探测函数换成恒 False 即可。
    必须打在 peft.tuners.lora.torchao 这个**模块自己的名字**上——它 import 时
    就 `from peft.import_utils import is_torchao_available` 绑定了引用，
    改 import_utils 里的同名函数对已绑定的引用无效。
    """
    patched = []
    try:
        import peft.tuners.lora.torchao as _ta
    except Exception:                                  # noqa: BLE001  老版本 peft 没这个模块
        return
    orig = getattr(_ta, "is_torchao_available", None)
    if orig is None:
        return
    try:
        if not orig():                                 # 干净返回 False：本来就不会进这个分支
            return
    except Exception as e:                             # noqa: BLE001  版本不兼容的那种抛法
        log(f"!! torchao 版本与 peft 不匹配（{e}）")
        log("   → 已跳过 torchao 量化分支（我们不用它）；彻底清掉请执行："
            "pip uninstall -y torchao")
        _ta.is_torchao_available = lambda: False
        patched.append("peft.tuners.lora.torchao")
    try:
        import peft.import_utils as _iu
        if patched and getattr(_iu, "is_torchao_available", None):
            _iu.is_torchao_available = lambda: False
            patched.append("peft.import_utils")
    except Exception:                                  # noqa: BLE001
        pass


def wrap_peft(model, r: int, alpha: int, dropout: float, targets: str, logger=None):
    """LoRA 注入。targets: qkvo（老师口径）| all（+MLP 三个投影，LLM SFT 常用加强版）"""
    from peft import LoraConfig, TaskType, get_peft_model
    _disable_torchao_dispatch(lambda m: (logger.info if logger else print)(m))
    mods = {"qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "all":  ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]}[targets]
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha,
                     lora_dropout=dropout, target_modules=mods)
    model = get_peft_model(model, cfg)
    # 一定要打印：target_modules 写错名字不会报错，只会「训了等于没训」（老师的教学点）
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f"LoRA: r={r} alpha={alpha} targets={targets} → 可训练 {n_tr/1e6:.2f}M / {n_all/1e6:.0f}M（{n_tr/n_all:.2%}）"
    (logger.info if logger else print)(msg)
    return model


_OPT_RESOLVED = {}


def resolve_optimizer(opt_name: str, logger=None) -> str:
    """
    开跑前先确认优化器**真的**能用，返回实际会用上的名字。
    单独拆出来是为了让显存预算在 model.to(cuda) 之前就能算准——
    等 build_optimizer 里静默退回 fp32 AdamW，账已经算错了。
    结果缓存：train_llm 会先问一次（算预算），build_optimizer 再问一次，
    不缓存就会把同一段警告打两遍。
    """
    if opt_name in _OPT_RESOLVED:
        return _OPT_RESOLVED[opt_name]
    log = logger.info if hasattr(logger, "info") else (logger or (lambda *a, **k: None))
    resolved = opt_name
    if opt_name == "adamw8bit":
        try:
            import bitsandbytes as bnb
            _ = bnb.optim.AdamW8bit      # 装了但 .so 加载失败（轮子和 CUDA 不匹配）也在这里炸
        except Exception as e:           # noqa: BLE001
            log(f"!! bitsandbytes 不可用（{type(e).__name__}: {e}），退回 fp32 AdamW")
            log("   → 全量微调的优化器状态会从 ~3.5 GiB 涨到 ~12.8 GiB（1.7B）。"
                "修法：pip install -q -U bitsandbytes")
            resolved = "adamw"
    _OPT_RESOLVED[opt_name] = resolved
    return resolved


def build_optimizer(model, mode: str, lr: float, opt_name: str, logger=None):
    """
    返回 (optimizer, 实际用上的优化器名)。
    full 默认 adamw8bit（T4 上的关键一步：fp32 的两个状态是 1.7B 的 12.8 GiB，
    8bit 之后约 3.5 GiB）；lora/qlora 参数量小，普通 AdamW 就够。
    """
    log = logger.info if hasattr(logger, "info") else (logger or (lambda *a, **k: None))
    opt_name = resolve_optimizer(opt_name, logger)
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr)
        log(f"优化器 AdamW8bit lr={lr}（8bit 状态省 ~2/3 优化器显存）")
        return opt, "adamw8bit"
    opt = torch.optim.AdamW(params, lr=lr)
    log(f"优化器 AdamW lr={lr}")
    return opt, "adamw"


# ─────────────────────────────────────────────────────────────────────────────
# 显存预算：参数侧是**可以精确算出来**的，算完再决定要不要开跑
#
# T4 上全量微调 1.7B 的账（fp16 权重 / fp32 主权重两种口径）：
#   权重 3.2 + 梯度 3.2 + AdamW(fp32 m,v) 12.8 = 19.2 GiB  > 14.6 GiB  → 无解
#   权重 6.4 + 梯度 6.4 + AdamW(fp32 m,v) 12.8 = 25.6 GiB  > 14.6 GiB  → 更无解
# 所以「跑三分钟才 OOM」不是运气问题，是算术问题——先算，别等它炸。
# ─────────────────────────────────────────────────────────────────────────────

_QLORA_BYTES = 0.55          # NF4 + double quant ≈ 0.5~0.55 bytes/param
_FP16, _FP32 = 2.0, 4.0      # bytes/param，估显存时按这几个数乘


def estimate_param_footprint(n_params: int, n_trainable: int, mode: str,
                             opt_name: str, master_fp32: bool) -> dict:
    """参数侧占用（GiB，不含激活）。激活取决于 batch×长度×层数，留给余量。"""
    if mode == "qlora":
        w_b = _QLORA_BYTES                      # 4bit NF4 冻结底座
    elif mode == "full":
        w_b = _FP32 if master_fp32 else _FP16
    else:
        w_b = _FP16                             # lora：底座 fp16 冻结，只有 adapter 要算状态
    g_b = (_FP32 if master_fp32 else _FP16) if mode == "full" else 0.0
    o_b = 2.2 if opt_name == "adamw8bit" else _FP32 * 2   # 8bit: 两个 int8 + fp32 block scale
    gib = lambda n, b: n * b / 2 ** 30                       # noqa: E731
    return {"weights": gib(n_params, w_b),
            "grads": gib(n_trainable, g_b) if mode == "full" else 0.0,
            "optimizer": gib(n_trainable, o_b) if n_trainable else 0.0}


def preflight_memory(meta: dict, mode: str, opt_name: str, device: str, logger=None,
                     batch_size: int = 0, max_length: int = 0,
                     master_dtype: str = "auto"):
    """
    开跑前把参数侧的账算清楚：
      · 参数侧就已经超过显存 → 直接拒绝，并给出**可执行的**三条改法
      · 参数侧吃掉 >65% 显存 → 警告，建议降 batch / 改 8bit
    返回 estimate 字典（调用方可以随手记进日志）。
    """
    log = logger.info if hasattr(logger, "info") else (logger or (lambda *a, **k: None))
    if device != "cuda" or not meta.get("n_params"):
        return {}
    n_params = int(meta["n_params"])
    model = meta.get("model")
    if mode != "full" and model is None:
        return {}                    # 数不出可训练参数就别硬判——宁可放过，不可误杀 LoRA
    n_trainable = n_params
    if mode != "full":
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    master_fp32 = mode == "full" and master_dtype != "fp16"
    est = estimate_param_footprint(n_params, n_trainable, mode, opt_name, master_fp32)
    fixed = sum(est.values())
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    left = total - fixed

    log(f"显存预算：权重 {est['weights']:.2f} + 梯度 {est['grads']:.2f} + "
        f"优化器 {est['optimizer']:.2f} = {fixed:.2f} GiB / 卡 {total:.1f} GiB"
        f"（含参数的激活余量 {left:.2f} GiB）")

    if left <= 0.8:
        name = str(meta.get("name") or "").strip() or "该模型"
        raise SystemExit(
            f"\n✗ {mode} 模式在 {total:.1f} GiB 的卡上装不下：\n"
            f"    权重 {est['weights']:.2f} + 梯度 {est['grads']:.2f} + "
            f"优化器 {est['optimizer']:.2f} = {fixed:.2f} GiB，"
            f"还没算激活就已经超了 {-left:.2f} GiB。\n"
            f"  这不是「调小 batch 能救」的问题，是参数侧的算术问题。按推荐顺序改：\n"
            f"    1) 装 bitsandbytes 后用 8bit 优化器：\n"
            f"         pip install -q -U bitsandbytes\n"
            f"         ... --mode {mode} --optimizer adamw8bit --batch_size 1 --grad_accum 32\n"
            f"       → 优化器状态从 {est['optimizer']:.1f} 降到 ~3.5 GiB\n"
            f"    2) 全量微调换小底座（0.6B 才是 T4 上做全量的正常规模）：\n"
            f"         ... --model qwen3-0.6b --mode full\n"
            f"    3) 1.7B 想省显存又要训全参 → 用 --master_dtype fp16"
            f"（权重+梯度砍半，代价是更新精度只有 ~1e-5）；\n"
            f"       或者在 T4 上就老老实实用 --mode lora / qlora，这是算力决定的，不是配置问题。\n"
            f"  算账方法见 model_llm.estimate_param_footprint。")
    if left < 0.35 * total:
        log(f"⚠️  激活余量只剩 {left:.2f} GiB（<35% 卡容量）。"
            f"当前 batch={batch_size} len={max_length}：建议 --batch_size 1 "
            f"并调大 --grad_accum，必要时设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    return est


# ─────────────────────────────────────────────────────────────────────────────
# logprob 打分：把生成式分类变成「三个标签 token 各算一次 logit」的判别式打分
# 比逐 token 生成快几十倍，且**结构上杜绝**「模型输出了别的内容」的解析失败。
# 生成式路径（--generate）保留在 predict_llm.py 里用于统计解析失败率。
# ─────────────────────────────────────────────────────────────────────────────

def label_first_tokens(tok, label_names: list, logger=None):
    """
    每个标签取**首 token id**。高/中/低 是单字标签，正常都是单 token；
    出现多 token 时如实告警并退回首 token（不悄悄吞掉这个信息）。
    """
    log = logger.info if logger else print
    ids = []
    for name in label_names:
        enc = tok.encode(name, add_special_tokens=False)
        if len(enc) == 0:
            raise ValueError(
                f"标签「{name}」被 tokenizer 切成 0 个 token"
                f"（tokenizer={type(tok).__name__}, vocab_size={getattr(tok, 'vocab_size', '?')}）。"
                "几乎总是 tokenizer 加载错了：全量 SFT 的 ckpt 目录只有权重、没有 tokenizer 文件。"
                "评测要从底座模型目录加载 tokenizer（predict_llm.py 已修），"
                "新训练会用 tok.save_pretrained 把 tokenizer 一并存进 ckpt。")
        if len(enc) != 1:
            log(f"!! 标签「{name}」被切成 {len(enc)} 个 token，打分只用首 token")
        ids.append(enc[0])
    return torch.tensor(ids, dtype=torch.long)


def _keep_last_token_hook(module, args):
    """
    lm_head 的前向钩子：把输入 [B, L, H] 砍成 [B, 1, H]，只算最后一个位置的 vocab logits。
    为什么用钩子而不是 model(..., logits_to_keep=1)：各版本 transformers 对
    logits_to_keep 的支持不一致（有的版本根本没有这个参数，有的切片位置不同），
    钩子直接作用在 lm_head 上，与版本 / PEFT 包装都无关。
    """
    if args and hasattr(args[0], "dim") and args[0].dim() == 3 and args[0].shape[1] > 1:
        return (args[0][:, -1:, :],) + tuple(args[1:])
    return None


@torch.no_grad()
def logprob_classify(model, tok, prompts: list, label_ids, device: str,
                     batch_size: int = 16, amp_dtype=None, logger=None):
    """
    prompts: 已渲染好的 prompt 文本列表（dataset.render_messages 的输出）。
    返回 (preds[list[int]], probs[list[list[float]]])。

    实现要点 1：只取「最后一个真实 token」位置的 logits——那个位置正是
                「下一个 token 该是标签」的分布。

    实现要点 2（显存，T4 上最容易炸的一处）：判别式打分只需要 1 个位置的 logits，
                但 model(...) 默认会给**每个 token** 都算 vocab 维 logits：
                  [B, L, 151936] fp16，B=16、L≈1270 时一次就是 5.77 GiB，
                  直接把 15GB 的 T4 顶爆（实测 OOM 报错正是这一笔分配）。
                这里在 lm_head 前挂钩子，只喂最后一位：
                  5.77 GiB → 4.6 MB，batch 和结果都不变。
                配套用**左 padding**：每个样本的末位才都是真实 token
                （RoPE 只看相对位置，整体平移不改变打分，等价性成立）。

    实现要点 3：万一还是 OOM（更长的 prompt / 更大的模型），自动 batch 减半重试，
                不让你丢掉整轮评测。
    """
    model.eval()
    preds, probs = [], []
    use_amp = amp_dtype is not None and amp_dtype != torch.float32
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    log = logger.info if logger else (lambda *a, **k: None)

    # lm_head 钩子：拿不到输出头就退回全序列 logits（会慢/费显存，但不会错）
    head, handle = None, None
    try:
        head = model.get_output_embeddings()
    except Exception:                                    # noqa: BLE001
        head = None
    if head is not None and hasattr(head, "register_forward_pre_hook"):
        handle = head.register_forward_pre_hook(_keep_last_token_hook)
    else:
        log("!! 拿不到输出头，退回全序列 logits（显存占用高，必要时 --batch_size 4）")

    old_side = tok.padding_side        # 用完还原，避免污染调用方的生成式解码
    tok.padding_side = "left"
    try:
        i = 0
        while i < len(prompts):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            try:
                with torch.autocast(device_type=device_type,
                                    dtype=amp_dtype or torch.float32, enabled=use_amp):
                    rows = model(**enc).logits[:, -1, :]     # [B, V]（钩子已裁到末位）
                lab = rows[:, label_ids.to(device)].float()  # [B, C]
                p = torch.softmax(lab, dim=-1)
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                new_bs = max(1, batch_size // 2)
                log(f"!! 显存不足（batch={batch_size}），自动降到 {new_bs} 重试同一批")
                batch_size = new_bs
                continue                    # 不推进 i，同一起点重跑
            preds.extend(p.argmax(dim=-1).cpu().tolist())
            probs.extend(p.cpu().tolist())
            i += len(batch)
    finally:
        tok.padding_side = old_side
        if handle is not None:
            handle.remove()
    return preds, probs
