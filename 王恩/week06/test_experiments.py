"""验证会影响对比结论的数据隔离、计分和最佳权重流程。"""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from data import load_data, normalize
from run_experiments import metrics


class ExperimentTests(unittest.TestCase):
    def write_splits(self, root, rows):
        for split, values in rows.items():
            with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label"])
                writer.writerows(values)

    def test_normalized_duplicates_removed_and_sampling_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_splits(root, {
                "train": [("Ａ B", "en"), ("abc", "en"), ("你好", "zh")],
                "dev": [(" a  b ", "en"), ("def", "en"), ("世界", "zh")],
                "test": [("DEF", "en"), ("ghi", "en"), ("朋友", "zh")]})
            splits, labels, meta = load_data(root)
            self.assertEqual(labels, ["en", "zh"])
            self.assertEqual(meta["removed_duplicates"], {"train": 0, "dev": 1, "test": 1})
            sets = [{normalize(r["text"]) for r in rows} for rows in splits.values()]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(meta["fingerprint"], load_data(root)[2]["fingerprint"])
            self.assertEqual(len(load_data(root, train_per_class=1)[0]["train"]), 2)

    def test_conflicting_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_splits(root, {"train": [("hello", "en"), ("你好", "zh")],
                                     "dev": [("HELLO", "zh")], "test": []})
            with self.assertRaisesRegex(ValueError, "标签冲突"):
                load_data(root)

    def test_missing_class_still_counts_in_macro_f1(self):
        report = metrics(["en", "zh"], ["en", "en"], ["en", "zh"])
        self.assertAlmostEqual(report["macro_f1"], 1 / 3)
        self.assertEqual(report["confusion_matrix"], [[1, 0], [1, 0]])
        with self.assertRaises(ValueError):
            metrics(["en"], [], ["en", "zh"])

    def test_char_cnn_padding_does_not_change_prediction(self):
        import torch
        from neural import CharCNN
        torch.manual_seed(42)
        model = CharCNN(10, 2).eval()
        with torch.no_grad():
            short = model(torch.tensor([[2, 3, 4]]))
            padded = model(torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]]))
        self.assertTrue(torch.allclose(short, padded, atol=1e-6))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "可选：需安装预训练依赖")
    def test_encoder_freeze_and_finetune_offline(self):
        """随机极小 BERT 只检查两条训练路径，不用其分数评价预训练效果。"""
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
        from neural import EncoderClassifier, train_neural

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32)
            original = BertModel(config)
            original.save_pretrained(root)
            backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3,
                                           "你": 4, "好": 5, "friend": 6, "朋友": 7}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
            tokenizer.save_pretrained(root)
            frozen = EncoderClassifier(str(root), 2, frozen=True).train()
            self.assertFalse(frozen.encoder.training)
            inputs = tokenizer(["hello world", "你 好"], padding=True, return_tensors="pt")
            frozen(**inputs).sum().backward()
            self.assertTrue(all(p.grad is None for p in frozen.encoder.parameters()))
            self.assertIsNotNone(frozen.head[-1].weight.grad)
            rows = [{"text": "hello world", "label": "en"}, {"text": "你 好", "label": "zh"}]
            args = SimpleNamespace(seed=42, threads=1, encoder=str(root), device="cpu", max_length=8,
                                   batch_size=2, lr=0.001, encoder_lr=2e-5, epochs=2, save_models=True)
            for method in ("encoder_frozen", "encoder_finetune"):
                predictions, report = train_neural(method, {s: rows for s in ("train", "dev", "test")},
                                                   ["en", "zh"], args, root)
                self.assertEqual(len(predictions), 2)
                best = max(report["history"], key=lambda item: item["dev_macro_f1"])
                self.assertEqual(report["best_epoch"], best["epoch"])
                saved = torch.load(root / f"{method}.pt", weights_only=False)
                changed = any(not torch.equal(saved["state_dict"]["encoder." + key], value)
                              for key, value in original.state_dict().items())
                self.assertEqual(changed, method == "encoder_finetune")


if __name__ == "__main__":
    unittest.main()
