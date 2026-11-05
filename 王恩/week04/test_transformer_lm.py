"""验证因果性、目标覆盖、PPL 定义，以及模型保存后的生成行为。"""

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from transformer_lm import (CharDataset, TransformerLM, generate_text,
                            load_model, make_loader, run_epoch, split_text)


class TransformerLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causal_attention_in_train_and_eval(self):
        torch.manual_seed(42)
        model = TransformerLM(10, seq_len=8, d_model=16, nhead=2,
                              num_layers=2, dim_feedforward=32, dropout=0)
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[1, 2, 3, 8, 9]])
        for training in (True, False):
            model.train(training)
            with torch.no_grad():
                torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3])

    def test_all_targets_once_including_short_tail(self):
        vocab = {c: i + 1 for i, c in enumerate("abcdefghij")}
        for length in (2, 4, 5, 8, 9, 10):
            text = "abcdefghij"[:length]
            dataset = CharDataset(text, vocab, seq_len=4)
            targets = []
            for i in range(len(dataset)):
                x, y = dataset[i]
                self.assertEqual(x.numel(), y.numel())
                targets.extend(y.tolist())
            self.assertEqual(targets, [vocab[c] for c in text[1:]])

    def test_uniform_predictions_ppl_and_padding(self):
        class UniformLM(nn.Module):
            def forward(self, x):
                return torch.zeros(*x.shape, 4)

        loader = make_loader("ababab?ab?", {"a": 1, "b": 2}, 4, 3)
        metrics = run_epoch(UniformLM(), loader, torch.device("cpu"))
        self.assertEqual(metrics["tokens"], 9)
        self.assertAlmostEqual(metrics["loss"], math.log(4), places=6)
        self.assertAlmostEqual(metrics["ppl"], 4, places=6)

    def test_split_before_vocab_and_keep_unknown(self):
        splits = split_text("abcdefghij", 0.2, 0.2)
        self.assertEqual(splits, {"train": "abcdef", "val": "gh", "test": "ij"})
        dataset = CharDataset(splits["val"], {"<UNK>": 0, "a": 1}, 4)
        self.assertEqual(dataset.ids.tolist(), [0, 0])
        with self.assertRaises(ValueError):
            split_text("abc", 0.05, 0.05)

    def test_checkpoint_roundtrip_and_generation(self):
        vocab = {"<UNK>": 0, "你": 1, "好": 2}
        config = dict(vocab_size=3, seq_len=4, d_model=8, nhead=2,
                      num_layers=1, dim_feedforward=16, dropout=0)
        model = TransformerLM(**config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": config, "state_dict": model.state_dict(), "vocab": vocab}, path)
            restored, checkpoint = load_model(path, torch.device("cpu"))
            x = torch.tensor([[1, 2]])
            torch.testing.assert_close(model(x), restored(x))
            text = generate_text(restored, checkpoint["vocab"], "你好你好你", 10, top_k=1)
            self.assertEqual(len(text), 15)
            self.assertTrue(text.startswith("你好你好你"))
            self.assertLessEqual(set(text), {"你", "好"})
            for prompt in ("", "未知"):
                with self.assertRaises(ValueError):
                    generate_text(restored, vocab, prompt)
            with self.assertRaises(ValueError):
                generate_text(restored, vocab, "你", temperature=0)


if __name__ == "__main__":
    unittest.main()
