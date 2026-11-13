"""SFT 的监督边界、因果性、权重迁移及生成停止条件测试。"""
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from sft_lm import (BASE, IGNORE_INDEX, SFTDataset, TransformerLM, check_splits,
                    collate_batch, extend_vocab, format_prompt, generate,
                    initialize_model, load_model, run_epoch)


class SFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rows = [dict(instruction="判断情感", input="很好", output="积极")]
        self.vocab = extend_vocab({"<UNK>": 0}, self.rows)

    def test_answer_shift_and_padding(self):
        dataset = SFTDataset(self.rows + [dict(instruction="判断情感", output="积极")], self.vocab, 64)
        x, y = dataset[0]
        n = 1 + len(format_prompt("判断情感", "很好"))
        self.assertTrue((y[:n-1] == IGNORE_INDEX).all())
        self.assertEqual(y[n-1].item(), self.vocab["积"])
        self.assertEqual(x[n].item(), self.vocab["积"])
        self.assertEqual(y[-1].item(), self.vocab["<EOS>"])
        _, padded = collate_batch([dataset[0], dataset[1]])
        self.assertTrue((padded[1, len(dataset[1][1]):] == IGNORE_INDEX).all())
        with self.assertRaises(ValueError):
            SFTDataset(self.rows, self.vocab, 4)

    def test_transfer_preserves_old_weights(self):
        base, checkpoint = load_model(BASE, "cpu")
        model, vocab, _ = initialize_model(BASE, self.rows)
        self.assertEqual(vocab["很"], checkpoint["vocab"]["很"])
        for name, weight in base.state_dict().items():
            self.assertTrue(torch.equal(weight, model.state_dict()[name][:len(weight)]))

    def test_causal_and_checkpoint_roundtrip(self):
        config = dict(vocab_size=len(self.vocab), d_model=16, nhead=2, num_layers=1,
                      dim_feedforward=32, dropout=0)
        model = TransformerLM(**config).eval()
        x = torch.tensor([[1, 2, 3, 4]])
        changed = x.clone()
        changed[0, 3] = 0
        self.assertTrue(torch.allclose(model(x)[:, :3], model(changed)[:, :3]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(dict(config=config, state_dict=model.state_dict()), path)
            restored, _ = load_model(path, "cpu")
            self.assertTrue(torch.equal(model(x), restored(x)))

    def test_loss_ignores_prompt_and_padding(self):
        class Uniform(nn.Module):
            def forward(inner, x):
                return torch.zeros(*x.shape, len(self.vocab))
        dataset = SFTDataset(self.rows, self.vocab, 64)
        metrics = run_epoch(Uniform(), [collate_batch([dataset[0]])], "cpu")
        self.assertEqual(metrics["tokens"], 3)
        self.assertAlmostEqual(metrics["ppl"], len(self.vocab), places=4)

    def test_eos_and_invalid_prompt(self):
        model = TransformerLM(len(self.vocab), d_model=8, nhead=2, num_layers=1,
                              dim_feedforward=16, dropout=0)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[self.vocab["<EOS>"]] = 20
        self.assertEqual(generate(model, self.vocab, "判断情感", "很好"), "")
        self.assertEqual(generate(model, self.vocab, "判断情感", "很好", greedy=False, top_p=0.01), "")
        with self.assertRaises(ValueError):
            generate(model, self.vocab, "未知问题")
        with self.assertRaises(ValueError):
            check_splits(dict(train=self.rows, val=self.rows))


if __name__ == "__main__":
    unittest.main()
