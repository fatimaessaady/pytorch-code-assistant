"""
evaluate.py
===========
Evaluates the fine-tuned StarCoder2-3B LoRA adapter on PyTorch code generation.
Run this inside Google Colab after training is complete.

Metrics:
  - Exact match: does the output match the reference exactly?
  - BLEU score: how similar is the output to the reference?
  - Syntax check: does the generated code parse without errors?
  - Manual prompts: qualitative review of 5 test prompts
"""

import ast
import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Paths ──────────────────────────────────────────────────────────────────
ADAPTER_PATH = "/content/drive/MyDrive/pytorch-code-assistant/checkpoints/final_adapter"
BASE_MODEL   = "bigcode/starcoder2-3b"
RESULTS_PATH = "/content/drive/MyDrive/pytorch-code-assistant/eval_results.json"

INSTRUCTION_TEMPLATE = "### Instruction:"
RESPONSE_TEMPLATE    = "### Response:"

# ── Test prompts — never seen during training ──────────────────────────────
TEST_PROMPTS = [
    {
        "instruction": "Write a PyTorch LSTM model for sequence classification.",
        "reference": """import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))
"""
    },
    {
        "instruction": "Implement a PyTorch learning rate warmup scheduler.",
        "reference": """from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)
"""
    },
    {
        "instruction": "Write a PyTorch function to calculate top-5 accuracy.",
        "reference": """def top5_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:5].reshape(-1).float().sum(0)
        return correct_k.mul_(100.0 / batch_size).item()
"""
    },
    {
        "instruction": "Create a PyTorch model that uses dropout for regularization.",
        "reference": """import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)
"""
    },
    {
        "instruction": "Write PyTorch code to move a model and data to GPU if available.",
        "reference": """import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move data to device in training loop
inputs = inputs.to(device)
labels = labels.to(device)
"""
    },
]


# ── Load model ─────────────────────────────────────────────────────────────
def load_model():
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!\n")
    return model, tokenizer


# ── Generate code ──────────────────────────────────────────────────────────
def generate(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    prompt = (
        f"{INSTRUCTION_TEMPLATE}\n"
        f"{instruction.strip()}\n\n"
        f"{RESPONSE_TEMPLATE}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only the generated part after the prompt
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated.strip()


# ── Metrics ────────────────────────────────────────────────────────────────
def check_syntax(code: str) -> bool:
    """Returns True if the code is valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def bleu_score(reference: str, hypothesis: str) -> float:
    """Simple word-level BLEU-1 score."""
    ref_tokens = set(reference.split())
    hyp_tokens = hypothesis.split()
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matches / len(hyp_tokens)


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference.strip() == hypothesis.strip()


# ── Main evaluation ────────────────────────────────────────────────────────
def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Enable T4 in Colab.")

    model, tokenizer = load_model()

    results = []
    syntax_passes = 0
    bleu_scores = []
    exact_matches = 0

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for i, test in enumerate(TEST_PROMPTS):
        print(f"\n[Test {i+1}/5]")
        print(f"Instruction: {test['instruction']}")
        print("-" * 40)

        generated = generate(model, tokenizer, test["instruction"])

        syntax_ok = check_syntax(generated)
        bleu = bleu_score(test["reference"], generated)
        em = exact_match(test["reference"], generated)

        if syntax_ok:
            syntax_passes += 1
        bleu_scores.append(bleu)
        if em:
            exact_matches += 1

        print(f"Generated code:\n{generated}")
        print(f"\nSyntax OK : {syntax_ok}")
        print(f"BLEU-1    : {bleu:.3f}")
        print(f"Exact match: {em}")

        results.append({
            "instruction": test["instruction"],
            "reference": test["reference"],
            "generated": generated,
            "syntax_ok": syntax_ok,
            "bleu": bleu,
            "exact_match": em,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Syntax pass rate : {syntax_passes}/5 ({syntax_passes*20}%)")
    print(f"Average BLEU-1   : {sum(bleu_scores)/len(bleu_scores):.3f}")
    print(f"Exact matches    : {exact_matches}/5")

    # Save results
    summary = {
        "model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "syntax_pass_rate": syntax_passes / 5,
        "average_bleu": sum(bleu_scores) / len(bleu_scores),
        "exact_matches": exact_matches,
        "results": results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()