"""
dataset_builder.py
==================
Builds a PyTorch instruction-code dataset from two sources:
  1. HuggingFace Hub (The Stack dataset - PyTorch files)
  2. Hand-crafted high-quality examples
Outputs a JSONL file to data/processed/pytorch_dataset.jsonl
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "pytorch_dataset.jsonl"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Hand-crafted PyTorch examples ─────────────────────────────────────────
HANDCRAFTED_EXAMPLES = [
    {
        "instruction": "Define a simple two-layer MLP in PyTorch with ReLU activation.",
        "output": """import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)
"""
    },
    {
        "instruction": "Write a PyTorch training loop with gradient clipping and cosine LR scheduler.",
        "output": """optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
"""
    },
    {
        "instruction": "Create a custom PyTorch Dataset for image classification.",
        "output": """import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
"""
    },
    {
        "instruction": "Implement a CNN with two convolutional layers for CIFAR-10 in PyTorch.",
        "output": """import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)
"""
    },
    {
        "instruction": "Write a PyTorch function to compute accuracy on a validation set.",
        "output": """def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
"""
    },
    {
        "instruction": "Use torch.compile() to speed up a PyTorch model.",
        "output": """import torch

model = MyModel().cuda()

# Wrap with torch.compile for automatic optimization
compiled_model = torch.compile(model, mode='reduce-overhead')

# Use exactly like the original model
output = compiled_model(inputs)
"""
    },
    {
        "instruction": "Implement early stopping in PyTorch based on validation loss.",
        "output": """class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
"""
    },
    {
        "instruction": "Save and load a PyTorch model checkpoint.",
        "output": """import torch

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
"""
    },
    {
        "instruction": "Create a PyTorch DataLoader with custom batch size and shuffling.",
        "output": """from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,   # faster GPU transfer
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
"""
    },
    {
        "instruction": "Implement batch normalization in a PyTorch neural network.",
        "output": """import torch.nn as nn

class BNNetwork(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)
"""
    },
]


# ── Helper: save to JSONL ──────────────────────────────────────────────────
def save_jsonl(examples: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


# ── Helper: basic code quality filter ─────────────────────────────────────
def is_good_pytorch_code(code: str) -> bool:
    if len(code) < 50 or len(code) > 2000:
        return False
    pytorch_keywords = ["torch", "nn.", "tensor", "cuda", "DataLoader",
                        "Module", "optimizer", "loss"]
    return any(kw in code for kw in pytorch_keywords)


# ── Source 1: HuggingFace The Stack (Python files using PyTorch) ───────────
def load_from_the_stack(max_samples: int = 500) -> list:
    print("Loading PyTorch samples from The Stack...")
    try:
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        examples = []
        for sample in tqdm(ds, total=max_samples, desc="Filtering PyTorch files"):
            if len(examples) >= max_samples:
                break
            code = sample.get("content", "")
            if is_good_pytorch_code(code):
                # Convert raw code into an instruction-output pair
                examples.append({
                    "instruction": "Complete the following PyTorch code snippet.",
                    "output": code[:1500],  # truncate very long files
                })
        print(f"Collected {len(examples)} samples from The Stack.")
        return examples
    except Exception as e:
        print(f"Could not load from The Stack: {e}")
        print("Continuing with handcrafted examples only.")
        return []


# ── Main ───────────────────────────────────────────────────────────────────
def build_dataset():
    print("=" * 50)
    print("Building PyTorch instruction dataset")
    print("=" * 50)

    all_examples = []

    # 1. Add handcrafted examples
    print(f"\nAdding {len(HANDCRAFTED_EXAMPLES)} handcrafted examples...")
    all_examples.extend(HANDCRAFTED_EXAMPLES)

    # 2. Try loading from The Stack
    stack_examples = load_from_the_stack(max_samples=500)
    all_examples.extend(stack_examples)

    # 3. Deduplicate by output
    seen = set()
    unique = []
    for ex in all_examples:
        key = ex["output"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"\nTotal unique examples: {len(unique)}")

    # 4. Save
    save_jsonl(unique, OUTPUT_FILE)
    print("\nDataset build complete!")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()