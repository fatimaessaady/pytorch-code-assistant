"""
demo.py
=======
Gradio demo app for the fine-tuned PyTorch code assistant.
Run this inside Google Colab after training is complete.
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL   = "bigcode/starcoder2-3b"
ADAPTER_PATH = "/content/drive/MyDrive/pytorch-code-assistant/checkpoints/final_adapter"

INSTRUCTION_TEMPLATE = "### Instruction:"
RESPONSE_TEMPLATE    = "### Response:"

# ── Load model once at startup ─────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
print("Model ready!")

# ── Generation function ────────────────────────────────────────────────────
def generate_code(instruction: str, max_new_tokens: int, temperature: float) -> str:
    if not instruction.strip():
        return "Please enter an instruction."

    prompt = (
        f"{INSTRUCTION_TEMPLATE}\n"
        f"{instruction.strip()}\n\n"
        f"{RESPONSE_TEMPLATE}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated.strip()

# ── Example prompts ────────────────────────────────────────────────────────
EXAMPLES = [
    ["Write a PyTorch CNN with two convolutional layers for image classification.", 256, 0.2],
    ["Create a custom PyTorch Dataset class for loading images from a folder.", 256, 0.2],
    ["Write a PyTorch training loop with early stopping.", 256, 0.2],
    ["Implement a PyTorch learning rate warmup scheduler.", 256, 0.2],
    ["Write PyTorch code to save and load a model checkpoint.", 200, 0.2],
]

# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(title="PyTorch Code Assistant") as demo:
    gr.Markdown("""
    # 🔥 PyTorch Code Assistant
    **Fine-tuned StarCoder2-3B with LoRA** — specialized for generating PyTorch code snippets.
    Describe what you want in plain English and get working PyTorch code instantly.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            instruction_box = gr.Textbox(
                label="Your instruction",
                placeholder="e.g. Write a PyTorch CNN for CIFAR-10 classification",
                lines=3,
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=256,
                    step=32,
                    label="Max tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Temperature (lower = more focused)",
                )
            generate_btn = gr.Button("Generate Code", variant="primary")

        with gr.Column(scale=3):
            output_box = gr.Code(
                label="Generated PyTorch code",
                language="python",
                lines=20,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction_box, max_tokens, temperature],
        label="Example prompts — click to try",
    )

    generate_btn.click(
        fn=generate_code,
        inputs=[instruction_box, max_tokens, temperature],
        outputs=output_box,
    )

# ── Launch ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives a public URL in Colab