import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "finetuned_models/pythia_python_final"

TARGET_LAYER = 6

NUM_SAMPLES = 1000
MAX_TOKENS = 256

ACT_SAVE_PATH = "/home/dk3rox/DK3ROX/Programs/Python/MechInterp/activations/finetuned/layer6_resid_post_finetuned.pt"


# ============================================================
# LOAD MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading fine-tuned model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading CodeParrot...")

dataset = load_dataset(
    "codeparrot/codeparrot-clean",
    split="train",
    streaming=True
)

# ============================================================
# HOOK
# ============================================================

captured = []

def hook_fn(module, inputs, outputs):

    if isinstance(outputs, tuple):
        hidden = outputs[0]
    else:
        hidden = outputs

    captured.append(
        hidden.detach().cpu()
    )

hook_handle = (
    model.gpt_neox.layers[TARGET_LAYER]
    .register_forward_hook(hook_fn)
)

# ============================================================
# EXTRACTION
# ============================================================

all_activations = []

print("Extracting activations...")

for sample_idx, sample in enumerate(tqdm(dataset)):

    code = sample["content"]

    try:

        tokens = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS
        )

        tokens_gpu = {
            k: v.to(device)
            for k, v in tokens.items()
        }

        captured.clear()

        with torch.no_grad():
            model(**tokens_gpu)

        acts = captured[0]

        if acts.ndim == 3:
            acts = acts.squeeze(0)

        all_activations.append(acts)

        if sample_idx + 1 >= NUM_SAMPLES:
            break

    except Exception as e:

        print("Skipping sample:", e)

# ============================================================
# SAVE
# ============================================================

hook_handle.remove()

print("\nConcatenating activations...")

all_activations = torch.cat(
    all_activations,
    dim=0
)

print("Activation shape:")
print(all_activations.shape)

torch.save(
    all_activations,
    ACT_SAVE_PATH
)

print("\nSaved activations:")
print(ACT_SAVE_PATH)

print("\nDone.")