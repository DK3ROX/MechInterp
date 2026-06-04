import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"

TARGET_LAYER = 6

NUM_SAMPLES = 1000
MAX_TOKENS = 256

ROOT = "../"

ACT_SAVE_PATH = os.path.join(
    ROOT,
    "activations/base/layer6_resid_post.pt"
)

TOKEN_MAP_PATH = os.path.join(
    ROOT,
    "analysis/token_map.pt"
)

os.makedirs(os.path.dirname(ACT_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TOKEN_MAP_PATH), exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
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

token_map = []

global_idx = 0

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

        input_ids = tokens["input_ids"][0]

        decoded_tokens = [
            tokenizer.decode([tok])
            for tok in input_ids
        ]

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

        for pos in range(len(input_ids)):

            token_map.append({

                "global_idx":
                    global_idx,

                "sample_idx":
                    sample_idx,

                "position":
                    pos,

                "token":
                    decoded_tokens[pos],

                "token_id":
                    int(input_ids[pos]),

                "all_tokens":
                    decoded_tokens,

                "full_text":
                    code

            })

            global_idx += 1

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

torch.save(
    token_map,
    TOKEN_MAP_PATH
)

print("\nSaved activations:")
print(ACT_SAVE_PATH)

print("\nSaved token map:")
print(TOKEN_MAP_PATH)

print("\nDone.")