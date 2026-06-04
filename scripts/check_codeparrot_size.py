from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "codeparrot/codeparrot-clean",
    split="train",
    streaming=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-160m"
)

NUM_SAMPLES = 1000

total_tokens = 0

for i, sample in enumerate(dataset):

    code = sample["content"]

    ids = tokenizer(
        code,
        truncation=True,
        max_length=256
    )["input_ids"]

    total_tokens += len(ids)

    if i + 1 >= NUM_SAMPLES:
        break

print("Samples:", NUM_SAMPLES)
print("Total tokens:", total_tokens)
print("Average:", total_tokens / NUM_SAMPLES)