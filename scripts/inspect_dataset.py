from datasets import load_dataset

dataset = load_dataset(
    "codeparrot/codeparrot-clean",
    split="train",
    streaming=True
)

first = next(iter(dataset))

print(first.keys())
print(first)