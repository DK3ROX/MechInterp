import torch

base = torch.load(
    "activations/base/layer6_resid_post.pt",
    map_location="cpu"
)

ft = torch.load(
    "activations/finetuned/layer6_resid_post_finetuned.pt",
    map_location="cpu"
)

print("BASE:", base.shape)
print("FT  :", ft.shape)

print("Same shape?", base.shape == ft.shape)

print("\nBase stats")
print("Mean:", base.mean().item())
print("Std :", base.std().item())

print("\nFT stats")
print("Mean:", ft.mean().item())
print("Std :", ft.std().item())