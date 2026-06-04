import torch

acts = torch.load(
    "activations/base/layer6_resid_post.pt",
    map_location="cpu"
)

norms = acts.norm(dim=1)

print("Mean token norm :", norms.mean().item())
print("Std token norm  :", norms.std().item())
print("Min token norm  :", norms.min().item())
print("Max token norm  :", norms.max().item())