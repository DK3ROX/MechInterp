# Mechanistic Interpretability of Python Fine-Tuning in Pythia-160M

This project investigates how Python fine-tuning changes the internal representations of Pythia-160M.

## Main Idea

Sparse Autoencoders (SAEs) are trained independently on:

* Base Pythia-160M activations
* Python fine-tuned Pythia-160M activations

The resulting feature spaces are compared using:

* Reconstruction transfer
* Activation drift
* Token-level drift
* Sparse feature drift

## Dataset

CodeParrot Clean

## Layer Analyzed

Layer 6 Residual Stream

## Key Findings

* Cross-domain SAE reconstruction degrades by 3-4×
* Mean activation drift = 8.93
* Strong changes around:

  * self
  * imports
  * comments
  * docstrings
  * error handling
* Fine-tuning selectively repurposes sparse features rather than globally reorganizing the representation space.


## Data and Model Artifacts

The repository contains all source code used for:

* Activation extraction
* Sparse Autoencoder training
* Reconstruction transfer analysis
* Activation drift analysis
* Token-level drift analysis
* Sparse feature drift analysis

Large artifacts such as:

* Activation tensors
* SAE checkpoints
* Fine-tuned model checkpoints
* Intermediate analysis files

are stored separately on Google Drive due to GitHub size limitations.

### Google Drive

Project files can be accessed here:

LINK = "https://drive.google.com/drive/folders/1GRO3B1lEiKPTNwuqkQHdZzZ_iHO9oX2r?usp=drive_link"

Directory structure:

```text
MechInterp/
│
├── activations/
│   ├── base/
│   └── finetuned/
│
├── sae_models/
│   ├── base/
│   └── finetuned/
│
├── analysis/
│
└── finetuned_models/
    └── pythia_python_final/
```

The repository and Google Drive together provide the complete experimental pipeline required to reproduce all reported results.
