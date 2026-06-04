# Mechanistic Interpretability of Python Fine-Tuning in Pythia-160M

## Overview

This project investigates how Python-specific fine-tuning changes the internal representations of Pythia-160M.

Using Sparse Autoencoders (SAEs), we compare Layer-6 activations from:

* Base Pythia-160M
* Python Fine-Tuned Pythia-160M

The goal is to determine whether fine-tuning merely changes outputs or reorganizes the model's internal feature space.

---

## Experimental Pipeline

```text
CodeParrot Clean
       │
       ▼
Pythia-160M (Base) ─────► Base SAE
       │
       ▼
Layer 6 Activations

Python Fine-Tuned Pythia-160M ─────► Fine-Tuned SAE
       │
       ▼
Layer 6 Activations

Comparison:
- Reconstruction Transfer
- Activation Drift
- Token Drift
- Sparse Feature Drift
```

---

## Dataset

Dataset:

* CodeParrot Clean

Samples:

* 1000 code files

Layer:

* Layer 6 Residual Stream

Activations:

* 255,984 tokens
* Hidden dimension: 768

---

## Key Results

### SAE Reconstruction

| Model          | Reconstruction Loss |
| -------------- | ------------------: |
| Base SAE       |              0.0944 |
| Fine-Tuned SAE |              0.0897 |

### Reconstruction Transfer

| SAE      | Base Acts | FT Acts |
| -------- | --------: | ------: |
| Base SAE |    0.0881 |  0.2570 |
| FT SAE   |    0.3548 |  0.0860 |

Cross-domain reconstruction degrades by approximately 3–4×, indicating significant representational change after fine-tuning.

### Activation Drift

Mean Drift:

* 8.93

Median Drift:

* 7.95

Max Drift:

* 100.08

### High-Drift Concepts

* self
* imports
* comments
* docstrings
* error handling
* Python data operations

---

## Main Findings

* Python fine-tuning significantly reorganizes Layer-6 representations.
* Internal activation patterns move substantially after fine-tuning.
* The largest changes occur around software-engineering concepts.
* Sparse feature drift is concentrated in a relatively small subset of SAE features.
* Multiple independent analyses support the same conclusion.

---

## Repository Structure

```text
MechInterp/

├── scripts/
├── results/
├── README.md
└── requirements.txt
```

---

## External Artifacts

Large files are stored separately due to GitHub size limits.

Google Drive:

LINK = "https://drive.google.com/drive/folders/1GRO3B1lEiKPTNwuqkQHdZzZ_iHO9oX2r?usp=drive_link"

Contains:

* Activation tensors
* SAE checkpoints
* Fine-tuned model
* Intermediate analysis files

---

## Reproducing

```bash
pip install -r requirements.txt
```

1. Extract activations
2. Train SAEs
3. Run transfer analysis
4. Run drift analysis

---

## Detailed Results

See:

```text
results/final_results.md
```
