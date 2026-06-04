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
