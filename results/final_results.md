# Final Results

## Project Goal

The objective of this project was to investigate how Python-specific fine-tuning changes the internal representations of Pythia-160M. Rather than measuring output quality alone, the goal was to determine whether fine-tuning reorganizes the model's latent representation space.

To study this, Sparse Autoencoders (SAEs) were trained independently on activations extracted from both a pretrained Pythia-160M model and a Python fine-tuned version of the same model.

---

## Experimental Setup

### Models

Base Model:

* Pythia-160M

Fine-Tuned Model:

* Python fine-tuned Pythia-160M

### Dataset

Dataset:

* CodeParrot Clean

Reason:

* Contains real-world GitHub code
* Includes functions, classes, imports, comments, docstrings, configuration files, and software-engineering patterns
* Better suited for Python interpretability than WikiText

### Activation Extraction

Layer analyzed:

* Layer 6 Residual Stream

Activation dimension:

* 768

Number of code samples:

* 1000

Total activations collected:

* 255,984

Activation tensor shape:

* [255984, 768]

Both models were evaluated on the exact same token sequences to ensure that any observed differences originated from model changes rather than dataset changes.

---

## Activation Statistics

### Base Model

Shape:

* [255984, 768]

Mean:

* -0.012337

Standard Deviation:

* 1.205263

Mean Token Norm:

* 20.3585

### Fine-Tuned Model

Shape:

* [255984, 768]

Mean:

* -0.013034

Standard Deviation:

* 1.179989

The activation distributions were extremely similar, indicating that later differences cannot be explained by simple scaling effects.

---

## Sparse Autoencoder Training

Library:

* SAE Lens

Architecture:

* StandardSAE

Input Dimension:

* 768

Dictionary Size:

* 6144

Training Configuration:

* Batch Size: 1024
* Learning Rate: 1e-4
* L1 Coefficient: 1e-3
* Epochs: 10

Separate SAEs were trained on the base and fine-tuned activations.

---

## Base SAE Results

Final Reconstruction Loss:

* 0.094386

Mean Active Features:

* 320.59

Median Active Features:

* 288

---

## Fine-Tuned SAE Results

Final Reconstruction Loss:

* 0.089690

Mean Active Features:

* 346.12

Median Active Features:

* 318

The fine-tuned SAE achieved slightly better reconstruction while using a somewhat larger number of active features.

---

# Reconstruction Transfer Analysis

To evaluate whether both models shared the same sparse feature basis, each SAE was tested on both activation distributions.

### Transfer Matrix

| SAE            | Base Activations | Fine-Tuned Activations |
| -------------- | ---------------: | ---------------------: |
| Base SAE       |         0.088149 |               0.257029 |
| Fine-Tuned SAE |         0.354776 |               0.086003 |

### Interpretation

Within-domain reconstruction remained very low:

* Base SAE on Base Activations: 0.088
* Fine-Tuned SAE on Fine-Tuned Activations: 0.086

However, cross-domain reconstruction degraded substantially:

* Base SAE on Fine-Tuned Activations: 0.257
* Fine-Tuned SAE on Base Activations: 0.355

Cross-domain reconstruction was approximately 3–4× worse than within-domain reconstruction.

This indicates that the sparse feature dictionaries learned by each SAE do not transfer effectively across models, suggesting significant representational change after fine-tuning.

---

# Activation Drift Analysis

For every token position, activation drift was computed as:

Drift = ||FineTunedActivation − BaseActivation||

### Results

Mean Drift:

* 8.9315

Median Drift:

* 7.9519

Maximum Drift:

* 100.0769

### Interpretation

Given an average activation norm of approximately 20.36, the mean drift corresponds to roughly 44% of the original activation magnitude.

This demonstrates substantial movement within the representation space and provides independent evidence that fine-tuning reorganized internal activations.

---

# Token-Level Drift Analysis

The highest-drift token positions were analyzed using the activation-to-token mapping.

### Frequently Occurring High-Drift Tokens

* #
* ##
* """
* self
* self
* import
* from
* Error
* dict
* append
* join
* pass
* decode

### Highest Average Drift Token Categories

Object-Oriented Programming:

* self
* self
* init
* args

Comments and Documentation:

* #
* """
* '''

Imports and Structure:

* import
* from

Error Handling:

* Error

Data Structures and Operations:

* dict
* append
* join
* data

### Interpretation

The largest representational shifts are concentrated around Python-specific concepts rather than arbitrary vocabulary.

This suggests that fine-tuning reorganized representations around practical software-engineering constructs.

---

# Sparse Feature Drift Analysis

Feature drift was computed directly in SAE space by comparing average feature activations between the two models.

### Top Drifting Features

| Feature | Drift |
| ------- | ----: |
| 647     | 15.93 |
| 684     | 10.93 |
| 1711    | 10.49 |
| 4415    | 10.21 |
| 3725    |  6.70 |
| 455     |  6.41 |
| 4074    |  5.91 |
| 3550    |  5.32 |

### Interpretation

Feature drift was highly concentrated.

Only a relatively small subset of the 6144 sparse features exhibited large changes, while the majority changed modestly.

This suggests targeted feature repurposing rather than global representational scrambling.

---

# Main Findings

### Finding 1

Python fine-tuning produces substantial internal representational change.

Evidence:

* Transfer matrix degradation
* Activation drift

### Finding 2

Layer 6 representations move significantly after fine-tuning.

Evidence:

* Mean Drift = 8.93
* Approximately 44% of average activation magnitude

### Finding 3

Representational changes are concentrated around Python-specific concepts.

Evidence:

* self
* imports
* comments
* docstrings
* error handling
* data structures

dominate token-level drift analysis.

### Finding 4

Only a small subset of sparse features changes dramatically.

Evidence:

* Highly concentrated feature drift distribution

### Finding 5

The observed changes are structured rather than random.

Evidence:

* Reconstruction transfer
* Activation drift
* Token drift
* Feature drift

all independently point toward the same conclusion.

---

# Final Conclusion

The results indicate that Python fine-tuning does not merely modify output behavior. Instead, it reorganizes internal Layer-6 representations.

The strongest representational changes are associated with software-engineering concepts including object-oriented programming, comments, documentation, imports, error handling, and common Python data operations.

Multiple independent analyses converge on the same conclusion:

Python fine-tuning selectively repurposes internal sparse features and produces meaningful representational reorganization within the model.
