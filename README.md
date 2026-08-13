---
title: EpiResNet-v5 Multimodal AMR Engine
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🧬 EpiResNet v5: Production-ready Multimodal AMR Modeling Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.org/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/1333497044.svg)](https://doi.org/10.5281/zenodo.21924590)

## 📌 Name Origin & Meaning
**EpiResNet-v5** stands for **Epi**demiologic & **Epi**static **Res**idual **Net**work (Version 5):
* **Epi**: Represents epidemiological surveillance of antimicrobial resistance (AMR) and epistatic protein sequence variation.
* **ResNet**: Denotes residual graph attention architectures (GAT) combined with deep residual co-attention blocks bridging protein structures and chemical scaffolds.
* **v5**: The fifth architecture introducing **Interval-Censored Log-Likelihood MIC Modeling** and **Union-Find Transitive Leakage Control**.

---

## Key Features
* 🧬 **Protein Encoder**: ESM-2 (`facebook/esm2_t6_8M_UR50D`) with Parameter-Efficient Fine-Tuning (LoRA).
* 🧪 **Molecule Encoder**: Edge-aware 2-layer Graph Attention Network (GAT) processing RDKit molecular graphs.
* 🔄 **Bidirectional Co-Attention**: Cross-attention between protein residue representations and molecular atom tokens.
* 🎯 **Dual Task Objective**: Joint binary S/R classification and exact/interval-censored $\log_2(\text{MIC})$ regression.
* 🛡️ **Zero-Leakage Splitting**: Connected-component partitioning over isolates, scaffolds, and clusters using Union-Find.

---

## 📊 Dataset Format Requirement
Your input CSV should contain the following assay and molecular attributes:
- `protein_sequence`: Target protein sequence (FASTA format string).
- `smiles`: Chemical structure of the drug.
- `mic_value`: Numeric MIC laboratory value.
- `mic_unit`: Measurement unit (e.g., `mg/L`).
- `measurement_sign`: Operator (`=`, `<=`, `>`, `>=`).
- `organism` & `antibiotic`: Categorical identifiers for zero-leakage split grouping.

## 🏋️ Training & Validation
Run pipeline validation or start end-to-end model training:

```bash
# Fast dataset & split validation check (No training)
python epiresnet.py --csv data/sample_amr_data.csv --output-dir artifacts/epiresnet --validate-only

# Train model end-to-end
python epiresnet.py --csv data/sample_amr_data.csv --output-dir artifacts/epiresnet --epochs 10 --batch-size 8

```

---

## 🚀 Quickstart

### Local Setup
```bash
git clone [https://github.com/mymkgith/epiresnet.git](https://github.com/mymkgith/epiresnet.git)
cd epiresnet
pip install -r requirements.txt

```

---

## 🛠️ Testing & Data Generation Utilities

To make local development and testing fast, this repository includes two helper generators:

### 1. Instant UI Testing (`create_mock_artifacts.py`)
Generates dummy model weights (`best_model.pt`), run manifests, and evaluation metrics directly into `artifacts/epiresnet_v5/`.
* **Why use it:** Allows you to test or demonstrate the Streamlit application (`app.py`) immediately without waiting for model training or needing a GPU.
```bash
python create_mock_artifacts.py
streamlit run app.py

```

### 2. Synthetic Dataset Generator (`generate_sample_dataset.py`)
​Generates a mock dataset CSV (`amr_sample_dataset.csv`) populated with canonical SMILES strings, FASTA protein sequences, MIC values, and clinical breakpoints.
* **​Why use it:** Lets you verify data loading, graph construction, and end-to-end training commands before plugging in your own biological datasets.
```bash
python generate_sample_dataset.py


```

---
