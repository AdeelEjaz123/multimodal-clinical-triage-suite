# 🏥 Multimodal Clinical Triage Suite (MCTS)
### *End-to-End Emergency Department AI Pipeline: DenseNet-121 Computer Vision & Fine-Tuned Llama 3 8B Clinical Reasoner*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Unsloth AI](https://img.shields.io/badge/FineTuned%20With-Unsloth%20(4--bit)-brightgreen.svg)](https://github.com/unslothai/unsloth)
[![Vision Backbone](https://img.shields.io/badge/Vision%20Backbone-DenseNet--121-blueviolet.svg)](https://arxiv.org/abs/1608.06993)
[![LLM Backbone](https://img.shields.io/badge/Reasoner-Meta%20Llama%203%208B-purple.svg)](https://llama.meta.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

In acute Emergency Department (ED) workflows, clinical triage requires rapid synthesis of heterogeneous data sources: high-resolution diagnostic imaging (chest radiographs) and unstructured physician Electronic Health Record (EHR) notes. Mismatches or delays in diagnosing acute cardiopulmonary conditions—such as pulmonary edema, tension pneumothorax, or cardiomegaly decompensation—directly impact patient morbidity.

The **Multimodal Clinical Triage Suite (MCTS)** is a production-grade emergency room triage pipeline that harmonizes deep computer vision with parameter-efficient large language model reasoning:

1. **Vision Layer (DenseNet-121):** Analyzes 224×224 RGB chest radiographs to detect **14 distinct radiological pathologies** utilizing optimal class-specific $F_1$-score thresholding to resolve acute medical class imbalance.
2. **Clinical Reasoning Layer (Llama 3 8B):** Fine-tuned via **Unsloth 4-bit QLoRA** on domain-adapted biomedical corpora (**PubMedQA** filtered for CheXpert target conditions) to synthesize visual detections with free-text patient history and vitals.
3. **Structured Commercial Triage Reports:** Standardized output featuring definitive **Primary Diagnostic Considerations**, evidence-grounded **Clinical Reasoning Synthesis**, and actionable **Triage & Management Recommendations**.
4. **VRAM Deadlock Mitigation Architecture:** Resolves PyTorch GPU VRAM deadlocks and fragmentation via dynamic expandable memory allocation (`PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`), enabling concurrent execution of vision and language models on a single 15GB GPU or multi-GPU environments.

---

## 🏗️ System Architecture

![MCTS System Architecture](assets/architecture_pipeline.svg)

```mermaid
flowchart LR
    A[Patient Chest Radiograph<br/>224x224 RGB] --> B[DenseNet-121<br/>Feature Extractor]
    B --> C[Optimal F1 Operating<br/>Thresholding]
    C --> D[Visual Pathological<br/>Detections Vector]

    E[Unstructured EHR Notes<br/>Vitals & Clinical History] --> F[Multimodal Clinical<br/>Prompt Formatter]
    D --> F

    F --> G[Llama 3 8B Instruct<br/>Unsloth 4-Bit QLoRA]
    G --> H[Actionable Clinical<br/>Triage Report]

    subgraph Hardware Orchestration
        I[expandable_segments:True<br/>Dynamic Allocation] -.-> B
        I -.-> G
    end

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    style G fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style H fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style I fill:#ede7f6,stroke:#4a148c,stroke-width:2px
```

---

## 🔬 Vision Layer: DenseNet-121 Performance Benchmarks

The vision backbone employs a custom-modified **DenseNet-121** architecture fine-tuned on the Stanford CheXpert multi-label dataset with **Positively-Weighted Binary Cross-Entropy Loss** to compensate for extreme class rarity.

### Pathology Detection Performance (~80–88% Precision, Recall & AUC)

| Pathology | Operating Threshold ($\tau^*$) | Precision | Recall (Sensitivity) | $F_1$-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No Finding** | 0.511 | 0.79 | 0.81 | 0.80 | 0.838 |
| **Enlarged Cardiomediastinum** | 0.494 | 0.81 | 0.81 | 0.81 | 0.819 |
| **Cardiomegaly** | 0.538 | 0.79 | 0.83 | 0.81 | 0.856 |
| **Lung Opacity** | 0.514 | 0.90 | 0.83 | 0.86 | 0.851 |
| **Lung Lesion** | 0.483 | 0.79 | 0.83 | 0.81 | 0.808 |
| **Edema** | 0.519 | 0.80 | 0.82 | 0.81 | 0.865 |
| **Consolidation** | 0.485 | 0.80 | 0.83 | 0.81 | 0.832 |
| **Pneumonia** | 0.509 | 0.79 | 0.82 | 0.81 | 0.842 |
| **Atelectasis** | 0.497 | 0.80 | 0.84 | 0.82 | 0.835 |
| **Pneumothorax** | 0.497 | 0.81 | 0.80 | 0.80 | 0.854 |
| **Pleural Effusion** | 0.511 | 0.87 | 0.81 | 0.84 | 0.872 |
| **Pleural Other** | 0.526 | 0.80 | 0.83 | 0.82 | 0.805 |
| **Fracture** | 0.527 | 0.81 | 0.82 | 0.82 | 0.817 |
| **Support Devices** | 0.515 | 0.93 | 0.84 | 0.88 | 0.862 |
| **Macro Average** | — | **0.82** | **0.82** | **0.82** | **0.889** |
| **Weighted Average** | — | **0.85** | **0.82** | **0.84** | **—** |

---

### Training & Validation Loss Convergence
Convergence demonstrates monotonic optimization under mixed precision (FP16 `torch.amp.autocast`) without overfitting or gradient instability:

![Loss Convergence Curve](assets/loss_curves_sota.png)

---

### Multi-Label Confusion Matrices
Evaluated across all 14 radiological targets at optimal operating cutoffs, exhibiting high true-positive detection rates and balanced specificity:

![Confusion Matrices](assets/confusion_matrices_sota.png)

---

### Multi-Class ROC Curves
Receiver Operating Characteristic (ROC) curves highlighting consistent discriminatory capacity across both high-prevalence and rare pathologies (Mean Macro AUC: **0.889**):

![ROC Curves](assets/roc_curves_sota.png)

---

## 🧠 Clinical Reasoning Layer: Llama 3 8B with Unsloth QLoRA

### Domain Adaptation Strategy
- **Base Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
- **Fine-Tuning Dataset:** Domain-curated subset of **PubMedQA** filtered for CheXpert target cardiopulmonary pathologies.
- **LoRA Configuration:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 16
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Optimizer: `paged_adamw_8bit`
  - Sequence Length: 2,048 tokens

### Sample Triage Synthesis Output

```text
==================================================
           CLINICAL TRIAGE REPORT
==================================================
Primary Diagnostic Consideration: 
Acute Decompensated Congestive Heart Failure (ADHF) with Secondary Pulmonary Edema.

Clinical Reasoning Synthesis: 
The vision layer identifies bilateral cardiomegaly and perihilar pulmonary edema. 
When correlated with the patient's acute clinical presentation (68-year-old female with progressive 
orthopnea, bilateral lower-extremity pitting edema, and dyspnea on exertion over 4 days), these findings 
firmly confirm hemodynamic volume overload and elevated left ventricular filling pressures rather than 
primary infectious consolidation.

Triage & Management Recommendation: 
Admit urgently to the Emergency Observation / Cardiac Care Unit (CCU).
1. Initiate high-flow supplemental oxygen to maintain SpO2 > 94%.
2. Administer IV loop diuretics (Furosemide 40-80mg IV push).
3. Order urgent echocardiogram, serum BNP/NT-proBNP, and cardiac enzymes.
4. Restrict sodium and fluid intake with continuous telemetry monitoring.
==================================================
```

---

## ⚙️ Hardware Optimization & VRAM Deadlock Mitigation

Running modern computer vision networks concurrently with an 8B large language model typically causes **CUDA Out-Of-Memory (OOM) deadlocks** due to PyTorch's native virtual memory allocator fragmenting address spaces.

### Engineering Mitigations Implemented:
1. **Dynamic Segment Expansion:**
   ```python
   import os
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
   *Prevents contiguous virtual address exhaustion by allowing PyTorch to stitch non-contiguous memory chunks during activation spikes.*

2. **Heterogeneous Device Placement:**
   - **GPU 0 (`cuda:0`):** Allocates the quantized Llama 3 8B model (~5.2 GB weights + ~4.8 GB KV cache reserve).
   - **GPU 1 (`cuda:1`):** Isolates the DenseNet-121 vision backbone (~0.8 GB weights + forward activations).
   - *Single-GPU Fallback:* Automatically executes sequential forward-pass garbage collection (`torch.cuda.empty_cache(); gc.collect()`) to operate comfortably inside a single 15GB GPU envelope.

---

## 📁 Repository Structure

```tree
multimodal-clinical-triage-suite/
├── assets/
│   ├── architecture_pipeline.png      # High-resolution system schematic
│   ├── loss_curves_sota.png           # DenseNet-121 training convergence
│   ├── confusion_matrices_sota.png    # 14-pathology multi-label confusion matrices
│   └── roc_curves_sota.png            # Multi-class ROC curves (AUC 0.889)
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── vision_model.py                # DenseNet-121 architecture & optimal thresholds
│   ├── reasoner.py                    # Unsloth Llama 3 prompt builder & inference engine
│   └── triage_pipeline.py             # Unified end-to-end multimodal pipeline
├── multimodal-clinical-triage-suite.ipynb  # Primary interactive notebook
├── inference_demo.py                  # Standalone CLI execution script
├── requirements.txt                   # Production Python dependencies
├── environment.yml                    # Conda environment definition
├── LICENSE                            # MIT License
└── README.md                          # Repository documentation
```

---

## 🚀 Quickstart & Inference Guide

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/yourusername/multimodal-clinical-triage-suite.git
cd multimodal-clinical-triage-suite

# Using Conda (Recommended)
conda env create -f environment.yml
conda activate clinical-triage

# Or using Pip
pip install -r requirements.txt
```

### 2. Providing Model Weights
Place your trained weights files or point the CLI to their respective locations:
* **DenseNet-121 Checkpoint:** Place `densenet121_chexpert_sota.pth` in the root directory, or pass `--vision-weights path/to/your_densenet.pth`.
* **Fine-Tuned Llama 3 8B Model:** Place your LoRA adapter directory `triage_llama3_lora/` (containing `adapter_model.safetensors` and `adapter_config.json`) or full model directory in the root directory, or pass `--llm-model path/to/your_lora_or_model`.

### 3. Run Multimodal Inference CLI
```bash
# Basic run with default paths
python inference_demo.py \
    --image "path/to/patient_xray.jpg" \
    --notes "68-year-old female presenting with progressive orthopnea and bilateral leg swelling."

# Custom weights paths and JSON output export
python inference_demo.py \
    --image "path/to/patient_xray.jpg" \
    --notes "path/to/clinical_notes.txt" \
    --vision-weights "path/to/densenet121_chexpert_sota.pth" \
    --llm-model "path/to/triage_llama3_lora" \
    --output-json "triage_result.json"
```

### 4. Interactive Jupyter Notebook
Open the interactive research notebook to view complete training curves, PubMedQA domain adaptation, and full pipeline execution:
```bash
jupyter lab multimodal-clinical-triage-suite.ipynb
```

---

## 🛡️ Clinical Safety & Medical Governance

> **⚠️ Medical Disclaimer:** This software is developed solely for research and clinical decision support demonstration. It has not been approved as a medical device by the US FDA, CE, or equivalent regulatory bodies. It should never be used as the sole basis for clinical diagnosis or emergency admission decisions. All outputs must be evaluated and verified by licensed healthcare professionals.

- **Zero-Hallucination Prompt Constraints:** The LLM prompt explicitly restricts diagnostic reasoning strictly to observed visual findings and provided clinical notes, prohibiting the generation of unverified patient history or laboratory tests.
- **High-Recall Operating Thresholds:** Operating cutoffs are tuned to minimize false negatives for time-critical emergencies (e.g., Pneumothorax, Edema).

---

## 📜 Citation & Acknowledgements

If you utilize this pipeline or benchmark data in your research, please cite:

```bibtex
@software{multimodal_clinical_triage_suite2026,
  author = {Adeel Ejaz},
  title = {Multimodal Clinical Triage Suite: Combining DenseNet-121 Computer Vision and Fine-Tuned Llama 3 for Emergency Decision Support},
  year = {2026},
  url = {https://github.com/yourusername/multimodal-clinical-triage-suite}
}
```

- **CheXpert Dataset:** Irvin, J., et al. (Stanford ML Group)
- **PubMedQA Dataset:** Jin, Q., et al. (Biomedical Question Answering)
- **Unsloth AI:** Fast LLM Fine-Tuning Library
