# Gene Expression Project
# Integrating Gene Regulatory Network Structure into Deep Learning
**Predicting gene expression across biological conditions using GRN-constrained GNNs**

**Author:** Hrithik Chandra  
**Supervisor:** Jan T. Kim

---

## Project summary
This project develops and evaluates hybrid machine learning models that integrate gene regulatory network (GRN) structure into Graph Neural Network (GNN) architectures to improve prediction of gene expression across different biological conditions. The aim is to combine biological interpretability (via GRNs) with the predictive power of deep learning, producing models that generalise better to unseen conditions.  
(This README is based on the Project Outline uploaded with the repository.) :contentReference[oaicite:1]{index=1}

---

## Aims & objectives
- Implement baseline ML models (Linear Regression, Random Forest, simple MLP/CNN) for expression prediction.  
- Design and implement a GRN-constrained GNN that uses known regulatory relationships as graph structure.  
- Compare baseline and GRN-GNN performance on held-out conditions using metrics: MSE / R² / Pearson correlation.  
- Interpret and visualise important regulatory relationships learned by the model.

Target deliverables: working code, trained models, evaluation plots, and a final report.

---

## Repo structure
```
.
├── simulators/              # GRN simulators (Transsys-style, GNW parser)
│   └── config_examples/     # Example GRN configs
├── data/                    # Generated and benchmark datasets
│   ├── synthetic_transsys/  # Training data from Transsys simulator
│   ├── synthetic_gnw_train/ # Extra GNW-generated training networks
│   └── dream4/              # Official DREAM4 datasets (benchmark only)
├── features/                # Feature extraction (GRN features, expression features)
├── models/                  # ML model implementations
├── experiments/             # Experiment scripts and early prototypes
├── notebooks/               # Jupyter notebooks for exploration and demos
├── results/                 # Output tables and figures
│   ├── tables/
│   └── figures/
├── report/                  # Report drafts
│   └── drafts/
├── docs/                    # Project documents (ProjectOutline.docx)
├── README.md
└── requirements.txt
```

---

## Setup (Windows)
1) Create a virtual environment (from the repo root):
	- `python -m venv .venv`
2) Activate it:
	- `\.venv\Scripts\Activate.ps1`
3) Install dependencies:
	- `pip install -r requirements.txt`

