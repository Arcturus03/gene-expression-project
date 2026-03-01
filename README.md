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
- /src # python scripts and modules; first trials
- /datasets # small example datasets and required datasets storage
- /docs # project documents and notes (ProjectOutline.docx)
- /experiments # saved model outputs, figures, logs
- /literature # research papers related to the project 
- README.md
- requirements.txt

---

## Setup (Windows)
1) Create a virtual environment (from the repo root):
	- `python -m venv .venv`
2) Activate it:
	- `\.venv\Scripts\Activate.ps1`
3) Install dependencies:
	- `pip install -r requirements.txt`

