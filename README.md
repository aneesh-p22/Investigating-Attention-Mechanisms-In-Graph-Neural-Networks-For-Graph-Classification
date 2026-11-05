
# Investigating Attention Mechanisms in Graph Neural Networks for Graph Classification

**EPSRC Summer Research Internship, University of Leicester**  
**Researcher:** Aneesh  
**Supervisor:** Dr Furqan Aziz

This repository contains the code and experimental results for an investigation of attention mechanisms in graph classification. The main focus is on **GAT** and **GATv2**, with **GCN**, **GraphSAGE** and **GIN** included as reference models. Experiments use the **MUTAG**, **PROTEINS** and **NCI1** datasets. The full methodology and discussion are in the [research report](report/).

## Research questions

1. **RQ1:** How does the number of GAT attention heads affect classification accuracy at fixed total hidden width?
2. **RQ2:** How does GAT compare with GATv2 under a shared graph-classification setup?
3. **RQ3:** How does learned attention compare with a GAT trained using uniform neighbourhood weights?
4. **RQ4:** How non-uniform are the attention weights in fitted GAT models, and what happens when their scoring is replaced by uniform weighting without retraining?

## Results

The experiments include **135 cross-validation fits**: 75 reference-model fits, 45 additional head-count fits and 15 uniform-attention fits. RQ4 analyses the 15 selected reference GAT models without further training. The saved results and model states are available in [`results/`](results/).

### Reference models

Outer-test accuracy (%), mean ± sample standard deviation across five folds:

| Model | MUTAG | PROTEINS | NCI1 |
|:--|--:|--:|--:|
| GCN | 77.58 ± 8.20 | 74.57 ± 2.73 | 71.41 ± 0.67 |
| GraphSAGE | 79.73 ± 6.10 | 73.67 ± 3.17 | 73.38 ± 1.26 |
| GIN | 81.35 ± 7.43 | 72.59 ± 3.28 | 73.38 ± 1.79 |
| GAT | 79.23 ± 5.19 | 74.12 ± 2.37 | 72.31 ± 1.18 |
| GATv2 | 81.34 ± 6.98 | 73.22 ± 2.78 | 71.97 ± 1.38 |

### Attention comparisons

Mean paired accuracy differences (percentage points):

| Comparison | MUTAG | PROTEINS | NCI1 |
|:--|--:|--:|--:|
| GATv2 − GAT (RQ2) | +2.11 | −0.90 | −0.34 |
| Learned − retrained uniform GAT (RQ3) | −1.59 | +0.54 | +1.22 |
| Uniform intervention − fitted GAT (RQ4) | −1.10 | −5.75 | −20.32 |

For RQ1, eight heads gave the highest observed mean accuracy on MUTAG and PROTEINS, while two heads gave the highest on NCI1. Overall, the effect of attention design varied by dataset. In particular, training a separate uniform-attention model (RQ3) produced different results from changing attention scoring after fitting (RQ4). The report covers the fold-level results, attention measurements and limitations.

## Installation

The experiments were carried out on Windows 11 with Python 3.14.7 and an NVIDIA RTX 4070 Laptop GPU. The Python dependencies are listed in [`requirements.txt`](requirements.txt).

From PowerShell:

```powershell
git clone https://github.com/aneesh-p22/Investigating-Attention-Mechanisms-In-Graph-Neural-Networks-For-Graph-Classification---Infinity.git
cd Investigating-Attention-Mechanisms-In-Graph-Neural-Networks-For-Graph-Classification---Infinity
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The datasets are loaded through PyTorch Geometric's `TUDataset` and downloaded when first needed.

## Reproducing the results

### Saved results

To validate the saved records and reproduce the reference and RQ1–RQ3 summaries:

```powershell
python -m experiments.summarise
```

To print the saved RQ4 summary:

```powershell
python -c "import json; from experiments.analyse_fitted_gat import print_summary; print_summary(json.load(open('results/rq4_fitted_gat_attention.json', encoding='utf-8'))['datasets'])"
```

Both commands use stored measurements and do not retrain the models.

### Rerunning the experiments

The training scripts do not overwrite saved results. For an independent rerun, create a **separate clone** and remove the committed results from that copy only:

```powershell
git clone https://github.com/aneesh-p22/Investigating-Attention-Mechanisms-In-Graph-Neural-Networks-For-Graph-Classification---Infinity.git gnn-reproduction
cd gnn-reproduction
git switch -c reproduction
git rm -r results
git commit -m "Prepare independent reproduction"
```

Set up the Python environment in this clone as above, then run:

```powershell
python -m experiments.train_cv
python -m experiments.runners.train_heads
python -m experiments.runners.train_uniform
python -m experiments.summarise
python -m experiments.analyse_fitted_gat
```

The first three commands train the 135 models. The final command analyses the newly trained reference GAT states. Git must be configured because the result records include the source commit. Exact numerical results may vary across environments.

## Experimental setup

All models use two message-passing layers, hidden width 64, ReLU activations, global sum pooling and a linear graph classifier. Inputs consist of categorical node labels and graph connectivity, without additional node or edge attributes.

Evaluation uses **five stratified outer folds** with approximately **70% training, 10% validation and 20% testing** in each fold. Each model is trained for **500 epochs** using Adam, batch size **32**, learning rate **0.01** and weight decay **0.0005**. The model state with the lowest validation cross-entropy is selected for outer-test evaluation. The same partitions are used across models and variants.

RQ1 compares **1, 2, 4 and 8 heads** at total width 64. RQ2 uses standard GAT and general-form GATv2 (`share_weights=False`). RQ3 trains a separate GAT with uniform attention in both layers. RQ4 measures departure from uniform attention and performs a uniform-scoring intervention on fitted GAT states without retraining.

The common epoch budget was informed by earlier validation experiments on the same benchmark datasets, so the final evaluation is not entirely independent of development decisions. Other methodological limitations are discussed in the report.

## Repository structure

| Directory | Contents |
|:--|:--|
| [`src/`](src/) | Models, data loading, training and evaluation |
| [`experiments/`](experiments/) | Experiment scripts, analysis and result summaries |
| [`results/`](results/) | Result records and selected model states |
| [`report/`](report/) | Final report and references |
| [`log/`](log/) | Research and development records |
| [`archive/`](archive/) | Earlier implementations, not used by the final experiments |

## Licence

Released under the [MIT License](LICENSE). Third-party libraries and datasets retain their own licences.