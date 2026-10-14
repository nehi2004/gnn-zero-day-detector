# Explainable GNN for Zero-Day Intrusion Detection in NetFlow Traffic

A Graph Neural Network (GraphSAGE) that classifies network flows as **Benign** or **Attack**, is tested on attack types it never saw during training (a zero-day simulation), and explains each decision with **GNNExplainer**. Comes with an interactive Streamlit dashboard.

## Problem
- Signature-based intrusion detection misses **zero-day attacks** (attacks with no known pattern).
- Deep-learning detectors are usually **black boxes**: they give a label but no reason.

## Approach
1. **Data:** NF-UNSW-NB15-v2 (NetFlow format, ~1.99M flows, 43 columns, 9 attack categories + benign).
2. **Graph construction:** the dataset has no IP addresses, so each flow becomes a **node** and edges connect each flow to its 8 nearest neighbours in feature space (KNN similarity graph).
3. **Model:** 2-layer GraphSAGE + linear classifier (binary: Benign / Attack).
4. **Zero-day simulation:** the **Worms** and **Backdoor** categories are removed from training and used only for testing.
5. **Explainability:** GNNExplainer shows which flow features drove each prediction.

## Results
| Test set | Accuracy | F1 |
|---|---|---|
| Known attacks (held-out test split) | 99.72% | 99.84% |
| Zero-day: unseen Worms + Backdoor (664 attack + 664 benign flows) | 99.25% | 99.25% |

Notes: training used a class-balanced sample (each attack type capped at 4,000 flows), so these numbers are on a balanced sample, not on real-world traffic ratios (about 96% of the full dataset is benign).

## Dashboard features
- Live metrics for the Known and Zero-Day graphs
- Network graph view (connected neighbourhoods, colour = predicted class)
- Simulate Next Flow, Batch Scan, and Alert History
- Per-flow explanation (GNNExplainer feature importance)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Required files in the same folder: `app.py`, `best_gnn_model.pt`, `known_graph_data.pt`, `zero_day_graph_data.pt`.

## Dataset credit
NF-UNSW-NB15 by Sarhan, Layeghy, Moustafa and Portmann (University of Queensland), derived from UNSW-NB15. See: *NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems* (arXiv:2011.09144).

## Planned
- Baseline comparison (Random Forest vs GNN on the same zero-day split)
- Uncertainty estimation (Monte Carlo Dropout)