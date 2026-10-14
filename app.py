# ============================================================
# STEP 5 (v3): Live Demo Dashboard - with Simulation + History + Batch
# GNN Zero-Day Attack Detection Project
#
# HOW TO RUN (on your laptop, in this project folder):
#   streamlit run app.py
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

st.set_page_config(page_title="GNN-IDS | Zero-Day Attack Detector", layout="wide", page_icon="🛡️")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLS = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS',
                 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
                 'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL',
                 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
                 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES',
                 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
                 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES',
                 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
                 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE',
                 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE']


# ============================================================
# DARK THEME CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(145deg, #1a1f2b, #161a24);
        border: 1px solid #2a2f3d;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: left;
    }
    .metric-label { color: #8b92a8; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .metric-green { color: #2ecc71; }
    .metric-red { color: #e74c3c; }
    .metric-orange { color: #f39c12; }
    .metric-blue { color: #3498db; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #e8eaf0;
        margin-top: 10px; margin-bottom: 10px;
    }
    div[data-testid="stSidebar"] { background-color: #12151d; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DEFINITION
# ============================================================
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.classifier(x)
        return out


@st.cache_resource
def load_model_and_data():
    model = GraphSAGE(in_channels=41, hidden_channels=64, out_channels=2).to(device)
    model.load_state_dict(torch.load('best_gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    known_graph = torch.load('known_graph_data.pt', map_location=device, weights_only=False)
    zero_day_graph = torch.load('zero_day_graph_data.pt', map_location=device, weights_only=False)
    return model, known_graph, zero_day_graph


@st.cache_data
def compute_full_metrics(_model, _graph, graph_name):
    _model.eval()
    with torch.no_grad():
        out = _model(_graph.x, _graph.edge_index)
        probs = F.softmax(out, dim=1).cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
    true = _graph.y.cpu().numpy()

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)

    return {
        "preds": preds, "probs": probs, "true": true,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "n_attack": int((preds == 1).sum()), "n_benign": int((preds == 0).sum()),
        "n_nodes": _graph.num_nodes, "n_edges": _graph.edge_index.shape[1],
    }



# --- GRAPH HELPERS START ---
def build_adjacency(edge_index_np, n_nodes):
    adj = [[] for _ in range(n_nodes)]
    for s_, t_ in zip(edge_index_np[0].tolist(), edge_index_np[1].tolist()):
        adj[s_].append(t_)
    return [sorted(set(a)) for a in adj]


def pick_subgraph(adj, preds, focus_node=None, seed=1, per_class_seeds=2, per_seed=14):
    """Pick a few CONNECTED neighbourhoods (ego-networks) so edges are actually visible.
    (A random sample of nodes from a 26k-node graph almost never contains edges.)"""
    rng = np.random.default_rng(seed)
    seeds = []
    if focus_node is not None:
        seeds.append(int(focus_node))
    for cls in (1, 0):
        pool = np.where(preds == cls)[0]
        if len(pool) > 0:
            k = min(per_class_seeds, len(pool))
            seeds.extend(rng.choice(pool, size=k, replace=False).tolist())

    selected = set()
    for sd in seeds:
        selected.add(sd)
        nbrs = adj[sd]
        if nbrs:
            take = min(per_seed, len(nbrs))
            selected.update(rng.choice(nbrs, size=take, replace=False).tolist())

    edges = set()
    for u in selected:
        for v in adj[u]:
            if v in selected and u != v:
                edges.add((min(u, v), max(u, v)))
    return sorted(selected), sorted(edges), seeds


def draw_subgraph(ax, nodes, edges, preds, focus_node=None):
    G_ = nx.Graph()
    G_.add_nodes_from(nodes)
    G_.add_edges_from(edges)
    pos_ = nx.spring_layout(G_, seed=42, k=0.35, iterations=80)
    colors_ = ['#e74c3c' if preds[n] == 1 else '#2ecc71' for n in G_.nodes()]
    nx.draw_networkx_edges(G_, pos_, ax=ax, edge_color='#8a94ad', width=1.3, alpha=0.9)
    nx.draw_networkx_nodes(G_, pos_, ax=ax, node_color=colors_, node_size=130, alpha=0.95)
    if focus_node is not None and focus_node in G_:
        c_ = '#e74c3c' if preds[focus_node] == 1 else '#2ecc71'
        nx.draw_networkx_nodes(G_, pos_, ax=ax, nodelist=[focus_node], node_color=c_,
                               node_size=460, edgecolors='white', linewidths=2.5)
    ax.axis('off')
# --- GRAPH HELPERS END ---


@st.cache_data(show_spinner=False)
def cached_adjacency(_graph, graph_name):
    return build_adjacency(_graph.edge_index.cpu().numpy(), _graph.num_nodes)

model, known_graph, zero_day_graph = load_model_and_data()

# ---- Session state for alert history ----
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts


def log_to_history(graph_label, node_idx, pred_label, confidence, true_label):
    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Graph": graph_label,
        "Node": int(node_idx),
        "Prediction": "Attack" if pred_label == 1 else "Benign",
        "Confidence": f"{confidence:.2f}%",
        "Ground Truth": "Attack" if true_label == 1 else "Benign",
        "Match": "✅" if pred_label == true_label else "❌",
    })
    st.session_state.history = st.session_state.history[:100]  # cap history size


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ GNN-IDS")
    st.markdown("**Explainable Zero-Day Attack Detector**")
    st.markdown("---")

    st.markdown("**Dataset**")
    st.code("NF-UNSW-NB15-v2 (NetFlow)", language=None)
    st.markdown("**Model**")
    st.code("GraphSAGE (2-layer GNN)", language=None)
    st.markdown("---")

    graph_choice = st.radio(
        "Select Graph",
        ["Known Attacks (Test Set)", "Zero-Day Attacks (Unseen: Worms + Backdoor)"]
    )
    active_graph = known_graph if graph_choice.startswith("Known") else zero_day_graph
    graph_key = "known" if graph_choice.startswith("Known") else "zeroday"
    graph_label = "Known Attacks" if graph_key == "known" else "Zero-Day (Unseen)"

    st.markdown("---")
    st.markdown("**🔬 Detailed Flow Inspection**")
    node_idx = st.number_input(
        "Network Flow (Node Index)",
        min_value=0, max_value=active_graph.num_nodes - 1, value=0, step=1
    )
    analyze_btn = st.button("🔍 Analyze This Flow (with XAI)", use_container_width=True)

    st.markdown("---")
    st.markdown("**⚡ Live Traffic Simulation**")
    simulate_btn = st.button("▶️ Simulate Next Flow", use_container_width=True)

    st.markdown("---")
    st.markdown("**📦 Batch Analysis**")
    batch_n = st.slider("Number of flows to scan", 5, 50, 15)
    batch_btn = st.button("⚙️ Run Batch Scan", use_container_width=True)

    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ============================================================
# HEADER + METRIC CARDS
# ============================================================
metrics = compute_full_metrics(model, active_graph, graph_key)

st.markdown(f"""
<div style="margin-bottom:20px;">
    <h1 style="margin-bottom:0; color:#e8eaf0;">Explainable GNN for Zero-Day Attack Detection</h1>
    <p style="color:#8b92a8; margin-top:4px;">Currently viewing: <b style="color:#3498db;">{graph_label}</b> graph</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div>
    <div class="metric-value metric-green">{metrics['accuracy']*100:.2f}%</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Precision</div>
    <div class="metric-value metric-blue">{metrics['precision']*100:.2f}%</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Recall</div>
    <div class="metric-value metric-blue">{metrics['recall']*100:.2f}%</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">F1-Score</div>
    <div class="metric-value metric-blue">{metrics['f1']*100:.2f}%</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Detected Attacks</div>
    <div class="metric-value metric-red">{metrics['n_attack']}</div></div>""", unsafe_allow_html=True)

if graph_key == "known":
    st.caption(f"ℹ️ Cards show the model's score over all {metrics['n_nodes']:,} sampled flows (this includes flows used in training). "
               "Score on the held-out test split from training: Accuracy 99.72% · F1 99.84%.")
else:
    half = metrics['n_nodes'] // 2
    st.caption(f"ℹ️ Zero-day set: {half:,} attack flows from Worms + Backdoor (never seen in training) plus {half:,} benign flows. "
               "Benign traffic is part of training too; only these attack types are unseen.")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 2: Graph overview + Threat summary
# ============================================================
col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.markdown('<div class="section-header">🌐 Network Graph Overview</div>', unsafe_allow_html=True)
    adj = cached_adjacency(active_graph, graph_key)
    focus = int(node_idx) if analyze_btn else None
    sub_nodes, sub_edges, _ = pick_subgraph(adj, metrics['preds'], focus_node=focus)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    draw_subgraph(ax, sub_nodes, sub_edges, metrics['preds'], focus_node=focus)
    st.pyplot(fig)
    cap = "🔴 Predicted Attack &nbsp;|&nbsp; 🟢 Predicted Benign &nbsp;|&nbsp; Lines = similar flows (KNN, k=8). A few connected neighbourhoods are shown, not the whole graph."
    if focus is not None:
        cap += " &nbsp;⚪ White ring = the flow being analyzed."
    st.caption(cap)

with col_right:
    st.markdown('<div class="section-header">📊 Traffic Summary</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    fig2.patch.set_facecolor('#0e1117')
    sizes = [metrics['n_benign'], metrics['n_attack']]
    colors = ['#2ecc71', '#e74c3c']
    ax2.pie(sizes, labels=['Benign', 'Attack'], colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'color': 'white', 'fontsize': 11}, wedgeprops={'width': 0.4})
    ax2.text(0, 0, f"{metrics['n_nodes']}\nFlows", ha='center', va='center',
              fontsize=14, color='white', fontweight='bold')
    st.pyplot(fig2)
    st.markdown(f"""
    - **Total Nodes:** {metrics['n_nodes']}
    - **Total Edges:** {metrics['n_edges']}
    - **Graph Type:** KNN Flow-Similarity Graph
    """)
    if graph_key == "known":
        st.caption("Class-balanced sample (each attack type capped at 4,000 flows). In the full dataset about 96.2% of flows are benign.")
    else:
        st.caption("Balanced 50/50 test set by design.")

st.markdown("---")

# ============================================================
# LIVE SIMULATION (fast, no explainer — just prediction)
# ============================================================
if simulate_btn:
    rand_node = int(np.random.randint(0, active_graph.num_nodes))
    pred_label = int(metrics['preds'][rand_node])
    true_label = int(metrics['true'][rand_node])
    confidence = metrics['probs'][rand_node][pred_label] * 100
    log_to_history(graph_label, rand_node, pred_label, confidence, true_label)

    if pred_label == 1:
        st.error(f"🚨 **Live Flow Detected — Node #{rand_node} → ATTACK** ({confidence:.1f}% confidence)")
    else:
        st.success(f"✅ **Live Flow Detected — Node #{rand_node} → BENIGN** ({confidence:.1f}% confidence)")

# ============================================================
# BATCH ANALYSIS
# ============================================================
if batch_btn:
    with st.spinner(f"Scanning {batch_n} network flows..."):
        rand_nodes = np.random.choice(active_graph.num_nodes, size=batch_n, replace=False)
        rows = []
        for n in rand_nodes:
            n = int(n)
            pred_label = int(metrics['preds'][n])
            true_label = int(metrics['true'][n])
            confidence = metrics['probs'][n][pred_label] * 100
            rows.append({
                "Node": n,
                "Prediction": "Attack" if pred_label == 1 else "Benign",
                "Confidence": f"{confidence:.2f}%",
                "Ground Truth": "Attack" if true_label == 1 else "Benign",
                "Match": "✅" if pred_label == true_label else "❌",
            })
            log_to_history(graph_label, n, pred_label, confidence, true_label)

    batch_df = pd.DataFrame(rows)
    st.session_state.last_batch = batch_df
    st.session_state.last_batch_summary = {
        "attacks": sum(1 for r in rows if r["Prediction"] == "Attack"),
        "benign": sum(1 for r in rows if r["Prediction"] == "Benign"),
        "acc": sum(1 for r in rows if r["Match"] == "✅") / len(rows) * 100,
    }

if "last_batch" in st.session_state:
    st.markdown('<div class="section-header">📦 Batch Scan Results</div>', unsafe_allow_html=True)
    summ = st.session_state.last_batch_summary
    b1, b2, b3 = st.columns(3)
    b1.metric("Attacks Found", summ["attacks"])
    b2.metric("Benign Flows", summ["benign"])
    b3.metric("Batch Accuracy", f"{summ['acc']:.1f}%")
    st.dataframe(st.session_state.last_batch, use_container_width=True, hide_index=True)
    st.markdown("---")

# ============================================================
# FLOW-LEVEL ANALYSIS (full explainer)
# ============================================================
st.markdown('<div class="section-header">🔍 Flow-Level Analysis (Detailed XAI)</div>', unsafe_allow_html=True)

if analyze_btn:
    pred_label = int(metrics['preds'][node_idx])
    true_label = int(metrics['true'][node_idx])
    confidence = metrics['probs'][node_idx][pred_label] * 100
    log_to_history(graph_label, node_idx, pred_label, confidence, true_label)

    r1, r2, r3 = st.columns(3)
    with r1:
        if pred_label == 1:
            st.error(f"🚨 **PREDICTION: ATTACK**  (Node #{node_idx})")
        else:
            st.success(f"✅ **PREDICTION: BENIGN**  (Node #{node_idx})")
    with r2:
        st.metric("Confidence", f"{confidence:.2f}%")
    with r3:
        true_str = "Attack" if true_label == 1 else "Benign"
        match = "✅ Correct" if pred_label == true_label else "❌ Incorrect"
        st.metric("Ground Truth", true_str, delta=match)

    with st.spinner("Running GNNExplainer... (~10-20 sec)"):
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
        )
        explanation = explainer(active_graph.x, active_graph.edge_index, index=int(node_idx))
        node_feat_mask = explanation.node_mask[node_idx].cpu().numpy()
        top_indices = np.argsort(node_feat_mask)[::-1][:8]

    ex1, ex2 = st.columns([1.3, 1])
    with ex1:
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        fig3.patch.set_facecolor('#0e1117')
        ax3.set_facecolor('#0e1117')
        top_features = [FEATURE_COLS[i] for i in top_indices][::-1]
        top_scores = node_feat_mask[top_indices][::-1]
        bar_color = '#e74c3c' if pred_label == 1 else '#2ecc71'
        ax3.barh(top_features, top_scores, color=bar_color)
        ax3.set_xlabel('Importance Score', color='white')
        ax3.tick_params(colors='white')
        for spine in ax3.spines.values():
            spine.set_color('#3a4050')
        st.pyplot(fig3)

    with ex2:
        st.markdown("**Explanation Summary**")
        verdict = "malicious" if pred_label == 1 else "benign"
        top3 = [FEATURE_COLS[i] for i in top_indices[:3]]
        st.markdown(f"""
        The model classified this flow as **{verdict.upper()}** with **{confidence:.1f}%** confidence,
        primarily driven by:
        - `{top3[0]}`
        - `{top3[1]}`
        - `{top3[2]}`

        These are the flow characteristics the GNN weighted most heavily
        (via GNNExplainer) when making this decision.
        """)
else:
    st.info("👈 Select a node index in the sidebar and click **Analyze This Flow** for a full explanation, or use **Simulate** / **Batch Scan** for quicker live-style checks.")

st.markdown("---")

# ============================================================
# ALERT HISTORY + TREND
# ============================================================
st.markdown('<div class="section-header">🕒 Recent Alerts (Session History)</div>', unsafe_allow_html=True)

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)

    h1, h2 = st.columns([1.6, 1])
    with h1:
        st.dataframe(hist_df.head(15), use_container_width=True, hide_index=True)

    with h2:
        # Trend: cumulative attack vs benign over the recorded sequence (oldest -> newest)
        seq = hist_df.iloc[::-1].reset_index(drop=True)
        seq['is_attack'] = (seq['Prediction'] == 'Attack').astype(int)
        seq['cum_attacks'] = seq['is_attack'].cumsum()
        seq['cum_benign'] = (1 - seq['is_attack']).cumsum()

        fig4, ax4 = plt.subplots(figsize=(4.5, 4))
        fig4.patch.set_facecolor('#0e1117')
        ax4.set_facecolor('#0e1117')
        ax4.plot(seq.index, seq['cum_attacks'], color='#e74c3c', label='Attacks', linewidth=2)
        ax4.plot(seq.index, seq['cum_benign'], color='#2ecc71', label='Benign', linewidth=2)
        ax4.set_xlabel('Alert #', color='white')
        ax4.set_ylabel('Cumulative Count', color='white')
        ax4.tick_params(colors='white')
        ax4.legend(facecolor='#1a1f2b', labelcolor='white')
        for spine in ax4.spines.values():
            spine.set_color('#3a4050')
        st.pyplot(fig4)
else:
    st.caption("No alerts yet — use **Simulate Next Flow**, **Run Batch Scan**, or **Analyze This Flow** to generate activity.")