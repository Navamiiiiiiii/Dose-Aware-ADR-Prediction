"""
Streamlit Web Dashboard: Dose-Aware Polypharmacy Adverse Drug Reaction Prediction Engine
Interactive Clinician & Researcher Interface fulfilling Blueprint 8.
Features 6 Comprehensive Tabs:
1. 💊 Monopharmacy & Cold-Start SMILES Prediction
2. 🧪 Drug-Combination & Polypharmacy Risk Predictor
3. 🎚️ Dose-Aware Counterfactual Engine & Bliss Synergy
4. 🧬 Biomedical Knowledge Graph (PrimeKG & Decagon Explorer)
5. 🔍 Explainable AI (XAI: Saliency, SHAP & Biological Pathways)
6. 📊 Benchmark Evaluation, Calibration & Safety Guardrails
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import adr_engine
from adr_engine import (
    ADR_CLASSES, ADR_LABELS, num_classes,
    DRUG_LIBRARY, TRAIN_DRUGS, COLD_START_DRUGS,
    ResidualEdgeConvNet, CounterfactualEngine,
    BiomedicalKnowledgeGraph, ExplainabilityModule,
    construct_edge_features, encode_arbitrary_drug
)

st.set_page_config(
    page_title="Dose-Aware Polypharmacy ADR Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK PURPLE + ELECTRIC BLUE DASHBOARD THEME
# UI STYLING ONLY — NO MODEL OR PREDICTION CHANGES
# ============================================================

st.markdown("""
<style>
/* Main application background */
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(99, 66, 190, 0.18), transparent 40%),
        radial-gradient(ellipse at 5% 90%, rgba(38, 96, 210, 0.12), transparent 45%),
        #0B1020;
    color: #E8ECFF;
}

/* Main content width and spacing */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    color: #F0EDFF !important;
    letter-spacing: -0.4px;
}

h1 {
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

/* Body text */
p, label, li {
    color: #CDD5F0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10152B;
    border-right: 1px solid #30365B;
}

/* Navigation tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #111831;
    padding: 10px;
    border: 1px solid #29355D;
    border-radius: 16px;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    background: #1B2444;
    color: #B8C5E8;
    border-radius: 11px;
    padding: 10px 16px;
    font-weight: 600;
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7446E8, #2875F0) !important;
    color: #FFFFFF !important;
    border-color: #8D86FF !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: transparent;
}

/* Input fields and dropdowns */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background-color: #171F3B !important;
    color: #F3F5FF !important;
    border-color: #39446D !important;
    border-radius: 10px !important;
}

input {
    color: #F3F5FF !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8048EF, #2878F5);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.65rem 1.5rem;
    box-shadow: 0 5px 20px rgba(91, 75, 230, 0.25);
}

.stButton > button:hover {
    border-color: #A59AFF;
    color: #FFFFFF;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1A2345, #151B35);
    border: 1px solid #354579;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

div[data-testid="stMetricValue"] {
    color: #A9A1FF;
    font-weight: 800;
}

div[data-testid="stMetricLabel"] {
    color: #BAC7ED;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #34446F;
    border-radius: 12px;
    overflow: hidden;
}

/* Information, warning and success messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #394878;
}

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

/* Dividers */
hr {
    border-color: #2C365D;
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #98A8D2 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 9px;
    height: 9px;
}

::-webkit-scrollbar-track {
    background: #0D1328;
}

::-webkit-scrollbar-thumb {
    background: #46558D;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #7868D9;
}
</style>
""", unsafe_allow_html=True)


# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #546E7A; margin-bottom: 20px; }
    .metric-card { background-color: #F8F9FA; border-radius: 8px; padding: 15px; border-left: 5px solid #1E88E5; margin-bottom: 15px; }
    .alert-high { color: #D32F2F; font-weight: 700; }
    .alert-low { color: #388E3C; font-weight: 600; }
    .badge-synergy { background-color: #E8F5E9; color: #2E7D32; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Title & Abstract
st.markdown("<div class='main-header'>💊 Dose-Aware Polypharmacy Adverse Drug Reaction Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>ResidualEdgeConvNet Graph Neural Networks with Inductive Cold-Start & Bliss Independence Synergy</div>", unsafe_allow_html=True)

# Cache Model and Engine
@st.cache_resource
def load_cached_engine():
    # Load the model trained by run_experiments.py (never use random weights).
    checkpoint_path = os.path.join(script_dir, "outputs", "adr_edgeconv_reproducible.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}. "
            "Run the checkpoint-saving training script first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ResidualEdgeConvNet(node_dim=70, edge_dim=11, hidden_dim=128, out_dim=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    taus = np.asarray(checkpoint["thresholds"], dtype=float)
    if taus.shape != (num_classes,):
        raise ValueError(f"Expected {num_classes} thresholds, got {taus.shape}")

    engine = CounterfactualEngine(model, taus)
    kg = BiomedicalKnowledgeGraph()
    xai = ExplainabilityModule(model)
    return engine, kg, xai, taus

try:
    engine, kg, xai, calibrated_taus = load_cached_engine()
except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
    st.error(f"Unable to load trained model: {exc}")
    st.stop()

st.sidebar.success("Trained EdgeConv checkpoint loaded")
st.sidebar.caption("Research simulation only; not validated for clinical decisions.")

# Sidebar: Global Patient Profile
st.sidebar.header("👤 Patient Demographics & Physiology")
patient_age = st.sidebar.slider("Patient Age (Years)", min_value=18, max_value=95, value=72, step=1)
patient_sex = st.sidebar.radio("Sex", options=["Female", "Male"], index=0)
sex_binary = 1 if patient_sex == "Male" else 0
renal_impairment = st.sidebar.checkbox("Renal Impairment (eGFR < 60 mL/min)", value=True)
renal_binary = 1 if renal_impairment else 0

st.sidebar.markdown("---")
st.sidebar.info("""
**Physiological Frailty Multiplier:**
- Age > 65: +35% baseline sensitivity
- Renal Impairment: +45% systemic exposure & reduced clearance
""")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💊 Single-Drug & SMILES",
    "🧪 Drug-Combination Prediction",
    "🎚️ Counterfactual Titration",
    "🧬 Knowledge Graph (PrimeKG)",
    "🔍 Explainable AI (XAI)",
    "📊 Benchmarks & Safety"
])

# ==============================================================================
# TAB 1: Monopharmacy & Cold-Start SMILES Prediction
# ==============================================================================
with tab1:
    st.subheader("Blueprint 1 & 2: Monopharmacy & Cold-Start Molecular Structure Prediction")
    
    mode = st.radio("Input Mode", options=["Select from Pharmacopeia Library (25 Drugs)", "Enter Custom Chemical SMILES (Cold-Start Molecule)"], horizontal=True)
    
    col_a, col_b = st.columns([1, 1])
    
    if mode == "Select from Pharmacopeia Library (25 Drugs)":
        with col_a:
            selected_drug = st.selectbox("Select Medicine", options=sorted(list(DRUG_LIBRARY.keys())), index=0)
            d_info = DRUG_LIBRARY[selected_drug]
            st.write(f"**Therapeutic Class:** `{d_info['class']}` | **Molecular Target:** `{d_info['target']}`")
            st.write(f"**SMILES:** `{d_info['smiles']}`")
            st.write(f"**Therapeutic Window:** `{d_info['min_d']} - {d_info['max_d']} {d_info['unit']}` | **DDD:** `{d_info['ddd']} {d_info['unit']}`")
            
            dose_val = st.slider(f"{selected_drug} Dose ({d_info['unit']}/day)",
                                 min_value=float(d_info['min_d'] * 0.5),
                                 max_value=float(d_info['max_d'] * 1.5),
                                 value=float(d_info['ddd']),
                                 step=float(max(0.1, round(d_info['ddd'] / 10.0, 1))))
            
            p_mono = engine.predict_single_drug(selected_drug, dose_val, patient_age, sex_binary, renal_binary)
            
        with col_b:
            st.markdown("#### 📋 Predicted Adverse Reaction Probabilities")
            res_data = []
            for lbl, p, tau in zip(ADR_LABELS, p_mono, calibrated_taus):
                is_alert = p >= tau
                status = "⚠️ ALERT" if is_alert else "✅ LOW RISK"
                res_data.append({
                    "Adverse Reaction (SOC)": lbl,
                    "Predicted Risk": f"{p*100:.1f}%",
                    "Calibrated Threshold": f"{tau*100:.1f}%",
                    "Clinical Status": status
                })
            df_res = pd.DataFrame(res_data)
            st.dataframe(df_res, use_container_width=True)
            
    else:
        with col_a:
            st.markdown("##### 🔬 Novel Molecule Input (Cold-Start Inductive Transfer)")
            custom_name = st.text_input("Compound Name", value="Novel Antithrombotic Factor Xa Inhibitor")
            custom_smiles = st.text_input("Canonical SMILES", value="O=C1N(c2ccc(N3CCOCC3=O)cc2)CC(CNC(=O)c4ccc(Cl)s4)O1")
            custom_class = st.selectbox("Presumed Pharmacological Class", options=adr_engine.CLASSES, index=6) # DOAC
            custom_target = st.selectbox("Presumed Molecular Target", options=adr_engine.TARGETS, index=5) # FXa
            custom_dose = st.number_input("Estimated Daily Dose (mg)", min_value=1.0, max_value=2000.0, value=20.0)
            
            encoded_feat = encode_arbitrary_drug(custom_smiles, custom_class, custom_target)
            edge_feat = construct_edge_features(custom_dose, 20.0, 0.0, 100.0, patient_age, sex_binary, renal_binary)
            p_novel = engine.predict_pair(encoded_feat, np.zeros_like(encoded_feat), edge_feat)
            
        with col_b:
            st.markdown(f"#### 🧬 Inductive Cold-Start Predictions for '{custom_name}'")
            st.info(
    "The model encodes SMILES using custom molecular descriptors "
    "and deterministic hashed 4-character fragments within a "
    "32-dimensional descriptor representation, combined with "
    "pharmacological class and target features to form a "
    "70-dimensional input vector. Predictions for unseen drugs "
    "are experimental and based on synthetic-data training."
)
            res_data = []
            for lbl, p, tau in zip(ADR_LABELS, p_novel, calibrated_taus):
                status = "⚠️ ALERT" if p >= tau else "✅ LOW RISK"
                res_data.append({
                    "Adverse Reaction (SOC)": lbl,
                    "Predicted Risk": f"{p*100:.1f}%",
                    "Calibrated Threshold": f"{tau*100:.1f}%",
                    "Status": status
                })
            st.dataframe(pd.DataFrame(res_data), use_container_width=True)

# ==============================================================================
# TAB 2: Drug Combination Prediction
# ==============================================================================
with tab2:
    st.subheader("Blueprint 3 & 5: Polypharmacy Combination & Pairwise Synergy Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.selectbox("Select Drug A", options=sorted(list(DRUG_LIBRARY.keys())), index=0)
        da_info = DRUG_LIBRARY[drug_a]
        dose_a = st.slider(f"{drug_a} Daily Dose (mg)", min_value=float(da_info['min_d']*0.5), max_value=float(da_info['max_d']*1.5), value=float(da_info['ddd']))
        
    with col2:
        drug_b_options = [d for d in sorted(list(DRUG_LIBRARY.keys())) if d != drug_a]
        drug_b = st.selectbox("Select Drug B", options=drug_b_options, index=0)
        db_info = DRUG_LIBRARY[drug_b]
        dose_b = st.slider(f"{drug_b} Daily Dose (mg)", min_value=float(db_info['min_d']*0.5), max_value=float(db_info['max_d']*1.5), value=float(db_info['ddd']))
        
    comb_res = engine.analyze_combination(drug_a, drug_b, dose_a, dose_b, patient_age, sex_binary, renal_binary)
    
    st.markdown("#### 🔬 Polypharmacy Synergy Breakdown (Bliss Independence Decomposition)")
    
    comb_table = []
    for i, lbl in enumerate(ADR_LABELS):
        p_c = comb_res['p_combination'][i]
        p_a = comb_res['p_mono_a'][i]
        p_b = comb_res['p_mono_b'][i]
        p_bl = comb_res['p_bliss'][i]
        ex = comb_res['excess_synergy'][i]
        tau = calibrated_taus[i]
        
        status = "⚠️ ALERT" if p_c >= tau else "✅ OK"
        syn_desc = (
    "Positive model-derived Bliss deviation (> +5%)" if ex > 0.05
    else "Negative model-derived Bliss deviation (< -5%)" if ex < -0.05
    else "Near-zero model-derived Bliss deviation (±5%)"
)
        
        comb_table.append({
            "Reaction (SOC)": lbl,
            "Combination Risk P(AB)": f"{p_c*100:.1f}%",
            f"Mono {drug_a} P(A)": f"{p_a*100:.1f}%",
            f"Mono {drug_b} P(B)": f"{p_b*100:.1f}%",
            "Bliss Expected P_ind": f"{p_bl*100:.1f}%",
            "Excess Synergy (Delta P)": f"{ex*100:+.1f}%",
            "Model-Based Bliss Classification": syn_desc,
            "Clinical Alert": status
        })
    st.dataframe(pd.DataFrame(comb_table), use_container_width=True)

# ==============================================================================
# TAB 3: Dose-Aware Counterfactual Engine
# ==============================================================================
with tab3:
    st.subheader("Blueprint 4: Dose-Aware Counterfactual Titration Engine")
    st.markdown("Simulate 'what-if' prescriber decisions by altering Drug A's dose while keeping Drug B and patient variables strictly fixed.")
    
    col_cf1, col_cf2 = st.columns(2)
    with col_cf1:
        cf_drug_a = st.selectbox("Titrate Drug A", options=sorted(list(DRUG_LIBRARY.keys())), index=0, key="cf_a")
        da_cf = DRUG_LIBRARY[cf_drug_a]
        base_dose = st.number_input("Baseline Daily Dose (mg)", min_value=float(da_cf['min_d']*0.2), max_value=float(da_cf['max_d']*2.0), value=float(da_cf['ddd']), key="base_d")
        sim_dose = st.slider("Simulated Counterfactual Dose (mg)", min_value=float(da_cf['min_d']*0.2), max_value=float(da_cf['max_d']*2.5), value=float(da_cf['max_d']), key="sim_d")
        
    with col_cf2:
        cf_drug_b_options = [d for d in sorted(list(DRUG_LIBRARY.keys())) if d != cf_drug_a]
        cf_drug_b = st.selectbox("Fixed Co-Prescribed Drug B", options=cf_drug_b_options, index=0, key="cf_b")
        db_cf = DRUG_LIBRARY[cf_drug_b]
        fixed_dose = st.number_input("Fixed Drug B Dose (mg)", min_value=float(db_cf['min_d']*0.5), max_value=float(db_cf['max_d']*1.5), value=float(db_cf['ddd']), key="fix_d")
        
    cf_res = engine.run_dose_counterfactual(cf_drug_a, cf_drug_b, base_dose, sim_dose, fixed_dose, patient_age, sex_binary, renal_binary)
    
    # Clinical Dose Window Guardrail Status
    if cf_res['in_window']:
        st.success(f"✅ In-Distribution Clinical Window: {cf_drug_a} simulated dose ({sim_dose:.1f} mg) falls within the approved therapeutic range [{cf_res['window_range'][0]}, {cf_res['window_range'][1]}] mg.")
    else:
        st.warning(f"⚠️ Out-of-Distribution (OOD) Guardrail Alert: {sim_dose:.1f} mg is outside standard clinical window [{cf_res['window_range'][0]}, {cf_res['window_range'][1]}] mg! Epistemic uncertainty is high.")
        
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 🚨 Newly Predicted ADRs")
        if cf_res['newly_predicted']:
            for item in cf_res['newly_predicted']:
                st.error(f"**{item['class']}**: Crossed threshold ({item['before']*100:.1f}% ➔ {item['after']*100:.1f}%) [Δ = {item['delta']*100:+.1f}%]")
        else:
            st.write("None crossed above threshold.")
            
    with c2:
        st.markdown("##### 🛡️ Ceased / Mitigated ADRs")
        if cf_res['ceased_reactions']:
            for item in cf_res['ceased_reactions']:
                st.success(f"**{item['class']}**: Mitigated below threshold ({item['before']*100:.1f}% ➔ {item['after']*100:.1f}%) [Δ = {item['delta']*100:+.1f}%]")
        else:
            st.write("No reactions ceased.")
            
    with c3:
        st.markdown("##### 📈 Significant Score Shifts (|Δ| ≥ 8%)")
        if cf_res['shifted_scores']:
            for item in cf_res['shifted_scores']:
                st.info(f"**{item['class']}**: {item['before']*100:.1f}% ➔ {item['after']*100:.1f}% (Δ = {item['delta']*100:+.1f}%)")
        else:
            st.write("No other score shifts ≥ 8%.")

# ==============================================================================
# TAB 4: Biomedical Knowledge Graph (PrimeKG)
# ==============================================================================
with tab4:
    st.subheader("Blueprint 6: Biomedical Knowledge Graph Explorer (PrimeKG & Decagon)")
    st.markdown("Inspect biological mechanism paths between drugs, target proteins, metabolizing enzymes, and adverse reaction phenotypes.")
    
    col_kg1, col_kg2 = st.columns(2)
    with col_kg1:
        kg_drug_a = st.selectbox("Drug A Entity", options=sorted(list(DRUG_LIBRARY.keys())), index=0, key="kg_a")
    with col_kg2:
        kg_drug_b = st.selectbox("Drug B Entity", options=[d for d in sorted(list(DRUG_LIBRARY.keys())) if d != kg_drug_a], index=0, key="kg_b")
        
    target_reaction = st.selectbox("Target Adverse Reaction Phenotype", options=ADR_LABELS, index=0)
    
    kg_explanation = kg.explain_interaction_mechanism(kg_drug_a, kg_drug_b, target_reaction)
    
    st.markdown("#### 🔬 Pharmacological Mechanism Pathway")
    st.info(kg_explanation['biological_mechanism'])
    
    st.markdown("#### 🌐 Knowledge Graph Traversal")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"**{kg_drug_a} PrimeKG Subgraph Targets:**")
        for p in kg_explanation['pathways_a']:
            st.markdown(f"- `{p}`")
    with col_p2:
        st.markdown(f"**{kg_drug_b} PrimeKG Subgraph Targets:**")
        for p in kg_explanation['pathways_b']:
            st.markdown(f"- `{p}`")

# ==============================================================================
# TAB 5: Explainable AI (XAI)
# ==============================================================================
with tab5:
    st.subheader("Blueprint 7: Explainable AI (Gradient-Based Input Attribution)")
    st.markdown("Deconstruct the model's prediction into feature attributions across all 11 continuous dose and patient context variables.")
    
    xai_d1 = st.selectbox("Drug A", options=sorted(list(DRUG_LIBRARY.keys())), index=0, key="xai_d1")
    xai_d2 = st.selectbox("Drug B", options=[d for d in sorted(list(DRUG_LIBRARY.keys())) if d != xai_d1], index=0, key="xai_d2")
    xai_class = st.selectbox("Explain Target ADR", options=ADR_LABELS, index=0, key="xai_cls")
    cls_idx = ADR_LABELS.index(xai_class)
    
    feat_a = DRUG_LIBRARY[xai_d1]["feat"]
    feat_b = DRUG_LIBRARY[xai_d2]["feat"]
    edge_in = construct_edge_features(DRUG_LIBRARY[xai_d1]["ddd"], DRUG_LIBRARY[xai_d1]["ddd"],
                                      DRUG_LIBRARY[xai_d2]["ddd"], DRUG_LIBRARY[xai_d2]["ddd"],
                                      patient_age, sex_binary, renal_binary)
    
    attrib_res = xai.compute_feature_attribution(feat_a, feat_b, edge_in, target_class_idx=cls_idx)
    
    st.markdown(f"#### 🔍 Attribution Summary for '{xai_class}'")
    st.write(attrib_res['narrative'])
    
    df_attrib = pd.DataFrame(attrib_res['ranked_features'])
    df_attrib.columns = ["Input Feature Name", "Gradient Magnitude", "% Contribution"]
    st.dataframe(df_attrib, use_container_width=True)
    
    st.bar_chart(data=df_attrib.set_index("Input Feature Name")["% Contribution"])

# ==============================================================================
# TAB 6: Benchmark Metrics & Safety Guardrails
# ==============================================================================
with tab6:
    st.subheader("Blueprint 8: Comprehensive Quantitative Benchmark Suite")
    st.caption("Recorded results from the latest completed synthetic-data experiment (30 Oct 2026). "
               "These values are not recomputed when the dashboard loads.")
    
    
    benchmark_data = {
        "Evaluation Metric": [
            "Micro F1", "Macro F1", "Micro Precision", "Macro Precision",
            "Micro Recall", "Macro Recall", "Subset Accuracy (Exact Match)",
            "Hamming Accuracy", "Micro ROC-AUC", "Macro ROC-AUC",
            "Micro AUPRC", "Macro AUPRC"
        ],
        "ResidualEdgeConvNet (In-Distribution)": [
            "97.00%", "97.21%", "96.34%", "96.44%",
            "97.67%", "98.05%", "95.90%",
            "99.48%", "99.98%", "99.98%",
            "99.75%", "99.79%"
        ],
        "Baseline MLP (In-Distribution)": [
            "94.70%", "94.46%", "90.34%", "90.07%",
            "99.49%", "99.49%", "92.85%",
            "99.04%", "99.97%", "99.97%",
            "99.67%", "99.71%"
        ],
        "ResidualEdgeConvNet (Cold-Start Unseen)": [
            "84.41%", "82.12%", "83.88%", "81.48%",
            "84.95%", "86.35%", "85.60%",
            "97.78%", "96.10%", "96.68%",
            "88.24%", "88.22%"
        ],
        "Baseline MLP (Cold-Start Unseen)": [
            "79.67%", "77.72%", "79.52%", "79.14%",
            "79.82%", "82.89%", "81.39%",
            "97.12%", "95.72%", "96.99%",
            "87.47%", "89.52%"
        ]
    }
    st.dataframe(
        pd.DataFrame(benchmark_data),
        use_container_width=True
    )
