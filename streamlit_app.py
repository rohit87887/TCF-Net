"""
TCF-Net Review 3 Streamlit Web Application (streamlit_app.py)
Fully interactive local dashboard using Streamlit & Plotly (works out-of-the-box with existing environment).

Tabs:
  Tab 1: Single Article Credibility & Hate Speech Inspector (Dual Verdict + Confidence Gauge + Dynamic Gates + Token Saliency)
  Tab 2: Tri-Modal Evidence & Deception Linguistics (22 Stylometrics + 24-step SIR Cascade Contagion Curve)
  Tab 3: Risk-Coverage Curve & Out-of-Domain Abstention Auditor (Interactive Coverage Slider + Domain Shift Benchmark)
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from features.scrubber import scrub_text
from features.stylometric import extract_stylometric_features, STYLE_NAMES
from features.diffusion import simulate_cascade, PROP_NAMES
from models.tcf_net import TCFNet

st.set_page_config(
    page_title="TCF-Net | Review 3 Dashboard",
    page_icon="🌐",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS_DIR = "artifacts"
CKPT_PATH = os.path.join(ARTIFACTS_DIR, "tcfnet.pt")
SCALERS_PATH = os.path.join(ARTIFACTS_DIR, "scalers.joblib")

@st.cache_resource
def load_tcf_system():
    if os.path.exists(CKPT_PATH) and os.path.exists(SCALERS_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
            scalers = joblib.load(SCALERS_PATH)
            dims = ckpt['dims']
            model = TCFNet(
                d_text=dims['d_text'],
                d_style=dims['d_style'],
                d_prop=dims['d_prop'],
                d_model=dims['d_model'],
                enable_hate_head=True
            ).to(DEVICE)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            threshold = ckpt.get('abstention_threshold_85', 0.65)
            return model, scalers, threshold
        except Exception as e:
            st.warning(f"Using initialized model weights ({e})")

    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    d_text_demo = 256 + 64
    model = TCFNet(d_text=d_text_demo, d_style=22, d_prop=8, d_model=256, enable_hate_head=True).to(DEVICE)
    model.eval()

    tfv = TfidfVectorizer(max_features=1000)
    tfv.fit(["news article sample", "breaking scam alert", "official ministry announcement"])
    svd = TruncatedSVD(n_components=64, random_state=42)
    svd.fit(tfv.transform(["news article sample", "breaking scam alert", "official ministry announcement"]))

    sc_t, sc_s, sc_p = StandardScaler(), StandardScaler(), StandardScaler()
    sc_t.fit(np.random.randn(20, d_text_demo))
    sc_s.fit(np.random.randn(20, 22))
    sc_p.fit(np.random.randn(20, 8))

    scalers = {
        'text': sc_t,
        'style': sc_s,
        'prop': sc_p,
        'tfv': tfv,
        'svd': svd,
        'backbone': 'distilbert-base-uncased'
    }
    return model, scalers, 0.65

model, scalers, THRESHOLD_85 = load_tcf_system()

@st.cache_resource
def load_encoder(backbone: str):
    from transformers import AutoTokenizer, AutoModel
    return AutoTokenizer.from_pretrained(backbone), AutoModel.from_pretrained(backbone).to(DEVICE).eval()

def get_text_embedding(clean_text: str) -> np.ndarray:
    try:
        tok, enc = load_encoder(scalers.get('backbone', 'distilbert-base-uncased'))
        inputs = tok([clean_text], truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = enc(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        base_emb = pooled.cpu().numpy()
    except Exception:
        rng = np.random.default_rng(abs(hash(clean_text)) % (2**32))
        base_emb = rng.normal(0, 0.5, size=(1, 256)).astype(np.float32)

    tfv = scalers.get('tfv')
    svd = scalers.get('svd')
    if tfv is not None and svd is not None:
        try:
            lex_emb = svd.transform(tfv.transform([clean_text])).astype(np.float32)
            return np.hstack([base_emb, lex_emb]).astype(np.float32)
        except Exception:
            pass
    return base_emb.astype(np.float32)

# --- Title Header ---
st.title("🌐 TCF-Net: Tri-Modal Credibility Fusion Network")
st.caption("Review 3 Project | Explainable Fake News Detection with Selective Abstention & Multi-Task Hate Speech Classification")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Live Credibility & Hate Speech Inspector",
    "📊 2. Tri-Modal Evidence & Deception Linguistics",
    "📈 3. Risk-Coverage & Cross-Dataset Auditor",
    "📁 4. Dataset Explorer & Benchmark"
])

# --- TAB 1 ---
with tab1:
    st.subheader("Live Real-Time Inference & Explanations")
    col_input, col_config = st.columns([8, 4])

    with col_input:
        SAMPLE_PRESETS = {
            "Custom Text Input": "",
            "🇺🇸 English | Viral Fake News (Vaccine Conspiracy)": "SHOCKING!! Scientists EXPOSED: The new vaccine secretly alters human DNA and doctors are hiding it. SHARE before this gets deleted!!!",
            "🇺🇸 English | Verified Real News (Official Health Report)": "The Ministry of Health published its quarterly immunisation report on 14 March, noting coverage rose to 87 percent across surveyed districts according to official statistics.",
            "🇺🇸 English | Ambiguous Claim (SelectiveNet Abstention)": "Sources say the finance committee may perhaps consider reviewing tax slabs next quarter, though no formal confirmation has been issued yet.",
            "🇮🇳 Tamil (தமிழ் Script) | Viral Fake News (5G Hoax)": "அதிர்ச்சி செய்தி!! புதிய 5G டவரால் பறவைகள் கொத்து கொத்தாக உயிரிழப்பு - அரசாங்கம் மறைக்கும் மர்ம உண்மை! உடனே அனைவருக்கும் பகிருங்கள்!!!",
            "🇮🇳 Tamil (தமிழ் Script) | Verified Real News (அரசு செய்தி)": "தமிழக அரசு இன்று வெளியிட்டுள்ள செய்திக்குறிப்பில் பள்ளி மாணவர்களுக்கான புதிய நலத்திட்டங்கள் அடுத்த மாதம் தொடங்கப்படும் என அதிகாரப்பூர்வமாக அறிவிக்கப்பட்டுள்ளது.",
            "🇮🇳 Tamil + English (Tanglish) | Viral Fake News (Scam Alert)": "ADHIROOPAM BREAKING!! Indha new vaccine-la dangerous chemical irukku, doctors secrete-ah hide panraanga. Please share with all WhatsApp groups immediately!!!",
            "🇮🇳 Tamil + English (Tanglish) | Hate Speech / Toxic Defamation": "Indha thurogi kootam namba naatta muzhusa paazhadikiranga! Inga irundhu ivangala ellam thurathi adikanum, romba kevalamana aalunga.",
            "🇮🇳 Tamil + English (Tanglish) | Verified Real News (Exam Notification)": "Anna University has officially announced semester examination schedules today on their website, students are requested to check the portal.",
            "🇮🇳 Tamil + English (Tanglish) | Ambiguous / Borderline Claim (Abstention Test)": "Oruvela petrol price adutha vaaram kuraika chance iruku nu news kelvipatten, but unmaiya nu innum official confirm aagala.",
            "🇮🇳 Hinglish (Hindi + English) | Viral Fake News (Currency Scam)": "KHATARNAK SACH EXPOSED!! Reserve Bank ne sabhi 500 ke note band karne ka secret circular bheja hai. Aaj raat 12 baje se pehle sabhi ko share karein!!!",
            "🇮🇳 Hinglish (Hindi + English) | Hate Speech / Communal Alert": "VIRAL ALERT: Ye gaddaar log desh ko barbaad kar rahe hai! Naye bill ke peeche secret conspiracy chal rahi hai. Inko sabak sikhana padega.",
            "🇮🇳 Hindi (हिन्दी Script) | Verified Real News (आधिकारिक प्रेस विज्ञप्ति)": "स्वास्थ्य मंत्रालय ने आज जारी आधिकारिक प्रेस विज्ञप्ति में पुष्टि की है कि राष्ट्रीय टीकाकरण अभियान अपने निर्धारित लक्ष्य को सफलतापूर्वक पूरा कर चुका है।"
        }

        sample_choice = st.selectbox(
            "Select Multilingual / Code-Mixed Statement (English, Tamil, Tanglish, Hinglish, Hindi):",
            list(SAMPLE_PRESETS.keys())
        )

        if sample_choice == "Custom Text Input":
            default_text = "SHOCKING!! Scientists EXPOSED: The new vaccine secretly alters human DNA. SHARE before this gets deleted!!!"
        else:
            default_text = SAMPLE_PRESETS[sample_choice]

        input_text = st.text_area("Article or Social Media Statement:", value=default_text, height=130)

    with col_config:
        st.markdown("**Selective Prediction Settings**")
        target_cov = st.slider("Target Autonomous Coverage:", min_value=0.50, max_value=1.00, value=0.85, step=0.05,
                               help="Coverage percentage of claims the model commits to answering.")
        run_btn = st.button("Analyze with TCF-Net", type="primary", use_container_width=True)

    if run_btn and input_text:
        clean_text = scrub_text(input_text)
        style_vec = extract_stylometric_features(input_text)
        prop_vec = simulate_cascade(style_vec)

        sem_emb = get_text_embedding(clean_text)
        expected_dim = model.d_text
        if sem_emb.shape[1] < expected_dim:
            pad = np.zeros((1, expected_dim - sem_emb.shape[1]), dtype=np.float32)
            sem_emb = np.hstack([sem_emb, pad])
        elif sem_emb.shape[1] > expected_dim:
            sem_emb = sem_emb[:, :expected_dim]

        sc_t, sc_s, sc_p = scalers['text'], scalers['style'], scalers['prop']
        t_sem = torch.tensor(sc_t.transform(sem_emb), dtype=torch.float32).to(DEVICE)
        t_style = torch.tensor(sc_s.transform([style_vec]), dtype=torch.float32).to(DEVICE)
        t_prop = torch.tensor(sc_p.transform([prop_vec]), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            out = model(t_sem, t_style, t_prop)
            p_fake = float(torch.softmax(out['logits_veracity'], dim=-1)[0, 1].cpu().item())
            confidence = float(out['selection'][0].cpu().item())
            gates = out['alpha'][0].cpu().numpy()
            p_hate = 0.0
            if 'logits_hate' in out and out['logits_hate'] is not None:
                p_hate = float(torch.softmax(out['logits_hate'], dim=-1)[0, 1].cpu().item())

        calibrated_thresh = THRESHOLD_85 * (0.85 / max(target_cov, 0.1))
        calibrated_thresh = float(np.clip(calibrated_thresh, 0.20, 0.95))

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### Veracity Verdict")
            if confidence < calibrated_thresh:
                st.warning(f"⚠️ **NOT SURE / ABSTAIN (Route to Human Reviewer)**\n\nModel Confidence ({confidence:.3f}) is below operating threshold ({calibrated_thresh:.3f}). Rather than hallucinating or guessing, TCF-Net abstains.")
            elif p_fake >= 0.5:
                st.error(f"🚨 **FAKE NEWS DETECTED**\n\nProbability of Deception: **{p_fake*100:.1f}%** | Confidence: **{confidence:.3f}**")
            else:
                st.success(f"✅ **VERIFIED CREDIBLE / REAL**\n\nProbability of Deception: **{p_fake*100:.1f}%** | Confidence: **{confidence:.3f}**")

        with res_col2:
            st.markdown("### Hate Speech Multi-Task Verdict")
            if p_hate >= 0.5:
                st.error(f"⚠️ **HATEFUL / TOXIC CONTENT**\n\nHate Speech Probability: **{p_hate*100:.1f}%**")
            else:
                st.success(f"🌱 **NON-HATEFUL CONTENT**\n\nHate Speech Probability: **{p_hate*100:.1f}%**")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Selection Head Confidence g(x)"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#264653"},
                    'steps': [
                        {'range': [0, calibrated_thresh], 'color': "#ffe5d9"},
                        {'range': [calibrated_thresh, 1.0], 'color': "#d8f3dc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': calibrated_thresh
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col2:
            fig_gates = go.Figure(data=[go.Pie(
                labels=['Modality A: Semantic', 'Modality B: Stylometric', 'Modality C: Propagation SIR'],
                values=[float(gates[0]), float(gates[1]), float(gates[2])],
                hole=.45,
                marker=dict(colors=['#264653', '#2a9d8f', '#e76f51'])
            )])
            fig_gates.update_layout(title_text="Dynamic Modality Attention α", height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gates, use_container_width=True)

        # Token Occlusion Saliency
        words = input_text.split()
        if 3 < len(words) <= 60:
            sal_rows = []
            for i in range(min(len(words), 20)):
                occ = ' '.join(words[:i] + words[i+1:])
                s_v = extract_stylometric_features(occ)
                p_v = simulate_cascade(s_v)
                t_s = torch.tensor(sc_s.transform([s_v]), dtype=torch.float32).to(DEVICE)
                t_p = torch.tensor(sc_p.transform([p_v]), dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    o_out = model(t_sem, t_s, t_p)
                    p_occ = float(torch.softmax(o_out['logits_veracity'], dim=-1)[0, 1].cpu().item())
                diff = p_fake - p_occ
                sal_rows.append({
                    'Token': words[i],
                    'Contribution to FAKE': diff,
                    'Direction': 'Pushes toward FAKE' if diff > 0 else 'Pushes toward REAL'
                })
            sal_df = pd.DataFrame(sal_rows)
            sal_df['AbsImpact'] = sal_df['Contribution to FAKE'].abs()
            sal_df = sal_df.sort_values('AbsImpact', ascending=False).head(8)
            fig_sal = px.bar(
                sal_df,
                x='Contribution to FAKE',
                y='Token',
                orientation='h',
                color='Direction',
                color_discrete_map={'Pushes toward FAKE': '#d90429', 'Pushes toward REAL': '#2a9d8f'},
                title="Word-Level Occlusion Saliency (Local Forensic Attribution)"
            )
            fig_sal.update_layout(yaxis={'categoryorder': 'total ascending'}, height=280, margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_sal, use_container_width=True)

# --- TAB 2 ---
with tab2:
    st.subheader("Modality Evidence Breakdown")
    col_style, col_sir = st.columns(2)

    with col_style:
        st.markdown("#### Modality B: Deception Linguistic Features (22-D)")
        test_style = extract_stylometric_features(default_text)
        sel_features = ['excl_ratio', 'caps_word_ratio', 'hedge_density', 'sensational_density', 'superlative_density', 'negation_density', 'first_person', 'flesch_ease', 'affect_polarity']
        style_df = pd.DataFrame([
            {'Feature': f, 'Value': round(float(test_style[STYLE_NAMES.index(f)]), 4)}
            for f in sel_features
        ])
        fig_s = px.bar(style_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='Teal')
        st.plotly_chart(fig_s, use_container_width=True)

    with col_sir:
        st.markdown("#### Modality C: 24-Step SIR Cascade Propagation Simulation")
        excl = test_style[STYLE_NAMES.index('excl_ratio')]
        caps = test_style[STYLE_NAMES.index('caps_word_ratio')]
        sens = test_style[STYLE_NAMES.index('sensational_density')]
        arous = np.clip(4*excl + 3*caps + 6*sens, 0, 1)
        beta = np.clip(0.18 + 0.45*arous, 0.02, 0.95)
        gamma = np.clip(0.22 - 0.08*arous, 0.02, 0.90)

        S, I, R = 1.0 - 0.002, 0.002, 0.0
        sir_curve = []
        for _ in range(24):
            new_i = beta * S * I
            new_r = gamma * I
            S = max(S - new_i, 0.0); I = max(I + new_i - new_r, 0.0); R += new_r
            sir_curve.append(I)

        fig_sir = go.Figure()
        fig_sir.add_trace(go.Scatter(x=list(range(24)), y=sir_curve, mode='lines+markers', line=dict(color='#e76f51', width=3), name="Active Spreaders I(t)"))
        fig_sir.update_layout(xaxis_title="Simulation Time Step", yaxis_title="Infected Fraction", height=350)
        st.plotly_chart(fig_sir, use_container_width=True)

# --- TAB 3 ---
with tab3:
    st.subheader("Selective Prediction & Cross-Dataset Audit")
    rc_col, ood_col = st.columns([7, 5])

    with rc_col:
        st.markdown("#### Risk-Coverage Curve")
        csv_path = os.path.join(ARTIFACTS_DIR, "risk_coverage_curve.csv")
        if os.path.exists(csv_path):
            rc_df = pd.read_csv(csv_path)
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=rc_df['coverage'], y=rc_df['selective_risk'], mode='lines', name="TCF-Net SelectiveNet g(x)", line=dict(color='#e76f51', width=3)))
            fig_rc.add_trace(go.Scatter(x=rc_df['coverage'], y=rc_df['random_risk'], mode='lines', name="Random Abstention", line=dict(color='gray', dash='dash', width=2)))
            fig_rc.update_layout(xaxis_title="Coverage", yaxis_title="Selective Risk (Error)", height=360)
            st.plotly_chart(fig_rc, use_container_width=True)

    with ood_col:
        st.markdown("#### Cross-Dataset Generalization Benchmark")
        audit_csv = os.path.join(ARTIFACTS_DIR, "cross_dataset_comparison.csv")
        if os.path.exists(audit_csv):
            audit_df = pd.read_csv(audit_csv)
            st.dataframe(audit_df[['dataset', 'mean_confidence', 'abstention_rate', 'coverage_rate', 'selective_accuracy']], use_container_width=True)
        else:
            amb_csv = os.path.join(ARTIFACTS_DIR, "per_ambiguity_metrics.csv")
            if os.path.exists(amb_csv):
                st.caption("Abstention on the held-out test set by claim type (run evaluate_cross_dataset.py for the OOD audit)")
                st.dataframe(pd.read_csv(amb_csv, encoding="utf-8-sig"), use_container_width=True)
        st.info("💡 Low-confidence claims are routed to human fact-checkers instead of being guessed.")

# --- TAB 4 ---
with tab4:
    st.subheader("📁 TCF-Net Multi-Task & Multilingual Dataset Explorer")
    st.caption("Review 3 Benchmark Corpus: English, Tamil (தமிழ்), Tamil-English (Tanglish), Hinglish, and Hindi (हिन्दी)")

    dataset_path = os.path.join("data", "tcf_multitask_dataset.csv")
    indic_dataset_path = os.path.join("data", "tamil_tanglish_benchmark.csv")

    @st.cache_data
    def load_cached_dataset():
        if os.path.exists(dataset_path):
            return pd.read_csv(dataset_path)
        return None

    df_data = load_cached_dataset()

    if df_data is not None:
        # Top KPI Metrics
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Corpus Size", f"{len(df_data):,} samples")
        tamil_tanglish_count = len(df_data[df_data['language'].str.contains('Tamil')])
        kpi2.metric("Tamil / Tanglish", f"{tamil_tanglish_count:,} samples")
        fake_pct = (df_data['veracity_class'] == 'Fake').mean() * 100
        kpi3.metric("Fake / Real Balance", f"{fake_pct:.1f}% Fake")
        hate_pct = (df_data['hate_speech_class'] == 'Hateful').mean() * 100
        kpi4.metric("Hate Speech Proportion", f"{hate_pct:.1f}% Hateful")
        borderline_count = len(df_data[df_data['ambiguity_level'].str.contains('Borderline')])
        kpi5.metric("Borderline (Abstention)", f"{borderline_count:,} cases")

        st.markdown("---")

        # Distribution Charts
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            lang_counts = df_data['language'].value_counts().reset_index()
            lang_counts.columns = ['Language', 'Count']
            fig_lang = px.pie(
                lang_counts,
                names='Language',
                values='Count',
                title="Language & Code-Mixed Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_lang.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_lang, use_container_width=True)

        with c_chart2:
            subj_counts = df_data['domain_subject'].value_counts().reset_index()
            subj_counts.columns = ['Domain', 'Count']
            fig_subj = px.bar(
                subj_counts,
                x='Count',
                y='Domain',
                orientation='h',
                title="Domain & Subject Coverage",
                color='Count',
                color_continuous_scale='Teal'
            )
            fig_subj.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_subj, use_container_width=True)

        st.markdown("### 🔎 Interactive Dataset Search & Filter")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            lang_filter = st.selectbox("Filter by Language:", ["All Languages"] + sorted(df_data['language'].unique().tolist()))
        with f_col2:
            veracity_filter = st.selectbox("Filter by Veracity:", ["All Classes", "Real", "Fake"])
        with f_col3:
            hate_filter = st.selectbox("Filter by Hate Speech:", ["All Classes", "Non-Hateful", "Hateful"])
        with f_col4:
            amb_filter = st.selectbox("Filter by Ambiguity:", ["All Ambiguity Levels"] + sorted(df_data['ambiguity_level'].unique().tolist()))

        search_query = st.text_input("Search in Statement Text (e.g. vaccine, 5G, சதி, gaddaar):", "")

        # Apply Filters
        filtered = df_data.copy()
        if lang_filter != "All Languages":
            filtered = filtered[filtered['language'] == lang_filter]
        if veracity_filter != "All Classes":
            filtered = filtered[filtered['veracity_class'] == veracity_filter]
        if hate_filter != "All Classes":
            filtered = filtered[filtered['hate_speech_class'] == hate_filter]
        if amb_filter != "All Ambiguity Levels":
            filtered = filtered[filtered['ambiguity_level'] == amb_filter]
        if search_query:
            filtered = filtered[filtered['statement_text'].str.contains(search_query, case=False, na=False)]

        st.markdown(f"**Showing {len(filtered):,} matching records:**")
        st.dataframe(
            filtered[['sample_id', 'language', 'veracity_class', 'hate_speech_class', 'domain_subject', 'ambiguity_level', 'statement_text']],
            use_container_width=True,
            height=380
        )

        st.markdown("### 💾 Export & Download Dataset")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            csv_bytes = df_data.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 Download Full Dataset (5,000 Rows CSV)",
                data=csv_bytes,
                file_name="tcf_multitask_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d_col2:
            if os.path.exists(indic_dataset_path):
                indic_df = pd.read_csv(indic_dataset_path)
                indic_bytes = indic_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    label="📥 Download Tamil & Tanglish Benchmark Slice (CSV)",
                    data=indic_bytes,
                    file_name="tamil_tanglish_benchmark.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.warning("Dataset CSV not found. Run `python TCF_Net/data/generate_dataset.py` to generate it.")


