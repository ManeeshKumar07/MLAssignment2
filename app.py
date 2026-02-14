"""
Obesity Level Estimation — Streamlit Web Application
======================================================
Interactive app for exploring 6 ML classification models trained on
the Obesity Levels dataset. Features CSV upload, model selection,
metrics display, and confusion matrix visualization.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

matplotlib.use("Agg")

# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Level Estimator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
        color: #0f2027;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0 !important;
    }
    /* Sidebar widget labels */
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    /* Specific fix for radio button text */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] .stRadio label p {
        color: #ffffff !important;
    }
    
    /* Input widgets in interaction area (not sidebar) */
    .stSelectbox label, .stFileUploader label {
         color: #0f2027 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #555555 !important;
        font-weight: 600;
        font-size: 0.9rem;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0f2027 !important;
        font-weight: 700;
        font-size: 1.8rem;
        font-family: 'Inter', sans-serif;
    }

    /* Headers */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0f2027 0%, #2c5364 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #444;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f2027 !important;
        color: white !important;
        border-bottom: none;
    }

    /* Tables */
    .stDataFrame {
         border-radius: 10px;
         overflow: hidden;
         border: 1px solid #eee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# LOAD MODELS & ARTIFACTS
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


@st.cache_resource
def load_models():
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_resource
def load_artifacts():
    artifacts = {}
    # Load preprocessor (pipeline) and input features
    for name in ["preprocessor", "input_features", "target_encoder", "target_names"]:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
            
    # Load metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            artifacts["metrics_results"] = json.load(f)
    return artifacts


models = load_models()
artifacts = load_artifacts()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚖️ Obesity Estimator v1.2")
    st.markdown("### Classification Dashboard")
    st.markdown("---")

    # Model selection dropdown
    selected_model = st.selectbox(
        "🤖 Select Model",
        list(models.keys()),
        index=0,
        help="Choose a classification model to evaluate",
    )

    st.markdown("---")

    # Data Source Selection
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Select Source",
        ["Upload CSV", "Test Dataset"],
        help="Choose between uploading your own file or using the pre-loaded test data."
    )

    upload_df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Upload test data CSV. The file should have the same features as the training data.",
        )
        if uploaded_file is not None:
             try:
                upload_df = pd.read_csv(uploaded_file)
             except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        # Test Data (formerly Demo Data)
        demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_data.csv")
        if os.path.exists(demo_path):
            # Custom styled info box for better contrast on dark sidebar
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255, 255, 255, 0.1); 
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 8px;
                    padding: 12px;
                    color: #e0e0e0;
                    font-size: 0.9rem;
                    margin-bottom: 10px;">
                    <strong>ℹ️ Using Test Dataset:</strong><br>
                    <code>test_data.csv</code> ({os.path.getsize(demo_path)/1024:.1f} KB)
                </div>
                """,
                unsafe_allow_html=True
            )
            upload_df = pd.read_csv(demo_path)
        else:
            st.error("Test dataset not found in `data/test_data.csv`.")

    st.markdown("---")
    st.markdown(
        """
        **Dataset:** Obesity Levels (UCI/Kaggle)  
        **Task:** Estimate Obesity Level (7 Classes)  
        **Features:** 16 input features  
        **Models:** 6 classifiers  
        """
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#ddd; font-size:0.85rem;'>"
        "ML Assignment 2 — BITS WILP<br>M.Tech AIML/DSE"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">⚖️ Obesity Level Estimation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Classifying Obesity Levels using 6 ML Models</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# TAB LAYOUT
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "📋 Comparison Table", "📈 Predictions"])

# ─── TAB 1: MODEL EVALUATION ───
with tab1:
    if "metrics_results" in artifacts:
        all_results = artifacts["metrics_results"]
        target_names = artifacts.get("target_names", [])

        if selected_model in all_results:
            model_data = all_results[selected_model]
            metrics = model_data["metrics"]

            st.markdown(f"### {selected_model} — Performance Metrics")
            st.markdown("")

            # Metric cards row
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
            col3.metric("Precision", f"{metrics['Precision']:.4f}")
            col4.metric("Recall", f"{metrics['Recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['F1']:.4f}")
            col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

            st.markdown("")
            st.markdown("---")

            # Confusion Matrix & Classification Report side by side
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.markdown("#### 🔢 Confusion Matrix")
                cm = np.array(model_data["confusion_matrix"])
                # Create figure with explicit white background for contrast
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="GnBu",
                    xticklabels=target_names if target_names else "auto",
                    yticklabels=target_names if target_names else "auto",
                    ax=ax,
                    linewidths=1,
                    linecolor="#f0f0f0",
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 10, "color": "black", "weight": "bold"},
                )
                
                # Force axes labels to be black
                ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", color="black")
                ax.set_ylabel("Actual", fontsize=12, fontweight="bold", color="black")
                ax.set_title(f"{selected_model}", fontsize=14, fontweight="bold", pad=12, color="black")
                
                # Ticks
                plt.xticks(rotation=45, ha="right", color="black")
                plt.yticks(rotation=0, color="black")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with right_col:
                st.markdown("#### 📄 Classification Report")
                report = model_data["classification_report"].copy()
                if "accuracy" in report:
                    report.pop("accuracy")
                report_df = pd.DataFrame(report).transpose()
                # Format nicely
                st.dataframe(
                    report_df.style.format(
                        {
                            "precision": "{:.4f}",
                            "recall": "{:.4f}",
                            "f1-score": "{:.4f}",
                            "support": "{:.0f}"
                        },
                        na_rep="—",
                    ),
                    use_container_width=True,
                )
        else:
            st.warning(f"No results found for {selected_model}")
    else:
        st.error("Metrics not found. Please run `python model/train_models.py` first.")

# ─── TAB 2: COMPARISON TABLE ───
with tab2:
    if "metrics_results" in artifacts:
        st.markdown("### 📋 Model Comparison — All 6 Classifiers")
        st.markdown("")

        all_results = artifacts["metrics_results"]
        comparison_data = []
        for model_name, data in all_results.items():
            row = {"Model": model_name}
            row.update(data["metrics"])
            comparison_data.append(row)

        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.set_index("Model")

        # Highlight best values
        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color: rgba(113, 178, 128, 0.25); font-weight: bold" if v else "" for v in is_max]

        styled_df = comp_df.style.apply(highlight_max).format("{:.4f}")
        st.dataframe(styled_df, use_container_width=True, height=280)

        st.markdown("")
        st.markdown("---")

        # Bar chart comparison
        st.markdown("### 📊 Visual Comparison")
        metric_to_plot = st.selectbox(
            "Select metric to compare",
            ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            index=0,
            key="comparison_metric_select"
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#134e5e", "#71b280", "#2b5876", "#4e4376", "#a8ff78", "#3a6073"]
        bars = ax.bar(comp_df.index, comp_df[metric_to_plot], color=colors, edgecolor="white", linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, comp_df[metric_to_plot]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        ax.set_ylabel(metric_to_plot, fontsize=13, fontweight="bold")
        ax.set_title(f"{metric_to_plot} Comparison Across Models", fontsize=15, fontweight="bold", pad=15)
        ax.set_ylim(0, min(1.15, comp_df[metric_to_plot].max() + 0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=25, ha="right", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.error("Metrics not found. Please run training first.")

# ─── TAB 3: PREDICTIONS ───
with tab3:
    st.markdown("### 📈 Make Predictions")
    st.markdown("")

    if upload_df is not None:
        try:
            st.success(f"✅ Loaded Data: {upload_df.shape[0]} rows × {upload_df.shape[1]} columns")

            # Show preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(upload_df.head(10), use_container_width=True)

            # Check if target column exists
            has_target = "NObeyesdad" in upload_df.columns
            
            # Load required artifacts for inference
            preprocessor = artifacts.get("preprocessor")
            input_features = artifacts.get("input_features", [])
            target_encoder = artifacts.get("target_encoder")

            if not preprocessor or not input_features:
                 st.error("Preprocessor or Input Features not found. Please retrain models.")
                 st.stop()

            # Separate Target if present
            if has_target:
                y_true_raw = upload_df["NObeyesdad"]
                X_upload_raw = upload_df.drop("NObeyesdad", axis=1)
                if "NObeyesdad_Label" in upload_df.columns:
                     X_upload_raw = X_upload_raw.drop("NObeyesdad_Label", axis=1)
            else:
                y_true_raw = None
                X_upload_raw = upload_df.copy()

            # Validate Columns
            if set(input_features).issubset(set(X_upload_raw.columns)):
                # Use the selected model for predictions
                model = models.get(selected_model)
                if model is not None:
                    
                    # ─── PREPROCESSING FOR INFERENCE ───
                    # 1. Filter to required columns explicitly
                    X_input = X_upload_raw[input_features].copy()
                    
                    # 2. Transform using the pipeline ( Handles Scaling & OneHot )
                    try:
                        # Ensure input dtypes match (convert object cols to string like in training)
                        # We must match the logic:  X[col] = le.fit_transform(X[col].astype(str))
                        # In pipeline we used OneHotEncoder. It handles strings.
                        # But we should ensure we don't pass complex objects. 
                        # Safe bet: cast categorical columns to string if needed.
                        # But preprocessor should handle it if dataframe is passed.
                        
                        X_processed = preprocessor.transform(X_input)
                        
                    except Exception as e:
                        st.error(f"Preprocessing Error: {e}. Check if input data values match training categories.")
                        st.stop()

                    # Predict
                    y_pred = model.predict(X_processed)
                    
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_processed)
                    else:
                        y_prob = None

                    # Decode predictions
                    if target_encoder:
                        y_pred_label = target_encoder.inverse_transform(y_pred)
                    else:
                        y_pred_label = y_pred

                    # Show predictions
                    pred_df = upload_df.copy()
                    pred_df["Prediction"] = y_pred_label
                    
                    st.markdown(f"#### Predictions using **{selected_model}**")
                    st.dataframe(pred_df, use_container_width=True, height=300)

                    # If target exists, show evaluation
                    if y_true_raw is not None:
                        st.markdown("---")
                        st.markdown("#### 📊 Evaluation on **Test Data**")
                        
                        # Encode y_true
                        if target_encoder:
                            # If labels are strings, encode. If ints, assume encoded? No, safer to check.
                            if y_true_raw.dtype == object:
                                y_true = target_encoder.transform(y_true_raw)
                            else:
                                y_true = y_true_raw
                        else:
                            y_true = y_true_raw

                        eval_cols = st.columns(6)
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                        mcc = matthews_corrcoef(y_true, y_pred)

                        eval_cols[0].metric("Accuracy", f"{acc:.4f}")
                        if y_prob is not None:
                             try:
                                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                                eval_cols[1].metric("AUC", f"{auc:.4f}")
                             except:
                                eval_cols[1].metric("AUC", "N/A")
                        eval_cols[2].metric("Precision", f"{prec:.4f}")
                        eval_cols[3].metric("Recall", f"{rec:.4f}")
                        eval_cols[4].metric("F1 Score", f"{f1:.4f}")
                        eval_cols[5].metric("MCC", f"{mcc:.4f}")

                        # Classification report
                        st.markdown("#### 📄 Classification Report (Test Data)")
                        report = classification_report(y_true, y_pred, target_names=target_encoder.classes_ if target_encoder else None, output_dict=True)
                        if "accuracy" in report:
                            report.pop("accuracy")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                else:
                    st.error(f"Model '{selected_model}' not loaded.")
            else:
                st.warning(
                    f"⚠️ Feature mismatch. Expected {len(input_features)} features: {input_features[:5]}... "
                    f"Got {len(X_upload_raw.columns)} columns in upload."
                )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info(
            "👈 Select a **Data Source** from the sidebar to start making predictions.\n\n"
            "You can upload your own CSV or use the built-in **Test Dataset**."
        )
