import os
import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from model.utils import load_json, compute_metrics_multiclass, make_confusion_matrix, make_classification_report, safe_predict_proba

st.set_page_config(page_title="ML Assignment 2 - Classification", layout="wide")

ART_DIR = "artifacts"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

st.title("ML Assignment 2: Classification Models Dashboard")

st.write(
    """
This Streamlit app loads **6 trained classification models**, displays their **evaluation metrics**,
and lets you **upload a CSV (test data)** to generate predictions and (if labels exist) a confusion matrix.
"""
)

def artifacts_ready():
    needed = ["metrics.csv", "label_encoder.joblib", "meta.json"]
    return all(os.path.exists(os.path.join(ART_DIR, f)) for f in needed)

if not artifacts_ready():
    st.error(
        "Artifacts not found. Run training first and commit the 'artifacts/' folder.\n\n"
        "Steps:\n"
        "1) Create har_full.csv using scripts/make_har_csv.py\n"
        "2) Train: python -m model.train_models --csv har_full.csv\n"
        "3) Commit artifacts/ to GitHub and redeploy."
    )
    st.stop()

meta = load_json(os.path.join(ART_DIR, "meta.json"))
target_col = meta["target_col"]
feature_cols = meta["feature_cols"]

le = joblib.load(os.path.join(ART_DIR, "label_encoder.joblib"))
class_names = list(le.classes_)

metrics_df = pd.read_csv(os.path.join(ART_DIR, "metrics.csv"))

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    show_all_metrics = st.checkbox("Show metrics table for all models", value=True)

st.subheader("Precomputed Evaluation Metrics (from training run)")
if show_all_metrics:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.dataframe(metrics_df[metrics_df["ML Model Name"] == model_name], use_container_width=True)

model_path = os.path.join(ART_DIR, MODEL_FILES[model_name])
model = joblib.load(model_path)

st.subheader("Upload Test CSV (Optional labels supported)")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info(
        f"Expected feature columns count: {len(feature_cols)}. "
        f"If you include labels, add a column named '{target_col}'."
    )
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(
        f"Uploaded CSV is missing {len(missing)} required feature columns. "
        f"Example missing: {missing[:10]}"
    )
    st.stop()

X = df[feature_cols].copy()

has_labels = target_col in df.columns
y_true_enc = None
if has_labels:
    # Transform labels using same encoder
    y_true_raw = df[target_col].astype(str)
    unknown = sorted(set(y_true_raw.unique()) - set(class_names))
    if unknown:
        st.error(f"Unknown label(s) in uploaded file: {unknown}. Must be one of: {class_names}")
        st.stop()
    y_true_enc = le.transform(y_true_raw)

y_pred_enc = model.predict(X)
y_pred_label = le.inverse_transform(y_pred_enc.astype(int))

st.subheader("Predictions Preview")
out_df = df.copy()
out_df["PredictedActivity"] = y_pred_label
st.dataframe(out_df.head(30), use_container_width=True)

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Predictions CSV",
    data=csv_bytes,
    file_name="predictions.csv",
    mime="text/csv",
)

if has_labels:
    st.subheader("Uploaded Data Evaluation (Selected Model)")
    y_proba = safe_predict_proba(model, X)
    metrics = compute_metrics_multiclass(y_true_enc, y_pred_enc, y_proba, average="macro")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("F1 (macro)", f"{metrics['F1']:.4f}")
    col3.metric("MCC", f"{metrics['MCC']:.4f}")

    st.write("Full metrics:", metrics)

    st.subheader("Classification Report")
    report = make_classification_report(y_true_enc, y_pred_enc, target_names=class_names)
    st.text(report)

    st.subheader("Confusion Matrix")
    cm = make_confusion_matrix(y_true_enc, y_pred_enc)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)