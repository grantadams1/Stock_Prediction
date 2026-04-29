import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump, load


# ── Setup & Path Configuration ───────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir   # on Streamlit Cloud, __file__ IS the repo root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ALL custom transformer classes so joblib can deserialize the pipeline.
# Try src/ subfolder first (local/SageMaker), fall back to root (Streamlit Cloud).
try:
    from src.Custom_Classes import (
        FeatureEngineer, DropHighMissing, DropLowVariance,
        DropHighCardinality, DropHighCorrelation,
    )
except ModuleNotFoundError:
    from Custom_Classes import (
        FeatureEngineer, DropHighMissing, DropLowVariance,
        DropHighCardinality, DropHighCorrelation,
    )

# Load the training-feature sample (used for baseline row + feature names)
# Works whether X_train.csv is in Portfolio/ or at the repo root
_portfolio_path = os.path.join(current_dir, 'Portfolio', 'X_train.csv')
_root_path      = os.path.join(current_dir, 'X_train.csv')
file_path = _portfolio_path if os.path.exists(_portfolio_path) else _root_path
dataset = pd.read_csv(file_path)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# ── AWS / Secrets ────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session Management ────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_sentiment.shap",
    "pipeline"  : "finalized_fraud_model.tar.gz",
    "keys"      : ['TransactionAmt', 'card3', 'C12', 'C1'],
    "inputs"    : [
        {"name": "TransactionAmt", "type": "number", "min": 0.0,   "max": 5000.0, "default": 100.0, "step": 1.0},
        {"name": "card3",          "type": "number", "min": 100.0, "max": 200.0,  "default": 150.0, "step": 1.0},
        {"name": "C12",            "type": "number", "min": 0.0,   "max": 10.0,   "default": 0.0,   "step": 0.1},
        {"name": "C1",             "type": "number", "min": 0.0,   "max": 20.0,   "default": 1.0,   "step": 0.5},
    ]
}


# ── S3 Loaders ────────────────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}")

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)


# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    # Teacher tip: JSONSerializer only — keeps column names, no deserializer
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
    )
    try:
        raw = predictor.predict(input_df)
        # Decode raw bytes → JSON list
        import json as _json
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode('utf-8')
        pred_list = _json.loads(raw) if isinstance(raw, str) else raw
        pred_val  = pred_list[-1] if isinstance(pred_list, list) else pred_list
        mapping   = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(int(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── Local Explainability ──────────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    input_df = pd.DataFrame([input_df]) if isinstance(input_df, dict) else pd.DataFrame(input_df)

    # SMOTE is train-only — exclude it and the classifier for inference preprocessing
    inference_steps = [(n, s) for n, s in best_pipeline.steps
                       if n not in ('smote', 'clf')]
    preprocessing_pipeline = Pipeline(steps=inference_steps)

    # Recover feature names that come out of FeatureEngineer (step name: 'feature_engineer')
    fe = best_pipeline.named_steps['feature_engineer']
    X_fe = fe.transform(input_df)
    feature_names_post_fe = list(X_fe.columns)

    input_df_transformed = preprocessing_pipeline.transform(input_df)

    selector = best_pipeline.named_steps['selector']
    selected_features = [feature_names_post_fe[i]
                         for i, v in enumerate(selector.get_support()) if v]
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=selected_features)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])   # class 1 = Fraud
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 1].values,
                  index=shap_values[0, :, 1].feature_names)
        .abs().idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection – ML Deployment", layout="wide")
st.title("👨‍💻 Fraud Detection – ML Deployment")
st.markdown(
    "Enter transaction details below. The model will classify the transaction "
    "as **Legitimate** or **Fraud** and explain the key drivers."
)

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step'])
            )

    submitted = st.form_submit_button("Run Prediction")

# Build full input row: baseline from first training row, override with user inputs
original = dataset.iloc[0:1].to_dict(orient='records')[0]
original.update(user_inputs)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        color = "🔴" if res == "Fraud" else "🟢"
        st.metric("Prediction Result", f"{color} {res}")
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
