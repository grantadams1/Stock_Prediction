import os, sys, warnings, json, tempfile, tarfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import boto3
from imblearn.pipeline import Pipeline
import shap

# --------------------------------------------------------------------------------------
# Streamlit + SageMaker Endpoint Inference (Streamlit Cloud-friendly)
# - Uses boto3 "sagemaker-runtime" with explicit credentials from st.secrets
# - Supports temporary session tokens (ASIA keys)
# - Downloads pipeline + SHAP explainer from S3 into /tmp
# --------------------------------------------------------------------------------------

warnings.simplefilter("ignore")

# Ensure project root is importable (for src.feature_utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

# -----------------------------
# Secrets (Streamlit Cloud)
# Put these under [aws_credentials] in Streamlit "Secrets"
# -----------------------------
creds = st.secrets["aws_credentials"]

AWS_ACCESS_KEY_ID = creds["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = creds["AWS_SECRET_ACCESS_KEY"]
AWS_SESSION_TOKEN = creds.get("AWS_SESSION_TOKEN")  # REQUIRED for ASIA keys
AWS_REGION = creds.get("AWS_REGION", "us-east-1")
AWS_BUCKET = creds["AWS_BUCKET"]
AWS_ENDPOINT = creds["AWS_ENDPOINT"]

# -----------------------------
# Cached AWS session + clients
# -----------------------------
@st.cache_resource
def get_boto_session():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_REGION,
    )

@st.cache_resource
def get_s3_client():
    return get_boto_session().client("s3")

@st.cache_resource
def get_runtime_client():
    return get_boto_session().client("sagemaker-runtime")

# -----------------------------
# Data + Model Info
# -----------------------------
df_prices = get_bitcoin_historical_prices()

MIN_VAL = float(0.5 * df_prices.iloc[:, 0].min())
MAX_VAL = float(2.0 * df_prices.iloc[:, 0].max())
DEFAULT_VAL = float(df_prices.iloc[:, 0].mean())

MODEL_INFO = {
    "endpoint": AWS_ENDPOINT,
    "explainer_key": "explainer/explainer_bitcoin.shap",
    "pipeline_key": "sklearn-pipeline-deployment/finalized_bitcoin_model.tar.gz",
    "keys": ["Close Price"],
    "inputs": [
        {
            "name": "Close Price",
            "min": MIN_VAL,
            "max": MAX_VAL,
            "default": DEFAULT_VAL,
            "step": 100.0,
        }
    ],
}

# -----------------------------
# S3 download helpers
# -----------------------------
def download_s3_to_tmp(bucket: str, key: str) -> str:
    """Download an S3 object to /tmp and return local filepath."""
    s3 = get_s3_client()
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    if not os.path.exists(local_path):
        s3.download_file(bucket, key, local_path)
    return local_path

@st.cache_resource
def load_pipeline_from_s3() -> Pipeline:
    tar_path = download_s3_to_tmp(AWS_BUCKET, MODEL_INFO["pipeline_key"])

    extract_dir = os.path.join(tempfile.gettempdir(), "sklearn_pipeline_extract")
    os.makedirs(extract_dir, exist_ok=True)

    joblib_path = None
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        for m in tar.getmembers():
            if m.name.endswith(".joblib"):
                joblib_path = os.path.join(extract_dir, os.path.basename(m.name))
                break

    if not joblib_path or not os.path.exists(joblib_path):
        raise FileNotFoundError("No .joblib file found inside the model tar.gz.")

    return joblib.load(joblib_path)

@st.cache_resource
def load_shap_explainer_from_s3():
    explainer_path = download_s3_to_tmp(AWS_BUCKET, MODEL_INFO["explainer_key"])
    with open(explainer_path, "rb") as f:
        return shap.Explainer.load(f)

# -----------------------------
# Endpoint invocation (CSV)
# -----------------------------
def invoke_endpoint_csv(one_row_df: pd.DataFrame):
    """
    Invoke SageMaker endpoint using CSV.
    (This matches the default SKLearn container expectation unless your inference.py overrides it.)
    """
    runtime = get_runtime_client()
    csv_body = one_row_df.to_csv(header=False, index=False)

    resp = runtime.invoke_endpoint(
        EndpointName=MODEL_INFO["endpoint"],
        ContentType="text/csv",
        Body=csv_body.encode("utf-8"),
    )

    raw = resp["Body"].read()
    text = raw.decode("utf-8").strip()

    try:
        return json.loads(text)
    except Exception:
        return text

def parse_prediction(pred):
    if isinstance(pred, dict):
        for k in ("prediction", "predictions", "result", "outputs"):
            if k in pred:
                pred = pred[k]
                break

    if isinstance(pred, (list, tuple, np.ndarray)):
        pred = pred[-1]

    # numeric string -> int/float
    try:
        if isinstance(pred, str):
            s = pred.strip()
            if s.replace(".", "", 1).replace("-", "", 1).isdigit():
                pred = float(s) if "." in s else int(s)
    except Exception:
        pass

    mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    if isinstance(pred, (int, np.integer)) and int(pred) in mapping:
        return mapping[int(pred)]
    return pred

def call_model_api(input_df: pd.DataFrame):
    try:
        last_row = input_df.tail(1)
        raw_pred = invoke_endpoint_csv(last_row)
        return parse_prediction(raw_pred), 200
    except Exception as e:
        return f"Error invoking endpoint: {e}", 500

# -----------------------------
# Explainability (SHAP)
# -----------------------------
def display_explanation(one_row_df: pd.DataFrame):
    explainer = load_shap_explainer_from_s3()
    full_pipeline = load_pipeline_from_s3()

    # Adjust if your pipeline structure differs
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])

    x = preprocessing_pipeline.transform(one_row_df)
    shap_values = explainer(x)

    try:
        feature_names = full_pipeline[1:4].get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(x.shape[1])]

    exp = shap.Explanation(
        values=shap_values[0, :, 0],
        base_values=getattr(explainer, "expected_value", [0])[0],
        data=x[0],
        feature_names=feature_names,
    )

    st.subheader("🔍 Decision Transparency (SHAP)")
    plt.figure(figsize=(10, 4))
    shap.plots.waterfall(exp, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    top_feature = pd.Series(exp.values, index=exp.feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ML Deployment Compiler", layout="wide")
st.title("👨‍💻 ML Deployment Compiler")

with st.form("pred_form"):
    st.subheader("Inputs")
    user_inputs = {}
    for inp in MODEL_INFO["inputs"]:
        user_inputs[inp["name"]] = st.number_input(
            inp["name"].upper(),
            min_value=float(inp["min"]),
            max_value=float(inp["max"]),
            value=float(inp["default"]),
            step=float(inp["step"]),
        )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    # Keep your original logic of appending to df_prices,
    # but send ONLY the user row to the endpoint.
    base_df = df_prices
    full_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)], ignore_index=True)

    pred, status = call_model_api(full_df.tail(1))
    if status == 200:
        st.metric("Prediction Result", pred)
        display_explanation(full_df.tail(1))
    else:
        st.error(pred)
