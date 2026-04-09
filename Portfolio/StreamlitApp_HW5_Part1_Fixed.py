"""
HW5 – Part 1 Streamlit App: IBM Stock Return Predictor
Uses the KernelPCA + Lasso pipeline trained in the HW5 notebook.

Flow:
  1. User enters the current IBM stock price.
  2. The app finds the closest historical date with that price.
  3. SP500 cumulative-return features are computed for that date.
  4. Features are sent to the SageMaker endpoint.
  5. The predicted 5-day cumulative future return is displayed.
  6. A SHAP waterfall plot explains the prediction.
"""

import os
import sys
import warnings
import posixpath
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import tarfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline
import shap

warnings.simplefilter("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── AWS credentials (from Streamlit secrets) ─────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── Model artefact configuration ─────────────────────────────────────────────
MODEL_INFO = {
    "endpoint":  aws_endpoint,
    "explainer": "explainer_pca.shap",
    "pipeline":  "finalized_pca_model.tar.gz",
    "target":    "IBM",          # target stock whose return we predict
    "inputs": [
        {
            "name":    "IBM",
            "type":    "number",
            "min":     0.0,
            "default": 100.0,
            "step":    10.0,
        }
    ],
}

RETURN_PERIOD = 5   # days used to compute cumulative returns

# ── AWS session ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )


session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Helper: load SP500 data ──────────────────────────────────────────────────
@st.cache_data
def load_sp500_dataset():
    return pd.read_csv("Portfolio/SP500Data.csv", index_col=0)


# ── Helper: feature engineering ──────────────────────────────────────────────
def build_features(dataset: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return the cumulative-return feature matrix (target stock excluded)."""
    X = np.log(dataset.drop([target], axis=1)).diff(RETURN_PERIOD)
    X = np.exp(X).cumsum()
    X.columns = [col + "_CR_Cum" for col in X.columns]
    return X


# ── Helper: load pipeline from S3 ────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket, s3_key_prefix):
    s3 = _session.client("s3")
    tar_name = MODEL_INFO["pipeline"]
    local_tar = os.path.join(tempfile.gettempdir(), tar_name)

    s3.download_file(Bucket=bucket, Key=f"{s3_key_prefix}/{tar_name}", Filename=local_tar)

    extract_dir = os.path.join(tempfile.gettempdir(), "model_extract")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_name = next(f for f in tar.getnames() if f.endswith(".joblib"))

    return joblib.load(os.path.join(extract_dir, joblib_name))


# ── Helper: load SHAP explainer from S3 ──────────────────────────────────────
@st.cache_resource
def load_shap_explainer(_session, bucket, s3_key, local_path):
    s3 = _session.client("s3")
    if not os.path.exists(local_path):
        s3.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# ── Prediction via SageMaker endpoint ────────────────────────────────────────
def call_model_api(feature_array: np.ndarray):
    """
    Parameters
    ----------
    feature_array : np.ndarray of shape (1, n_features)
        Pre-computed cumulative-return features for one date.

    Returns
    -------
    (predicted_value, status_code)
    """
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw_pred  = predictor.predict(feature_array)
        pred_val  = float(np.array(raw_pred).flatten()[0])
        return round(pred_val, 4), 200
    except Exception as exc:
        return f"Error: {exc}", 500


# ── SHAP explanation ──────────────────────────────────────────────────────────
def display_explanation(input_row: pd.DataFrame):
    """
    Parameters
    ----------
    input_row : pd.DataFrame with exactly one row of cumulative-return features.
    """
    with st.spinner("Loading explainer from S3 …"):
        explainer_key   = posixpath.join("explainer", MODEL_INFO["explainer"])
        local_explainer = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])
        explainer = load_shap_explainer(
            session, aws_bucket, explainer_key, local_explainer
        )

    with st.spinner("Loading pipeline from S3 …"):
        best_pipeline = load_pipeline(session, aws_bucket, "sklearn-pipeline-deployment")

    # Build preprocessing pipeline: all steps EXCEPT the final model
    preprocessing_steps = best_pipeline.steps[:-1]  # imputer + scaler + kpca
    preprocessing_pipeline = Pipeline(steps=preprocessing_steps)

    # Transform the single input row
    input_transformed = preprocessing_pipeline.transform(input_row)

    # KPCA component names
    n_comp = best_pipeline.named_steps["kpca"].n_components
    feature_names = [f"kernelpca{i}" for i in range(n_comp)]
    input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names)

    # SHAP values
    shap_values = explainer(input_transformed_df)

    st.subheader("🔍 Decision Transparency (SHAP Waterfall)")
    fig, _ = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # Top contributing component
    top_idx     = int(np.abs(shap_values[0].values).argmax())
    top_feature = shap_values[0].feature_names[top_idx]
    st.info(
        f"**Business Insight:** The most influential factor in this prediction "
        f"was **{top_feature}** (a KernelPCA latent component)."
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IBM Return Predictor", layout="wide")
st.title("📈 IBM Stock Return Predictor")
st.markdown(
    "Enter the current **IBM stock price**. "
    "The model will find the most similar historical market environment "
    "and predict IBM's **5-day cumulative future return**."
)

with st.form("pred_form"):
    st.subheader("Input")
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].upper(),
                min_value=inp["min"],
                value=inp["default"],
                step=inp["step"],
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    target      = MODEL_INFO["target"]
    ibm_price   = float(user_inputs[target])

    # ── Feature engineering ──────────────────────────────────────────────────
    with st.spinner("Building features from SP500 history …"):
        dataset      = load_sp500_dataset()
        X_features   = build_features(dataset, target)

        # Find the historical date whose IBM price is closest to the input
        closest_date = (dataset[target] - ibm_price).abs().idxmin()
        st.caption(f"Closest historical date: **{closest_date}** (IBM ≈ {dataset.loc[closest_date, target]:.2f})")

        # Single-row feature vector for that date
        input_row    = X_features.loc[[closest_date]]   # shape (1, n_features)

    # ── API call ─────────────────────────────────────────────────────────────
    with st.spinner("Calling SageMaker endpoint …"):
        result, status = call_model_api(input_row.values.astype(np.float32))

    if status == 200:
        st.metric(
            label=f"Predicted 5-day Cumulative Return for {target}",
            value=result,
        )
        display_explanation(input_row)
    else:
        st.error(result)
