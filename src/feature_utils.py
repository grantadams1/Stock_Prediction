"""feature_utils.py

Utilities for the Bitcoin/crypto Streamlit + SageMaker endpoint project.

This module is designed to be:
- Streamlit-friendly (no side effects at import time, optional Streamlit integration)
- Lightweight (no yfinance / pandas_datareader dependency)
- Clear about the request payload shape for SageMaker invocation

Typical usage in Streamlit:

    import streamlit as st
    from feature_utils import (
        validate_close_price,
        build_payload_single_feature,
        make_sagemaker_runtime_client_from_secrets,
        invoke_sagemaker_json_endpoint,
    )

    close_price = validate_close_price(st.number_input("Close price", value=0.0))
    payload = build_payload_single_feature(close_price)
    runtime = make_sagemaker_runtime_client_from_secrets(st.secrets)
    result = invoke_sagemaker_json_endpoint(runtime, st.secrets["AWS_ENDPOINT"], payload)

Notes:
- Many classroom SageMaker examples use temporary (session) credentials (ASIA... keys).
  Those require AWS_SESSION_TOKEN. If you omit it, you will see "security token is invalid".
- If your SageMaker inference script expects a different JSON shape, change the payload
  builder to match (see build_payload_* helpers below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import json
import math
import os

import boto3
import numpy as np
import pandas as pd
import requests


Number = Union[int, float, np.number]


# ---------------------------------------------------------------------
# Basic input validation
# ---------------------------------------------------------------------
def validate_close_price(x: Number) -> float:
    """Validate and coerce a close price input.

    Raises:
        ValueError: if x is NaN, infinite, or <= 0
    """
    val = float(x)
    if math.isnan(val) or math.isinf(val):
        raise ValueError("Close price must be a finite number.")
    if val <= 0:
        raise ValueError("Close price must be > 0.")
    return val


# ---------------------------------------------------------------------
# Feature engineering (simple, transparent, and easy to adapt)
# ---------------------------------------------------------------------
def compute_time_series_features(
    close: Union[pd.Series, Sequence[Number]],
    *,
    rsi_window: int = 14,
    sma_windows: Sequence[int] = (7, 14),
    vol_window: int = 7,
) -> pd.DataFrame:
    """Compute a small set of common time-series features from close prices.

    Returns a DataFrame with columns:
      - close
      - log_return_1
      - sma_{w} for each w in sma_windows
      - ema_{w} for each w in sma_windows
      - rsi_{rsi_window}
      - vol_{vol_window} (rolling std of log returns)

    This is intentionally "generic"; if your trained model expects different
    features, modify this function to match your training pipeline.
    """
    s = pd.Series(close, dtype="float64").rename("close")
    df = pd.DataFrame({"close": s})

    df["log_return_1"] = np.log(df["close"]).diff()

    for w in sma_windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{rsi_window}"] = 100 - (100 / (1 + rs))

    df[f"vol_{vol_window}"] = df["log_return_1"].rolling(vol_window).std()

    return df


def latest_feature_row_as_list(features_df: pd.DataFrame) -> List[float]:
    """Return the last row of a features DataFrame as a plain list of floats."""
    row = features_df.tail(1)
    if row.empty:
        raise ValueError("Not enough data to compute features.")
    return [float(x) for x in row.iloc[0].tolist()]


# ---------------------------------------------------------------------
# Payload builders for SageMaker (pick the one your inference script expects)
# ---------------------------------------------------------------------
def build_payload_single_feature(close_price: Number, *, key: str = "features") -> Dict[str, Any]:
    """Payload for the common case where your model expects ONE numeric feature.

    Default JSON produced:
        {"features": [12345.67]}

    If your inference handler expects a different key, pass key="...".
    """
    return {key: [float(close_price)]}


def build_payload_instances(feature_vector: Sequence[Number]) -> Dict[str, Any]:
    """TensorFlow-style shape: {"instances": [[...]]}."""
    return {"instances": [[float(x) for x in feature_vector]]}


def build_payload_records(feature_dict: Mapping[str, Number]) -> Dict[str, Any]:
    """Record-style shape: {"records": [{"f1": 1.0, ...}]}."""
    return {"records": [{k: float(v) for k, v in feature_dict.items()}]}


# ---------------------------------------------------------------------
# SageMaker Runtime helpers (works locally and on Streamlit Community Cloud)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AwsRuntimeConfig:
    region: str
    endpoint: str
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None


def load_aws_runtime_config_from_env(prefix: str = "AWS_") -> AwsRuntimeConfig:
    """Load AWS config from environment variables (useful for local runs)."""
    region = os.environ.get(f"{prefix}REGION") or os.environ.get("AWS_REGION")
    endpoint = os.environ.get(f"{prefix}ENDPOINT") or os.environ.get("AWS_ENDPOINT")
    access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN")

    missing = [k for k, v in {
        "region": region,
        "endpoint": endpoint,
        "AWS_ACCESS_KEY_ID": access_key_id,
        "AWS_SECRET_ACCESS_KEY": secret_access_key,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing required AWS config values: {missing}")

    return AwsRuntimeConfig(
        region=str(region),
        endpoint=str(endpoint),
        access_key_id=str(access_key_id),
        secret_access_key=str(secret_access_key),
        session_token=session_token,
    )


def load_aws_runtime_config_from_secrets(secrets: Mapping[str, Any]) -> AwsRuntimeConfig:
    """Load AWS config from Streamlit secrets mapping.

    Supports either:
      - flat keys: AWS_REGION, AWS_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
      - or nested table: secrets["aws_credentials"][...]
    """
    if "aws_credentials" in secrets:
        s = secrets["aws_credentials"]
        region = s.get("AWS_REGION") or secrets.get("AWS_REGION")
        endpoint = s.get("AWS_ENDPOINT") or secrets.get("AWS_ENDPOINT")
        access_key_id = s.get("AWS_ACCESS_KEY_ID")
        secret_access_key = s.get("AWS_SECRET_ACCESS_KEY")
        session_token = s.get("AWS_SESSION_TOKEN")
    else:
        region = secrets.get("AWS_REGION")
        endpoint = secrets.get("AWS_ENDPOINT")
        access_key_id = secrets.get("AWS_ACCESS_KEY_ID")
        secret_access_key = secrets.get("AWS_SECRET_ACCESS_KEY")
        session_token = secrets.get("AWS_SESSION_TOKEN")

    missing = [k for k, v in {
        "AWS_REGION": region,
        "AWS_ENDPOINT": endpoint,
        "AWS_ACCESS_KEY_ID": access_key_id,
        "AWS_SECRET_ACCESS_KEY": secret_access_key,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing required Streamlit secrets: {missing}")

    return AwsRuntimeConfig(
        region=str(region),
        endpoint=str(endpoint),
        access_key_id=str(access_key_id),
        secret_access_key=str(secret_access_key),
        session_token=str(session_token) if session_token else None,
    )


def make_sagemaker_runtime_client(config: AwsRuntimeConfig):
    """Create a boto3 sagemaker-runtime client."""
    return boto3.client(
        "sagemaker-runtime",
        region_name=config.region,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        aws_session_token=config.session_token,  # REQUIRED for session (ASIA...) creds
    )


def make_sagemaker_runtime_client_from_secrets(secrets: Mapping[str, Any]):
    """Convenience wrapper for Streamlit."""
    cfg = load_aws_runtime_config_from_secrets(secrets)
    return make_sagemaker_runtime_client(cfg)


def invoke_sagemaker_json_endpoint(
    runtime_client,
    endpoint_name: str,
    payload: Dict[str, Any],
    *,
    accept: str = "application/json",
    content_type: str = "application/json",
    timeout_seconds: Optional[int] = None,
) -> Any:
    """Invoke a SageMaker endpoint with JSON payload and parse JSON response.

    If your model returns plain text, change the response parsing accordingly.
    """
    body = json.dumps(payload).encode("utf-8")

    # boto3 doesn't expose per-request timeout cleanly; if you need it, configure botocore.
    resp = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=body,
    )
    raw = resp["Body"].read()
    try:
        return json.loads(raw)
    except Exception:
        # Fall back to returning raw text for easier debugging
        return {"raw": raw.decode("utf-8", errors="replace")}


# ---------------------------------------------------------------------
# Optional: fetch Bitcoin prices (useful if you want to auto-fill the input)
# ---------------------------------------------------------------------
def fetch_bitcoin_price_usd_coingecko(*, timeout: int = 10) -> float:
    """Fetch current BTC price (USD) using CoinGecko public API."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return float(data["bitcoin"]["usd"])
