"""
Custom_Classes.py — all transformer classes used in the fraud detection pipeline.
Bundled in finalized_fraud_model.tar.gz so SageMaker can deserialize the pipeline.
Streamlit also imports from here via src/ for the SHAP display_explanation() function.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropHighMissing(BaseEstimator, TransformerMixin):
    """CLEANING STEP 1: Drop cols where NaN rate > threshold (default 50%)."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.drop_cols_ = X.columns[X.isnull().mean() > self.threshold].tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.drop_cols_, errors='ignore')
        return X


class DropLowVariance(BaseEstimator, TransformerMixin):
    """CLEANING STEP 2: Drop numeric cols where dominant value > dominance_threshold."""
    def __init__(self, dominance_threshold=0.95):
        self.dominance_threshold = dominance_threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            drop = []
            for col in X.select_dtypes(include=[np.number]).columns:
                top_freq = X[col].value_counts(normalize=True).iloc[0] if X[col].nunique() > 0 else 1.0
                if top_freq >= self.dominance_threshold:
                    drop.append(col)
            self.drop_cols_ = drop
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.drop_cols_, errors='ignore')
        return X


class DropHighCardinality(BaseEstimator, TransformerMixin):
    """
    CLEANING STEP 3: Drop object cols with > cardinality_threshold unique values.
    Adversarial validation: high-cardinality string cols cause train-test data drift
    because unseen categories at inference time cannot be encoded reliably.
    """
    def __init__(self, cardinality_threshold=100):
        self.cardinality_threshold = cardinality_threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.drop_cols_ = [
                col for col in X.select_dtypes(include=['object']).columns
                if X[col].nunique() > self.cardinality_threshold
            ]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.drop_cols_, errors='ignore')
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    CLEANING STEP 4 + 10-step feature engineering:
    Step 1.  Temporal recoding/recode: TransactionDT -> tx_hour, tx_dayofwk, tx_day, tx_week
    Step 2.  Log-transform TransactionAmt (scale transformation, skew reduction)
    Step 3.  Frequency encode high-cardinality categoricals (P/R_emaildomain, DeviceInfo)
    Step 4.  Interaction feature: card6_credit_flag x TransactionAmt (composite weighted)
    Step 5.  Aggregation: mean & std of TransactionAmt per card1 cluster bucket
    Step 6.  Ratio feature: TransactionAmt / card1_amt_mean (debt-to-mean ratio)
    Step 7.  Binary flag: P_emaildomain == R_emaildomain (email_match)
    Step 8.  M-column recoding: T->1, F->0, NaN->-1
    Step 9.  Ordinal encoding: card4, card6, ProductCD -> integer codes
    Step 10. Median imputation of remaining NaN values
    Note: VIF collinearity removal handled downstream by DropHighCorrelation (r>0.95).
    """
    def __init__(self):
        self.card1_stats_  = {}
        self.freq_maps_    = {}
        self.ordinal_maps_ = {}

    def _m_encode(self, df):
        for c in [col for col in df.columns if col.startswith('M')]:
            df[c] = df[c].map({'T': 1, 'F': 0}).fillna(-1)
        return df

    def _freq_encode(self, df, col):
        return df[col].map(self.freq_maps_.get(col, {})).fillna(0)

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'card1' in df.columns and 'TransactionAmt' in df.columns:
            grp = df.groupby('card1')['TransactionAmt']
            self.card1_stats_['mean'] = grp.mean().to_dict()
            self.card1_stats_['std']  = grp.std().fillna(0).to_dict()
        for col in ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'id_30', 'id_31']:
            if col in df.columns:
                self.freq_maps_[col] = df[col].value_counts(normalize=True).to_dict()
        for col in ['card4', 'card6', 'ProductCD']:
            if col in df.columns:
                self.ordinal_maps_[col] = {v: i for i, v in enumerate(df[col].dropna().unique())}
        # Store training medians for imputation (Step 10)
        # Must be computed on raw X BEFORE encoding so we capture all numeric cols.
        # We recompute after a dry-run transform on a copy to get post-FE medians.
        try:
            X_dry = self.transform(df.copy())
            self.medians_ = X_dry.median(numeric_only=True).to_dict()
        except Exception:
            self.medians_ = {}
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Step 1
        if 'TransactionDT' in df.columns:
            START = pd.Timestamp('2017-12-01')
            dt = pd.to_datetime(df['TransactionDT'], unit='s', origin=START)
            df['tx_hour']    = dt.dt.hour
            df['tx_dayofwk'] = dt.dt.dayofweek
            df['tx_day']     = (df['TransactionDT'] // 86400).astype(int)
            df['tx_week']    = (df['TransactionDT'] // (86400 * 7)).astype(int)
            df.drop(columns=['TransactionDT'], inplace=True)
        # Step 2
        if 'TransactionAmt' in df.columns:
            df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
        # Step 3
        for col in ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'id_30', 'id_31']:
            if col in df.columns:
                df[f'{col}_freq'] = self._freq_encode(df, col)
                df.drop(columns=[col], inplace=True)
        # Step 4
        if 'card6' in df.columns and 'TransactionAmt' in df.columns:
            card6_bin = df['card6'].map({'credit': 1, 'debit': 0}).fillna(0)
            df['card6_freq_enc'] = card6_bin
            df['card6_x_amt']    = card6_bin * df['TransactionAmt']
        # Step 5 & 6
        if 'card1' in df.columns and 'TransactionAmt' in df.columns:
            df['card1_amt_mean'] = df['card1'].map(self.card1_stats_.get('mean', {})).fillna(0)
            df['card1_amt_std']  = df['card1'].map(self.card1_stats_.get('std',  {})).fillna(0)
            df['amt_over_mean']  = df['TransactionAmt'] / (df['card1_amt_mean'] + 1e-9)
        # Step 7
        if 'P_emaildomain' in X.columns and 'R_emaildomain' in X.columns:
            df['email_match'] = (X['P_emaildomain'].fillna('') == X['R_emaildomain'].fillna('')).astype(int)
        # Step 8
        df = self._m_encode(df)
        # Step 9
        for col in ['card4', 'card6', 'ProductCD']:
            if col in df.columns:
                df[col] = df[col].map(self.ordinal_maps_.get(col, {})).fillna(-1)
        # Drop remaining object cols & TransactionID
        for col in df.select_dtypes(include=['object']).columns:
            df.drop(columns=[col], inplace=True)
        df.drop(columns=['TransactionID'], inplace=True, errors='ignore')
        # Step 10 – median imputation using training medians (safe for single-row inference)
        if hasattr(self, 'medians_') and self.medians_:
            df = df.fillna(value={k: v for k, v in self.medians_.items() if k in df.columns})
        else:
            df = df.fillna(df.median(numeric_only=True))
        return df

    def get_feature_names_out(self, input_features=None):
        return None


class DropHighCorrelation(BaseEstimator, TransformerMixin):
    """
    CLEANING STEP 5: Drop one col from each pair with |Pearson r| > threshold.
    Approximates VIF (Variance Inflation Factor) collinearity removal.
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr  = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            self.drop_cols_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.drop_cols_, errors='ignore')
        return X
