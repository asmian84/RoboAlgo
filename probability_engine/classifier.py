"""
RoboAlgo - Probability Engine
Trains an XGBoost classifier to predict the probability of a significant
price increase within a forward-looking window.

Training label:
  success = price increases >= 8% within the next 20 trading days.

Input:  feature vectors from the features table.
Output: probability of success (0.0 - 1.0).
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sqlalchemy import select
from tqdm import tqdm

from config.settings import PROBABILITY_PARAMS
from database.connection import get_session
from database.models import Instrument, PriceData, Feature

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "trend_strength", "momentum", "volatility_percentile", "volume_ratio",
    "cycle_phase", "macd_norm", "bb_position", "price_to_ma50",
    "return_5d", "return_20d",
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ProbabilityClassifier:
    """XGBoost-based probability classifier for cycle-based trading signals."""

    def __init__(self):
        self.lookahead = PROBABILITY_PARAMS["lookahead_days"]
        self.target_return = PROBABILITY_PARAMS["target_return"]
        self.model = None

    def _build_labels(self, close_prices: pd.Series) -> pd.Series:
        """Build binary labels: 1 if max forward return >= target within lookahead window.

        Args:
            close_prices: Series of close prices indexed by date.

        Returns:
            Binary label series (1 = success, 0 = fail).
        """
        labels = pd.Series(index=close_prices.index, dtype=float)

        for i in range(len(close_prices) - self.lookahead):
            current_price = close_prices.iloc[i]
            future_window = close_prices.iloc[i + 1:i + 1 + self.lookahead]
            max_future = future_window.max()
            max_return = (max_future - current_price) / current_price
            labels.iloc[i] = 1.0 if max_return >= self.target_return else 0.0

        return labels

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load features and build labels for all instruments.

        Returns:
            (X, y) tuple of feature matrix and labels.
        """
        session = get_session()
        try:
            instruments = session.execute(select(Instrument)).scalars().all()

            all_X = []
            all_y = []

            for instrument in tqdm(instruments, desc="Loading training data"):
                # Load features
                features = pd.read_sql(
                    select(Feature.date, *[getattr(Feature, c) for c in FEATURE_COLS])
                    .where(Feature.instrument_id == instrument.id)
                    .order_by(Feature.date),
                    session.bind,
                )
                if features.empty or len(features) < 50:
                    continue
                features["date"] = pd.to_datetime(features["date"])
                features = features.set_index("date")

                # Load close prices for label generation
                prices = pd.read_sql(
                    select(PriceData.date, PriceData.close)
                    .where(PriceData.instrument_id == instrument.id)
                    .order_by(PriceData.date),
                    session.bind,
                )
                prices["date"] = pd.to_datetime(prices["date"])
                prices = prices.set_index("date")

                # Align features with prices and build labels
                aligned = features.join(prices[["close"]], how="inner")
                if len(aligned) < 50:
                    continue

                labels = self._build_labels(aligned["close"])

                # Drop rows without labels (tail of series)
                valid_mask = labels.notna()
                X = aligned[FEATURE_COLS][valid_mask]
                y = labels[valid_mask]

                # Drop rows with NaN features
                valid = X.notna().all(axis=1)
                X = X[valid]
                y = y[valid]

                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)

            if not all_X:
                return pd.DataFrame(), pd.Series(dtype=float)

            return pd.concat(all_X), pd.concat(all_y)
        finally:
            session.close()

    def train(self) -> dict:
        """Train the XGBoost classifier on all available data.

        Returns:
            Dict with training metrics (AUC, class distribution, etc.).
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed. Install with: pip install xgboost")
            raise

        X, y = self._load_training_data()
        if X.empty:
            logger.error("No training data available. Run feature generation first.")
            return {"error": "no data"}

        logger.info(f"Training data: {len(X)} samples, {y.sum():.0f} positive ({100*y.mean():.1f}%)")

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        auc_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores.append(auc)

        # Final model on all data
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X, y, verbose=False)

        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "probability_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        metrics = {
            "samples": len(X),
            "positive_rate": float(y.mean()),
            "cv_auc_mean": float(avg_auc),
            "cv_auc_scores": [float(a) for a in auc_scores],
            "model_path": model_path,
        }
        logger.info(f"Training complete. CV AUC: {avg_auc:.4f}")
        return metrics

    def load_model(self):
        """Load a previously trained model from disk."""
        model_path = os.path.join(MODEL_DIR, "probability_model.pkl")
        if not os.path.exists(model_path):
            logger.error(f"No model found at {model_path}. Run training first.")
            return False
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded successfully.")
        return True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict probability of success for each row in the feature matrix.

        Args:
            features: DataFrame with FEATURE_COLS columns.

        Returns:
            Series of probabilities indexed by date.
        """
        if self.model is None:
            if not self.load_model():
                return pd.Series(dtype=float)

        X = features[FEATURE_COLS].copy()

        # Handle missing values
        X = X.fillna(0.0)

        probas = self.model.predict_proba(X)[:, 1]
        return pd.Series(probas, index=features.index, name="probability")

    def predict_for_symbol(self, symbol: str) -> pd.Series:
        """Load features for a symbol and return predictions.

        Returns:
            Series of probabilities indexed by date.
        """
        session = get_session()
        try:
            instrument_id = session.execute(
                select(Instrument.id).where(Instrument.symbol == symbol)
            ).scalar()
            if instrument_id is None:
                return pd.Series(dtype=float)

            features = pd.read_sql(
                select(Feature.date, *[getattr(Feature, c) for c in FEATURE_COLS])
                .where(Feature.instrument_id == instrument_id)
                .order_by(Feature.date),
                session.bind,
            )
            if features.empty:
                return pd.Series(dtype=float)

            features["date"] = pd.to_datetime(features["date"])
            features = features.set_index("date")

            return self.predict(features)
        finally:
            session.close()
