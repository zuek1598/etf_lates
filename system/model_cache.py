"""ML Model Caching System - Avoid retraining on identical data"""

import joblib
from pathlib import Path
from datetime import datetime
import hashlib
import pandas as pd


class ModelCache:
    """Cache trained ML models to avoid retraining on unchanged data"""

    def __init__(self, cache_dir='cache/models'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _get_data_hash(self, prices: pd.Series) -> str:
        """Create unique hash from price data (last date + length)"""
        try:
            data_str = str(prices.index[-1]) + str(len(prices))
            return hashlib.md5(data_str.encode()).hexdigest()[:8]
        except:
            return "invalid"

    def get_cached_model(self, ticker: str, prices: pd.Series) -> dict:
        """Load cached model if data unchanged and fresh"""
        data_hash = self._get_data_hash(prices)
        if data_hash == "invalid":
            return None

        cache_file = self.cache_dir / f"{ticker}_{data_hash}.joblib"

        if not cache_file.exists():
            return None

        try:
            model_data = joblib.load(cache_file)
            # Verify cache is fresh (< 1 day old)
            cache_age = (datetime.now() - model_data['cached_at']).days
            if cache_age < 1:
                return model_data
        except:
            return None

        return None

    def save_model(self, ticker: str, prices: pd.Series, model_data: dict) -> None:
        """Save trained model with metadata"""
        data_hash = self._get_data_hash(prices)
        if data_hash == "invalid":
            return

        cache_file = self.cache_dir / f"{ticker}_{data_hash}.joblib"

        model_data.update({
            'ticker': ticker,
            'data_hash': data_hash,
            'cached_at': datetime.now(),
            'data_length': len(prices),
            'last_date': str(prices.index[-1])
        })

        try:
            joblib.dump(model_data, cache_file)
        except:
            pass  # Silent fail - caching not critical
