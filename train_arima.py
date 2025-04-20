import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima.arima import ndiffs
import yfinance as yf
import joblib
import warnings
from feature_engineering import add_features
from feature_engineering import exogenous_features
warnings.filterwarnings("ignore", category=FutureWarning)

symbols = {
    "GOOG": pd.read_csv('./Stock Data/GOOG.txt'),
    "ACGL": pd.read_csv('./Stock Data/ACGL.txt'),
    "NVDA": pd.read_csv('./Stock Data/NVDA.txt'),
    "TSLA": pd.read_csv('./Stock Data/TSLA.txt'),
    "MCHP": pd.read_csv('./Stock Data/MCHP.txt'),
    "IEX" : pd.read_csv('./Stock Data/IEX.txt'),
    "AZN" : pd.read_csv('./Stock Data/AZN.txt'),
    "QCOM": pd.read_csv('./Stock Data/QCOM.txt')
}

for symbol, df in symbols.items():
    print(f"\nTraining model for {symbol}...")

    df = add_features(df)
    train_len = int(df.shape[0] * 0.8)
    train_data, test_data = df[:train_len], df[train_len:]

    y_train = train_data['Open'].values
    y_test = test_data['Open'].values
    
    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    print(f"Estimated differencing (d): {n_diffs}")

    model = pm.auto_arima(
        y_train,
        d=n_diffs,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=6,
        trace=True,
        exogenous = train_data[exogenous_features]

    )

    model_filename = f"./Models/arima_model_{symbol.lower()}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model for {symbol} saved as {model_filename}")
