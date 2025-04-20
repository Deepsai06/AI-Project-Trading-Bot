import pandas as pd
import numpy as np

exogenous_features =  ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", "Low_std_lag3",
                      "Volume_mean_lag3", "Volume_std_lag3", "High_mean_lag7", "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",
                      "Volume_mean_lag7", "Volume_std_lag7", "High_mean_lag30", "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",
                      "Volume_mean_lag30", "Volume_std_lag30"]

def add_features(df):
    # df.reset_index(drop=True, inplace=True)
    lag_features = ["High", "Low", "Volume"]
    window1 = 3
    window2 = 7
    window3 = 30

    df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
    df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
    df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

    df_mean_3d = df_rolled_3d.mean()
    df_mean_7d = df_rolled_7d.mean()
    df_mean_30d = df_rolled_30d.mean()

    df_std_3d = df_rolled_3d.std()
    df_std_7d = df_rolled_7d.std()
    df_std_30d = df_rolled_30d.std()

    for feature in lag_features:
        df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
        df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
        df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
        
        df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
        df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
        df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]
    return df
