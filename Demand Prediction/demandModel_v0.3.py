import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# --- Συνάρτηση δημιουργίας lag features ---
def create_lag_features(df, lags=5):
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        lagged = df.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
        df_lagged = pd.concat([df_lagged, lagged], axis=1)
    df_lagged.dropna(inplace=True)
    df_lagged.reset_index(drop=True, inplace=True)
    return df_lagged

# --- Φόρτωση δεδομένων ---
df_net3 = pd.read_csv("demand_no_leak(Net 3).csv")
df_net1 = pd.read_csv("demand_no_leak(Net 1).csv")

# --- Καθαρισμός index ---
if df_net3.columns[0] == 'Unnamed: 0':
    df_net3 = df_net3.drop(columns=df_net3.columns[0])
if df_net1.columns[0] == 'Unnamed: 0':
    df_net1 = df_net1.drop(columns=df_net1.columns[0])

# --- Κοινοί κόμβοι ---
common_nodes = df_net3.columns.intersection(df_net1.columns).tolist()
print("Κοινά Nodes:", common_nodes)

# --- Επιλογή κοινών κόμβων ---
df_net3_common = df_net3[common_nodes]
df_net1_common = df_net1[common_nodes]

# --- Lag features ---
lags = 5
df_net3_lagged = create_lag_features(df_net3_common, lags)
df_net1_lagged = create_lag_features(df_net1_common, lags)

# --- Είσοδοι / στόχοι ---
X_net3 = df_net3_lagged.drop(columns=common_nodes)
y_net3 = df_net3_lagged[common_nodes]

# --- TimeSeriesSplit Cross-Validation ---
tscv = TimeSeriesSplit(n_splits=5)
mae_scores = []
r2_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X_net3)):
    X_train, X_test = X_net3.iloc[train_index], X_net3.iloc[test_index]
    y_train, y_test = y_net3.iloc[train_index], y_net3.iloc[test_index]

    # Κανονικοποίηση
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Εκπαίδευση
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Πρόβλεψη
    y_pred = model.predict(X_test_scaled)

    # Αξιολόγηση
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print(f"Fold {fold+1} - MAE: {mae:.4f}, R2: {r2:.4f}")
    mae_scores.append(mae)
    r2_scores.append(r2)

# --- Μέσοι Όροι ---
print(f"\nAverage MAE: {np.mean(mae_scores):.4f}")
print(f"Average R2: {np.mean(r2_scores):.4f}")
