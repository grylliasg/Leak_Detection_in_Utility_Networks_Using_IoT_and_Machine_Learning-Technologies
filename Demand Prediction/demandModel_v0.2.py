# Testing the model on a different network than the one it was trained on


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def create_lag_features(df, lags=5):
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        lagged = df.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
        df_lagged = pd.concat([df_lagged, lagged], axis=1)
    df_lagged.dropna(inplace=True)
    df_lagged.reset_index(drop=True, inplace=True)
    return df_lagged

# Φόρτωση δεδομένων
df_net3 = pd.read_csv("demand_no_leak(Net 3).csv")
df_net1 = pd.read_csv("demand_no_leak(Net 1).csv")

# Αφαίρεση index αν υπάρχει
if df_net3.columns[0] == 'Unnamed: 0':
    df_net3 = df_net3.drop(columns=df_net3.columns[0])
if df_net1.columns[0] == 'Unnamed: 0':
    df_net1 = df_net1.drop(columns=df_net1.columns[0])

# Βρες common κόμβους (στήλες)
common_nodes = df_net3.columns.intersection(df_net1.columns).tolist()
print("Κοινά Nodes:", common_nodes)

# Κράτα μόνο κοινά nodes
df_net3_common = df_net3[common_nodes]
df_net1_common = df_net1[common_nodes]

# Δημιουργία lag features στα κοινά κόμβους
lags = 5
df_net3_lagged = create_lag_features(df_net3_common, lags)
df_net1_lagged = create_lag_features(df_net1_common, lags)

# Χαρακτηριστικά και στόχοι για Net3
X_net3 = df_net3_lagged.drop(columns=common_nodes) # Για είσοδο μόνο τα lags
y_net3 = df_net3_lagged[common_nodes]

# Χαρακτηριστικά και στόχοι για Net1
X_net1 = df_net1_lagged.drop(columns=common_nodes)
y_net1 = df_net1_lagged[common_nodes]

# Διαχωρισμός Net3 σε train/test (χρονικά)
middle = len(X_net3) // 2
X_train, X_test = X_net3.iloc[:middle], X_net3.iloc[middle:]
y_train, y_test = y_net3.iloc[:middle], y_net3.iloc[middle:]

# Κανονικοποίηση
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Εκπαίδευση Linear Regression σε Net3
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Πρόβλεψη σε test set Net3
y_test_pred = model.predict(X_test_scaled)

# Αξιολόγηση στο test set Net3
mae_net3 = mean_absolute_error(y_test, y_test_pred)
r2_net3 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
print(f"Net3 Test MAE: {mae_net3:.4f}")
print(f"Net3 Test R2: {r2_net3:.4f}")

# --- Δοκιμή στο Net1 ---

# Κανονικοποίηση X_net1 με scaler του Net3
X_net1_scaled = scaler.transform(X_net1)

# Πρόβλεψη για Net1
y_net1_pred = model.predict(X_net1_scaled)

# Αξιολόγηση στο Net1
mae_net1 = mean_absolute_error(y_net1, y_net1_pred)
r2_net1 = r2_score(y_net1, y_net1_pred, multioutput='uniform_average')
print(f"Net1 MAE: {mae_net1:.4f}")
print(f"Net1 R2: {r2_net1:.4f}")

# Οπτικοποίηση για έναν κοινό κόμβο (π.χ. πρώτο κοινό)
node_col = common_nodes[0]
plt.figure(figsize=(10,6))
plt.plot(y_net1.index, y_net1[node_col], label="Actual Net1")
plt.plot(y_net1.index, y_net1_pred[:, 0], label="Predicted Net1", linestyle='--')
plt.title(f"Demand Prediction for {node_col} (Net1)")
plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.legend()
plt.show()
