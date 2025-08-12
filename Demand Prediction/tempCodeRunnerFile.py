import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def read_junctions_from_inp(filepath):
    junctions = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    inside_junctions = False
    for line in lines:
        line = line.strip()
        if line.upper() == '[JUNCTIONS]':
            inside_junctions = True
            continue
        if inside_junctions:
            if line == '' or line.startswith('['):  # Τέλος ενότητας
                break
            # Η πρώτη λέξη κάθε γραμμής είναι το όνομα junction (αριθμός)
            junction_id = line.split()[0]
            junctions.append(junction_id)
    return junctions

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
df_net3 = pd.read_csv("Data/Normal flow/demand_no_leak(Net 3).csv")
df_net1 = pd.read_csv("Data/Normal flow/demand_no_leak(Net 1).csv")

# Αφαίρεση index αν υπάρχει
if df_net3.columns[0] == 'Unnamed: 0':
    df_net3 = df_net3.drop(columns=df_net3.columns[0])
if df_net1.columns[0] == 'Unnamed: 0':
    df_net1 = df_net1.drop(columns=df_net1.columns[0])

# Διάβασε junction nodes από τα αρχεία .inp
junctions_net3 = read_junctions_from_inp('Data/Net3.inp')
junctions_net1 = read_junctions_from_inp('Data/Net1.inp')

# Κοινά junction nodes (που υπάρχουν και στα δύο)
common_junctions = list(set(junctions_net3).intersection(set(junctions_net1)))
print(f"Κοινά Junctions (πριν φιλτράρισμα με βάση στήλες): {len(common_junctions)}")

# Φιλτράρισμα για να κρατήσουμε μόνο junctions που υπάρχουν στα dataframes
common_junctions = [node for node in common_junctions if node in df_net3.columns and node in df_net1.columns]
print(f"Κοινά Junctions (μετά φιλτράρισμα με βάση στήλες): {len(common_junctions)}")
print("Κοινά Junction Nodes:", common_junctions)

# Κράτα μόνο κοινά junction nodes στα dataframes
df_net3_common = df_net3[common_junctions]
df_net1_common = df_net1[common_junctions]

# Δημιουργία lag features στα κοινά junction nodes
lags = 5
df_net3_lagged = create_lag_features(df_net3_common, lags)
df_net1_lagged = create_lag_features(df_net1_common, lags)

# Χαρακτηριστικά και στόχοι για Net3
X_net3 = df_net3_lagged.drop(columns=common_junctions)  # Για είσοδο μόνο τα lags
y_net3 = df_net3_lagged[common_junctions]

# Χαρακτηριστικά και στόχοι για Net1
X_net1 = df_net1_lagged.drop(columns=common_junctions)
y_net1 = df_net1_lagged[common_junctions]

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

# -----------------------
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

# Οπτικοποίηση για έναν κοινό κόμβο (π.χ. πρώτο κοινό junction)
node_col = common_junctions[0]
plt.figure(figsize=(10,6))
plt.plot(y_net1.index, y_net1[node_col], label="Actual Net1")
plt.plot(y_net1.index, y_net1_pred[:, 0], label="Predicted Net1", linestyle='--')
plt.title(f"Demand Prediction for {node_col} (Net1)")
plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.legend()
plt.show()
