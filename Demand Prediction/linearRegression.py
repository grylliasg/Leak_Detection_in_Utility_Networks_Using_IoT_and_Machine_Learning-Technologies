import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump

# 1. Φόρτωση δεδομένων
df = pd.read_csv("Data/Normal Flow/demand_no_leak(Net 3).csv")

# 2. Αφαίρεση index στήλης αν υπάρχει
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=df.columns[0])

# 3. Δημιουργία lag features για κάθε κόμβο
lags = 5
df_lagged = df.copy()

for lag in range(1, lags + 1):
    lagged = df.shift(lag) # Create the lag matrix
    lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
    df_lagged = pd.concat([df_lagged, lagged], axis=1)

# 4. Αφαίρεση NaN γραμμών που προκύπτουν από τα lags
df_lagged.dropna(inplace=True)
df_lagged.reset_index(drop=True, inplace=True)

# 5. Ορισμός χαρακτηριστικών και στόχων
X = df_lagged.drop(columns=df.columns)  # κρατάμε μόνο τα lags
y = df_lagged[df.columns]               # τα αρχικά demand columns είναι τα target

# 6. Διαχωρισμός σε train/test με βάση το χρόνο (αφού αφαιρέσαμε τα πρώτα lags)
middle = len(X) // 2
X_train, X_test = X.iloc[:middle], X.iloc[middle:]
y_train, y_test = y.iloc[:middle], y.iloc[middle:]

# 7. Εκπαίδευση Random Forest ή Linear Regression   
#model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
dump(model, 'Models/linear_model.pkl')

# 8. Πρόβλεψη
y_pred = model.predict(X_test)

# 9. Αξιολόγηση συνολικά
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 10. Οπτικοποίηση ενός κόμβου 
node_col = y.columns[30] # Name of column
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test[node_col], label="Actual")
plt.plot(y_test.index, y_pred[:, 30], label="Predicted", linestyle='--')
plt.title(f"Demand Prediction for {node_col}")
plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.legend()
plt.show()
