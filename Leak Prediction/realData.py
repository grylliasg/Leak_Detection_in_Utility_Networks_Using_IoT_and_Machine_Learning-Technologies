import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. φόρτωση dataset
df = pd.read_csv("Data/Leaks/PATRAS_DATA_WITH_LEAKS\dataset_with_leakages/1000626_with_leakages.csv")

# μετατροπή του Timestamp σε datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek

# 2. δημιουργία labels (χρησιμοποιούμε τη διαφορά με Variable Leakage)
threshold = 8.5
df['leak_label'] = (df['Volume_with_Variable_Leakage'] - df['Volume']) > threshold

# 3. Feature engineering
df['diff'] = df['Volume'].diff().fillna(0)
df['rolling_mean'] = df['Volume'].rolling(3, min_periods=1).mean()
df['rolling_std'] = df['Volume'].rolling(3, min_periods=1).std().fillna(0)

features = ['hour', 'day_of_week', 'Volume', 'diff', 'rolling_mean', 'rolling_std']
X = df[features]
y = df['leak_label']

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(df['leak_label'].value_counts())

# 5. εκπαιδευση μοντέλου
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. αξιολόγηση
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# κατανομη περιπτώσεων
df['leak_label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xticks([0,1], ['No Leak (False)', 'Leak (True)'])
plt.title("Κατανομή περιπτώσεων διαρροής")
plt.ylabel("Πλήθος")
plt.show()

# σύγκριση volume vs volume with variable leakage
plt.figure(figsize=(12,5))
plt.plot(df['Timestamp'], df['Volume'], label='Κανονικός Όγκος')
plt.plot(df['Timestamp'], df['Volume_with_Variable_Leakage'], label='Μεταβλητή Διαρροή', alpha=0.7)
plt.legend()
plt.title("Κατανάλωση με και χωρίς διαρροή")
plt.xlabel("Χρόνος")
plt.ylabel("Όγκος")
plt.show()

# Επισήμανση σημείων διαρροής
plt.figure(figsize=(12,5))
plt.plot(df['Timestamp'], df['Volume'], label='Κανονικός Όγκος')
plt.scatter(df.loc[df['leak_label'], 'Timestamp'], df.loc[df['leak_label'], 'Volume'],
            color='red', label='Ανιχνευμένη Διαρροή')
plt.legend()
plt.title("Ανίχνευση Διαρροών με βάση το threshold")
plt.xlabel("Χρόνος")
plt.ylabel("Όγκος")
plt.show()
