# Leak detection model using Random Forest Classifier

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Φόρτωση δεδομένων
df = pd.read_csv("leaks/pressure__labeled(Net3).csv")

# 2. Διαχωρισμός features και στόχου
X = df.drop(columns=["leak"])
y = df["leak"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Εκπαίδευση μοντέλου
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# 5. Πρόβλεψη & αναφορά
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Leak", "Leak"], yticklabels=["No Leak", "Leak"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# # 7. Plot feature importances
# importances = model.feature_importances_
# feature_names = X.columns

# # Top 10 σημαντικότεροι κόμβοι
# indices = importances.argsort()[-10:][::-1]

# plt.figure(figsize=(8,5))
# sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
# plt.title("Top 10 Important Features (Nodes)")
# plt.xlabel("Importance")
# plt.ylabel("Sensor Node")
# plt.tight_layout()
# plt.show()
