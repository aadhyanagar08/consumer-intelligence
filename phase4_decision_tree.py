import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

print("=" * 50)
print("PHASE 4 — DECISION TREE CLASSIFIER")
print("=" * 50)

# ── Load ──────────────────────────────────────────
df_raw = pd.read_excel('data/Online Retail.xlsx')
df_raw = df_raw.dropna(subset=['CustomerID'])
df_raw = df_raw[~df_raw['InvoiceNo'].astype(str).str.startswith('C')]
df_raw = df_raw[df_raw['Quantity'] > 0]
df_raw = df_raw[df_raw['UnitPrice'] > 0]
df_raw = df_raw.drop_duplicates()
df_raw['Revenue'] = df_raw['Quantity'] * df_raw['UnitPrice']

customers = pd.read_csv('data/clustered.csv')

# ── Churn label ───────────────────────────────────
# Churned = no purchase in last 90 days of the dataset
df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
snapshot_date = df_raw['InvoiceDate'].max()

last_purchase = df_raw.groupby('CustomerID')['InvoiceDate'].max().reset_index()
last_purchase.columns = ['CustomerID', 'last_purchase_date']
last_purchase['days_since'] = (snapshot_date - last_purchase['last_purchase_date']).dt.days
last_purchase['churned'] = (last_purchase['days_since'] > 90).astype(int)

customers = customers.merge(last_purchase, on='CustomerID')
print(f"\nChurn rate: {customers['churned'].mean():.1%}")
print(f"Churned: {customers['churned'].sum()} | Active: {(customers['churned']==0).sum()}")

# ── Features & split ──────────────────────────────
feature_cols = ['cluster', 'total_spend', 'num_orders', 'avg_order_value', 'days_since']
X = customers[feature_cols]
y = customers['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── Train ─────────────────────────────────────────
# max_depth=4 keeps it readable and reduces overfitting
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────
y_pred = dt.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"\nConfusion matrix:\n{cm}")

# ── Visualize tree ────────────────────────────────
os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=feature_cols,
          class_names=['Active', 'Churned'],
          filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree — Churn Classifier (max depth 4)')
plt.tight_layout()
plt.savefig('outputs/decision_tree.png', dpi=150)
plt.close()
print("\nTree saved to outputs/decision_tree.png")

# ── Save results ──────────────────────────────────
customers.to_csv('data/with_churn.csv', index=False)
print("Saved to data/with_churn.csv")
