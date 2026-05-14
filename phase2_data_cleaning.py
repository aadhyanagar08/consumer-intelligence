import pandas as pd
import numpy as np
import os

print("=" * 50)
print("PHASE 2 — DATA CLEANING")
print("=" * 50)

# ── Load ──────────────────────────────────────────
df = pd.read_excel('data/Online Retail.xlsx')
print(f"\nRaw data shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── Clean ─────────────────────────────────────────
df = df.dropna(subset=['CustomerID'])
print(f"\nAfter dropping missing CustomerIDs: {df.shape}")

df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"After dropping cancellations: {df.shape}")

df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
print(f"After dropping bad quantities/prices: {df.shape}")

df = df.drop_duplicates()
print(f"After dropping duplicates: {df.shape}")

# ── Feature engineering ───────────────────────────
df['Revenue'] = df['Quantity'] * df['UnitPrice']

customers = df.groupby('CustomerID').agg(
    total_spend     = ('Revenue', 'sum'),
    num_orders      = ('InvoiceNo', 'nunique'),
    avg_order_value = ('Revenue', 'mean')
).reset_index()

print(f"\nCustomer features shape: {customers.shape}")
print(f"\nSample:\n{customers.head()}")
print(f"\nStats:\n{customers.describe()}")

# ── Save ──────────────────────────────────────────
os.makedirs('data', exist_ok=True)
customers.to_csv('data/cleaned.csv', index=False)
print("\nSaved to data/cleaned.csv")
