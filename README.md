# Customer Intelligence Engine

**Dataset:** UCI Online Retail — 500k transactions from a UK gift shop (2010–2011)

---

## What each phase answers

| Phase | Question answered |
|-------|-------------------|
| 1 — Setup | How do I structure a data science project? |
| 2 — Data Cleaning | Who are our real customers, and what did they actually spend? |
| 3 — K-Means | What natural customer segments exist in our data? |
| 4 — Decision Tree | Which customers are about to churn? |
| 5 — RL Bandit | What marketing action should we take for each segment? |
| 6 — Dashboard | How do we present all of this to a non-technical stakeholder? |

---

## Project structure

```
customer-intelligence/
├── data/
│   ├── Online Retail.xlsx      # Raw dataset (UCI)
│   ├── cleaned.csv             # One row per customer, 3 features
│   ├── clustered.csv           # cleaned.csv + cluster labels
│   └── with_churn.csv          # clustered.csv + churn label + days_since
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_kmeans.ipynb
│   ├── 03_decision_tree.ipynb
│   ├── 04_rl_targeting.ipynb
│   └── 05_dashboard.ipynb
├── outputs/
│   ├── elbow_plot.png
│   ├── clusters_pca.png
│   ├── decision_tree.png
│   ├── rl_reward_curve.png
│   └── rl_q_values.png
├── phase2_data_cleaning.py
├── phase3_kmeans.py
├── phase4_decision_tree.py
├── phase5_rl_bandit.py
└── README.md
```

---

## How to run

```bash
# Run each phase in order
python phase2_data_cleaning.py
python phase3_kmeans.py
python phase4_decision_tree.py
python phase5_rl_bandit.py
```

Each script reads from the previous phase's output. Run them in order.

---

## Key concepts learned

- **K-Means** minimizes within-cluster sum of squared distances (inertia). Scaling matters because it uses Euclidean distance.
- **Decision Tree** splits on information gain — the feature that best separates churned from active customers at each node.
- **Multi-armed bandit** is the simplest form of RL: no states, just actions and rewards. Epsilon-greedy balances exploration vs exploitation.

---

## Results summary

After running all phases, fill in your actual numbers here:
- Number of customers after cleaning: ___
- Optimal K (from elbow): ___
- Churn rate: ___
- Decision tree accuracy: ___
- RL agent total reward after 1000 rounds: ___
