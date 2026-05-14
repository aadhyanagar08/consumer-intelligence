import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("PHASE 5 — REINFORCEMENT LEARNING (MULTI-ARMED BANDIT)")
print("=" * 50)

# ── Setup ─────────────────────────────────────────
# 3 actions x 4 segments
# True response rates (hidden from agent — it has to learn these)
# Rows = segments, Cols = actions
# Actions: 0=discount, 1=email, 2=loyalty points

SEGMENTS = ['High-value loyal', 'High spend low freq',
            'Frequent budget', 'Occasional low spender']
ACTIONS  = ['Discount', 'Email', 'Loyalty points']

# Ground truth response rates per segment per action
TRUE_RATES = np.array([
    [0.30, 0.50, 0.60],   # High-value loyal: loyalty points work best
    [0.50, 0.20, 0.25],   # High spend low freq: discounts bring them back
    [0.25, 0.40, 0.35],   # Frequent budget: email works
    [0.40, 0.15, 0.10],   # Occasional low spender: discounts only thing that works
])

N_SEGMENTS = len(SEGMENTS)
N_ACTIONS  = len(ACTIONS)
N_ROUNDS   = 1000
EPSILON    = 0.1   # 10% explore, 90% exploit

# ── Agent state ───────────────────────────────────
# Q = estimated reward per (segment, action)
# N = how many times each (segment, action) was tried
Q = np.zeros((N_SEGMENTS, N_ACTIONS))
N = np.zeros((N_SEGMENTS, N_ACTIONS))

cumulative_rewards = []
total_reward = 0

print(f"\nRunning {N_ROUNDS} rounds with epsilon={EPSILON}")
print("Agent starts knowing nothing — learns by trying actions...")

# ── Epsilon-greedy loop ───────────────────────────
np.random.seed(42)
for round_num in range(N_ROUNDS):
    segment = np.random.randint(N_SEGMENTS)   # random customer arrives

    # Explore vs exploit
    if np.random.random() < EPSILON:
        action = np.random.randint(N_ACTIONS)  # explore: random action
    else:
        action = np.argmax(Q[segment])         # exploit: best known action

    # Simulate customer response
    reward = int(np.random.random() < TRUE_RATES[segment, action])

    # Update Q estimate (running average)
    N[segment, action] += 1
    Q[segment, action] += (reward - Q[segment, action]) / N[segment, action]

    total_reward += reward
    cumulative_rewards.append(total_reward)

print(f"\nTotal reward after {N_ROUNDS} rounds: {total_reward}")
print(f"Average response rate: {total_reward/N_ROUNDS:.3f}")

# ── What the agent learned ────────────────────────
print("\nLearned best action per segment:")
for s in range(N_SEGMENTS):
    best = np.argmax(Q[s])
    print(f"  {SEGMENTS[s]:30s} → {ACTIONS[best]:20s} (estimated rate: {Q[s,best]:.3f})")

print("\nTrue best action per segment:")
for s in range(N_SEGMENTS):
    best = np.argmax(TRUE_RATES[s])
    print(f"  {SEGMENTS[s]:30s} → {ACTIONS[best]:20s} (true rate: {TRUE_RATES[s,best]:.3f})")

# ── Plot cumulative reward ────────────────────────
os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(cumulative_rewards, color='#1D9E75', linewidth=1.5)
plt.xlabel('Round')
plt.ylabel('Cumulative reward')
plt.title('RL Agent — Cumulative Reward Over Time')
plt.tight_layout()
plt.savefig('outputs/rl_reward_curve.png', dpi=150)
plt.close()
print("\nReward curve saved to outputs/rl_reward_curve.png")

# ── Plot learned Q values ─────────────────────────
fig, axes = plt.subplots(1, N_SEGMENTS, figsize=(14, 4))
colors = ['#1D9E75', '#7F77DD', '#D85A30']
for s, ax in enumerate(axes):
    bars = ax.bar(ACTIONS, Q[s], color=colors)
    ax.set_title(SEGMENTS[s], fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel('Estimated response rate' if s == 0 else '')
    ax.tick_params(axis='x', labelsize=8)
plt.suptitle('Agent Learned Q-Values per Segment', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/rl_q_values.png', dpi=150)
plt.close()
print("Q-values plot saved to outputs/rl_q_values.png")
