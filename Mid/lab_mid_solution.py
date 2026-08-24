"""
CSE4214 Pattern Recognition Lab - LAB MID SOLUTION
Minimum Error Rate Classifier with a derived feature

HOW TO USE
----------
1. Put your train.txt in the same folder as this script.
   Expected format (no header): x1  x2  label
   (label is 0 or 1). If your file uses commas, that's handled too.
2. Set LAST_3_DIGITS below to the last three digits of YOUR student ID.
3. Run:  python lab_mid_solution.py
   Two PNG figures + all Part A/B numbers will be printed/saved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- CONFIG -------------------------
LAST_3_DIGITS = 123          # <-- CHANGE THIS to your ID's last 3 digits
TRAIN_FILE = "train.txt"
# ------------------------------------------------------------

# =========================================================
# PART A
# =========================================================

# ---- 1. Read data & plot raw (x1, x2) ----
df = pd.read_csv(TRAIN_FILE, sep=r"[,\s]+", engine="python",
                  header=None, names=["x1", "x2", "label"])
df["label"] = df["label"].astype(int)

markers = {0: ("o", "blue"), 1: ("^", "red")}

plt.figure()
for cls, (mk, col) in markers.items():
    sub = df[df["label"] == cls]
    plt.scatter(sub["x1"], sub["x2"], marker=mk, color=col, label=f"Class {cls}")
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Raw training data (x1 vs x2)")
plt.legend()
plt.savefig("part_a1_raw_data.png", dpi=150)
plt.close()

# ---- 2. Derived feature = x2 + (last 3 digits of ID)/1000 ----
df["derived"] = df["x2"] + LAST_3_DIGITS / 1000.0

plt.figure()
for cls, (mk, col) in markers.items():
    sub = df[df["label"] == cls]
    plt.scatter(sub["x1"], sub["derived"], marker=mk, color=col, label=f"Class {cls}")
plt.xlabel("x1"); plt.ylabel("derived feature")
plt.title("x1 vs derived feature")
plt.legend()
plt.savefig("part_a2_derived_feature.png", dpi=150)
plt.close()

# ---- 3. Min / Max / Average for each feature ----
print("=" * 50)
print("PART A - Task 3: Min / Max / Average")
print("=" * 50)
for col in ["x1", "x2", "derived"]:
    print(f"{col:10s}  min={df[col].min():.4f}   "
          f"max={df[col].max():.4f}   mean={df[col].mean():.4f}")

# ---- 4. mu0, mu1, sum0(covariance0), sum1(covariance1) ----
# Using x1 and the derived feature, per the assignment instructions.
def class_stats(data, cls):
    sub = data.loc[data["label"] == cls, ["x1", "derived"]].to_numpy()
    mu = sub.mean(axis=0)
    sigma = np.cov(sub, rowvar=False)   # 2x2 covariance matrix
    return mu, sigma

mu0, sum0 = class_stats(df, 0)
mu1, sum1 = class_stats(df, 1)

print("\n" + "=" * 50)
print("PART A - Task 4: Mean vectors & Covariance matrices")
print("=" * 50)
print("mu0 (class 0 mean) =", mu0)
print("sum0 (class 0 covariance) =\n", sum0)
print("mu1 (class 1 mean) =", mu1)
print("sum1 (class 1 covariance) =\n", sum1)

# =========================================================
# PART B - Minimum Error Rate Classification
# =========================================================

prior0, prior1 = 0.5, 0.5

def discriminant(x, mu, sigma, prior):
    """g_i(x) = -1/2 (x-mu)^T Sigma^-1 (x-mu) - 1/2 ln|Sigma| + ln P(w_i)
    (the additive constant -D/2 ln(2*pi) is dropped since it's common
    to both classes and doesn't affect the argmax decision)"""
    x = np.asarray(x, dtype=float)
    inv_sigma = np.linalg.inv(sigma)
    diff = x - mu
    quad_term = diff @ inv_sigma @ diff
    log_det = np.log(np.linalg.det(sigma))
    return -0.5 * quad_term - 0.5 * log_det + np.log(prior)

# (x1, derived_feature), true_label
test_points = [
    ((2.5, 2.5),   0),
    ((2.0, 2.0),   0),
    ((0.8, 0.7),   0),
    ((-1.25, -1.25), 1),
    ((-0.7, -0.5), 1),
    ((-0.9, -0.7), 0),
]

print("\n" + "=" * 50)
print("PART B - Classification results")
print("=" * 50)

correct = 0
for pt, true_label in test_points:
    g0 = discriminant(pt, mu0, sum0, prior0)
    g1 = discriminant(pt, mu1, sum1, prior1)
    pred = 0 if g0 > g1 else 1
    is_correct = (pred == true_label)
    correct += is_correct
    print(f"point={pt}  g0={g0:8.4f}  g1={g1:8.4f}  "
          f"predicted={pred}  true={true_label}  {'OK' if is_correct else 'WRONG'}")

accuracy = correct / len(test_points) * 100
print(f"\nAccuracy = {correct}/{len(test_points)} = {accuracy:.2f}%")
