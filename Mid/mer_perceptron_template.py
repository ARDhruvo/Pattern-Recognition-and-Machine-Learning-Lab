"""
CSE4214 Pattern Recognition Lab - REUSABLE TEMPLATE
Covers two recurring question types:
   (A) Minimum Error Rate (Bayes/Gaussian) Classifier
   (B) Perceptron Algorithm (single-sample & batch) with polynomial features

Copy the block(s) you need into your own script and adjust the CONFIG
section. Each block is independent - you don't need to run all of them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SHARED UTILITIES
# ============================================================

def load_xy_label(path, col_names=("x1", "x2", "label")):
    """Load a whitespace- or comma-separated file with no header."""
    df = pd.read_csv(path, sep=r"[,\s]+", engine="python",
                      header=None, names=list(col_names))
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


def plot_classes(df, xcol, ycol, title="", label_col="label", save_as=None):
    """Scatter plot with a distinct marker/color per class."""
    marker_cycle = [("o", "blue"), ("^", "red"), ("s", "green"), ("D", "purple")]
    plt.figure()
    for i, cls in enumerate(sorted(df[label_col].unique())):
        mk, col = marker_cycle[i % len(marker_cycle)]
        sub = df[df[label_col] == cls]
        plt.scatter(sub[xcol], sub[ycol], marker=mk, color=col, label=f"Class {cls}")
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(title)
    plt.legend()
    if save_as:
        plt.savefig(save_as, dpi=150)
    return plt.gcf()


def class_mean_cov(df, cls, feature_cols, label_col="label"):
    """Return (mean_vector, covariance_matrix) for one class."""
    sub = df.loc[df[label_col] == cls, list(feature_cols)].to_numpy()
    return sub.mean(axis=0), np.cov(sub, rowvar=False)


# ============================================================
# BLOCK A: MINIMUM ERROR RATE (GAUSSIAN) CLASSIFIER
# ============================================================

def manual_gaussian_pdf(x, mu, sigma):
    """N(x | mu, Sigma) computed by hand (no scipy.stats), per the
    'no library function for normal distribution' requirement.
    Works for any dimension D."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    D = len(mu)
    diff = x - mu
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)
    coeff = 1.0 / np.sqrt(((2 * np.pi) ** D) * det_sigma)
    exponent = -0.5 * diff @ inv_sigma @ diff
    return coeff * np.exp(exponent)


def discriminant(x, mu, sigma, prior):
    """Log-based discriminant g_i(x); numerically safer than raw pdf,
    and using log(prior) still gives the correct argmax decision."""
    x = np.asarray(x, dtype=float)
    diff = x - mu
    inv_sigma = np.linalg.inv(sigma)
    quad_term = diff @ inv_sigma @ diff
    log_det = np.log(np.linalg.det(sigma))
    return -0.5 * quad_term - 0.5 * log_det + np.log(prior)


def mer_classify(x, class_params):
    """class_params: list of (mu, sigma, prior) per class, class index
    order defines the label. Returns predicted class index."""
    scores = [discriminant(x, mu, sigma, prior) for mu, sigma, prior in class_params]
    return int(np.argmax(scores))


def plot_decision_boundary_2d(class_params, xlim, ylim, points_df=None,
                               x_col="x1", y_col="x2", label_col="label",
                               resolution=300, save_as=None):
    """Draws filled regions + contour of the winning class over a grid,
    plus the actual data points on top (Experiment-3 style figure)."""
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX, dtype=int)
    for i in range(resolution):
        for j in range(resolution):
            ZZ[i, j] = mer_classify([XX[i, j], YY[i, j]], class_params)

    plt.figure()
    plt.contourf(XX, YY, ZZ, alpha=0.25, levels=np.arange(len(class_params) + 1) - 0.5)
    plt.contour(XX, YY, ZZ, colors="k", linewidths=1)  # decision boundary line

    if points_df is not None:
        marker_cycle = [("o", "blue"), ("^", "red"), ("s", "green")]
        for i, cls in enumerate(sorted(points_df[label_col].unique())):
            mk, col = marker_cycle[i % len(marker_cycle)]
            sub = points_df[points_df[label_col] == cls]
            plt.scatter(sub[x_col], sub[y_col], marker=mk, color=col,
                        edgecolor="k", label=f"Class {cls}")
        plt.legend()

    plt.xlabel(x_col); plt.ylabel(y_col); plt.title("Decision boundary")
    if save_as:
        plt.savefig(save_as, dpi=150)
    return plt.gcf()


# ============================================================
# BLOCK B: PERCEPTRON (single-sample & batch) + POLY FEATURES
# ============================================================

def quadratic_features(x1, x2):
    """y = [x1^2, x2^2, x1*x2, x1, x2, 1]  (second-order mapping)."""
    return np.array([x1 ** 2, x2 ** 2, x1 * x2, x1, x2, 1.0])


def normalize_class(Y, labels, class_to_flip=1):
    """'Normalize' one class by negating its augmented feature vectors,
    so the perceptron rule becomes a single 'w^T y > 0' test for all
    samples (standard trick from the lecture)."""
    Y = Y.copy()
    mask = labels == class_to_flip
    Y[mask] = -Y[mask]
    return Y


def perceptron_single(Y, w0, alpha, max_iters=10000):
    """One-at-a-time (sequential) perceptron. Y already normalized so the
    correct rule is w^T y > 0 for every sample."""
    w = w0.astype(float).copy()
    n = len(Y)
    updates = 0
    for it in range(max_iters):
        misclassified = False
        for k in range(n):
            if w @ Y[k] <= 0:
                w = w + alpha * Y[k]
                updates += 1
                misclassified = True
        if not misclassified:
            return w, updates, True   # converged
    return w, updates, False          # did not converge in max_iters


def perceptron_batch(Y, w0, alpha, max_iters=10000):
    """Many-at-a-time (batch) perceptron: sum the corrections from all
    misclassified samples before updating w."""
    w = w0.astype(float).copy()
    n = len(Y)
    updates = 0
    for it in range(max_iters):
        mis_idx = [k for k in range(n) if w @ Y[k] <= 0]
        if not mis_idx:
            return w, updates, True
        correction = sum(Y[k] for k in mis_idx)
        w = w + alpha * correction
        updates += 1
    return w, updates, False


def sweep_learning_rates(Y, w0, alphas=np.arange(0.1, 1.01, 0.1)):
    """Builds the table required by the assignment: for a given initial
    weight vector, run both perceptron variants across a range of alphas."""
    rows = []
    for alpha in alphas:
        _, upd_single, _ = perceptron_single(Y, w0, alpha)
        _, upd_batch, _ = perceptron_batch(Y, w0, alpha)
        rows.append({"alpha": round(alpha, 2),
                     "one_at_a_time": upd_single,
                     "many_at_a_time": upd_batch})
    return pd.DataFrame(rows)


def bar_chart_from_table(table_df, title="Perceptron Comparison", save_as=None):
    x = np.arange(len(table_df))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, table_df["one_at_a_time"], width, label="One at a time")
    plt.bar(x + width / 2, table_df["many_at_a_time"], width, label="Many at a time")
    plt.xticks(x, table_df["alpha"])
    plt.xlabel("Learning rate"); plt.ylabel("No of iterations")
    plt.title(title); plt.legend()
    if save_as:
        plt.savefig(save_as, dpi=150)
    return plt.gcf()


# ============================================================
# EXAMPLE USAGE (delete / adapt for your specific assignment)
# ============================================================
if __name__ == "__main__":
    # --- Example: Block A ---
    # df = load_xy_label("train.txt")
    # mu0, sum0 = class_mean_cov(df, 0, ["x1", "x2"])
    # mu1, sum1 = class_mean_cov(df, 1, ["x1", "x2"])
    # params = [(mu0, sum0, 0.5), (mu1, sum1, 0.5)]
    # print(mer_classify([1.0, 1.0], params))
    # plot_decision_boundary_2d(params, xlim=(-4, 6), ylim=(-4, 6), points_df=df)

    # --- Example: Block B ---
    # df = load_xy_label("train.txt")
    # Y = np.array([quadratic_features(r.x1, r.x2) for r in df.itertuples()])
    # Y_norm = normalize_class(Y, df["label"].to_numpy())
    # for init_name, w0 in [("all_ones", np.ones(6)),
    #                       ("all_zeros", np.zeros(6)),
    #                       ("random", np.random.default_rng(42).normal(size=6))]:
    #     table = sweep_learning_rates(Y_norm, w0)
    #     print(init_name, "\n", table)
    #     bar_chart_from_table(table, title=f"Perceptron - {init_name}",
    #                           save_as=f"perceptron_{init_name}.png")
    pass
