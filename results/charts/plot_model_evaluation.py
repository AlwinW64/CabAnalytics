

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Save figures in the project directory.
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------
# Figure 1: Final 80/20 model comparison
# -------------------------------------------
models = ["Ridge Regression\n(regParam=0.01)", "Random Forest\n(40 trees)"]

rmse = [4.9986, 4.8331]
mae = [2.7384, 2.7086]
r2 = [0.8980, 0.9046]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

metrics = [
    ("RMSE", rmse, "Lower is better"),
    ("MAE", mae, "Lower is better"),
    ("R²", r2, "Higher is better"),
]

for ax, (metric_name, values, subtitle) in zip(axes, metrics):
    bars = ax.bar(models, values)
    ax.set_title(f"{metric_name}\n{subtitle}")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", labelrotation=0)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle(
    "CabAnalytics: Model Performance on the 80/20 Split",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "model_comparison_80_20.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# -------------------------------------------
# Figure 2: Three-fold CV RMSE
# -------------------------------------------
folds = np.array([1, 2, 3])

cv_rmse = {
    "Linear Regression (regParam=0.1)": [5.0059, 5.0267, 5.0113],
    "Random Forest (20 trees)": [4.9200, 5.1730, 5.1114],
    "Random Forest (40 trees)": [4.9450, 4.9171, 4.9330],
}

fig, ax = plt.subplots(figsize=(9, 5.5))

for model_name, scores in cv_rmse.items():
    ax.plot(
        folds,
        scores,
        marker="o",
        linewidth=2,
        label=model_name,
    )

ax.set_title("Three-Fold Cross-Validation: RMSE by Fold")
ax.set_xlabel("Cross-validation fold")
ax.set_ylabel("RMSE (lower is better)")
ax.set_xticks(folds)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "cv_rmse_by_fold.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("Figures saved to:", OUTPUT_DIR)
print("- model_comparison_80_20.png")
print("- cv_rmse_by_fold.png")
