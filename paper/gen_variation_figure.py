"""Generate variation sensitivity figure for SemBlend paper."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load results
with open("../results/variation-sensitivity-coldfix.json") as f:
    data = json.load(f)

summary = data["summary"]

# Ordered by decreasing cosine similarity
order = ["exact", "reorder", "partial_80", "paraphrase", "partial_60", "partial_40", "diverse"]
labels = ["Exact\n(1.00)", "Reorder\n(0.90)", "Partial 80\n(0.80)", "Paraphrase\n(0.77)",
          "Partial 60\n(0.65)", "Partial 40\n(0.47)", "Diverse\n(0.20)"]
cosines = [1.00, 0.90, 0.80, 0.77, 0.65, 0.47, 0.20]
means = [summary[vt]["mean_ppl"] for vt in order]
hit_rates = [summary[vt]["hit_rate"] for vt in order]
all_ppls = [summary[vt]["ppls"] for vt in order]

FONT_TICK = 9
FONT_LABEL = 10
FONT_TITLE = 11

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Left: PPL ratio vs variation type ---
x = np.arange(len(order))
# Box plot of individual PPL ratios
bp = ax1.boxplot(all_ppls, positions=x, widths=0.5, patch_artist=True,
                 medianprops=dict(color="firebrick", linewidth=2),
                 boxprops=dict(facecolor="steelblue", alpha=0.6),
                 whiskerprops=dict(linewidth=1.5),
                 flierprops=dict(marker="o", markersize=4, color="gray"))

# Add the FM2 outlier from threshold sweep
ax1.scatter([0.2], [1.270], color="red", marker="*", s=120, zorder=5, label="FM2 outlier (PPL=1.27)")

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Quality floor (PPL ratio=1.0)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=FONT_TICK)
ax1.set_xlabel("Variation Type (cosine similarity to seed)", fontsize=FONT_LABEL)
ax1.set_ylabel("PPL Ratio (SemBlend / Cold)", fontsize=FONT_LABEL)
ax1.set_title("Quality Invariance Across Similarity Levels", fontsize=FONT_TITLE, fontweight="bold")
ax1.set_ylim(0.95, 1.35)
ax1.legend(fontsize=9, loc="upper right")
ax1.tick_params(axis="y", labelsize=FONT_TICK)
ax1.grid(axis="y", alpha=0.3)

# --- Right: Hit rate vs cosine ---
ax2.bar(x, [h * 100 for h in hit_rates], color="steelblue", alpha=0.7, edgecolor="navy")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=FONT_TICK)
ax2.set_xlabel("Variation Type (cosine similarity to seed)", fontsize=FONT_LABEL)
ax2.set_ylabel("Hit Rate (%)", fontsize=FONT_LABEL)
ax2.set_title("LMCache Hit Rate by Variation Type", fontsize=FONT_TITLE, fontweight="bold")
ax2.set_ylim(0, 110)
ax2.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax2.tick_params(axis="y", labelsize=FONT_TICK)
ax2.grid(axis="y", alpha=0.3)

# Annotate: non-monotonic note
ax2.annotate("Non-monotonic:\nblock-hash, not cosine",
             xy=(5, 62), xytext=(4.0, 88),
             arrowprops=dict(arrowstyle="->", color="black", lw=1),
             fontsize=9, ha="center")

plt.tight_layout(pad=1.5)
plt.savefig("fig_variation_sensitivity.pdf", bbox_inches="tight", dpi=150)
plt.savefig("fig_variation_sensitivity.png", bbox_inches="tight", dpi=150)
print("Saved fig_variation_sensitivity.pdf and .png")
