# ==========================================================
# ev_exit_baseline_compare.py – Compare Exit-Touch Scores vs. Model EV Ratings
# ==========================================================
"""
This script calculates a simple baseline player EV rating by assigning value only to the last 
player who touches the puck before a zone exit. It compares this to the model-derived EV delta 
scores for defensive zone players, and computes the correlation between the two approaches.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ---------- Load sequence data ----------
with open("hockey_sequences.json", "r") as f:
    zone_exit_data = json.load(f)["sequences"]

# ---------- Simple exit-touch EV attribution ----------
score_map = {"SC": 0.38, "SD": 0.16, "F": 0.06}
player_exit_counter = defaultdict(int)
player_exit_score = defaultdict(float)

for seq in zone_exit_data:
    result = seq["result"]
    score_val = score_map.get(result, 0.0)

    # Attribute exit EV only to final player with a valid ID
    last_touch = next(
        (ev for ev in reversed(seq["events"]) if ev.get("playerid") is not None),
        None
    )
    if last_touch:
        pid = last_touch["playerid"]
        player_exit_counter[pid] += 1
        player_exit_score[pid] += score_val

# ---------- Build exit stat dataframe ----------
exit_df = pd.DataFrame({
    "playerid": list(player_exit_counter.keys()),
    "exit_touch_counter": list(player_exit_counter.values()),
    "exit_value_score": list(player_exit_score.values())
})
exit_df["EVold"] = exit_df["exit_value_score"] / exit_df["exit_touch_counter"]

# ---------- Load model-derived EV ratings ----------
model_df = pd.read_csv("Player_D_Team_EV_Scores.csv")
model_df = model_df.rename(columns={"player_id": "playerid"})

# ---------- Merge and compare ----------
merged = pd.merge(model_df, exit_df[["playerid", "EVold"]], how="left", on="playerid")
merged = merged.dropna(subset=["avg_delta_ev", "EVold"])

# ---------- Correlation ----------
correlation = merged["avg_delta_ev"].corr(merged["EVold"])
print(f"📊 Correlation between model EV and hand-tagged EV: {correlation:.4f}")

# ---------- Plot ----------
plt.figure(figsize=(8, 6))
sns.regplot(
    data=merged, x="EVold", y="avg_delta_ev",
    scatter_kws={"s": 40, "alpha": 0.6}, line_kws={"color": "red"}
)
plt.xlabel("EV from Exit Touch Only")
plt.ylabel("EV from LSTM Model")
plt.title("Model EV vs. Last-Touch Exit Attribution")
plt.grid(True)
plt.tight_layout()
plt.text(
    x=0.05, y=0.95,
    s=f"r = {correlation:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Replace with your actual DataFrame slices
top_5_model = merged.sort_values(by="avg_delta_ev", ascending=False).head(5)
bottom_5_model = merged.sort_values(by="avg_delta_ev", ascending=True).head(5)

def plot_table(df, title):
    fig, ax = plt.subplots(figsize=(6, 2 + 0.3 * len(df)))
    ax.axis('off')
    table = ax.table(cellText=df[["playerid", "avg_delta_ev", "EVold"]].round(3).values,
                     colLabels=["Player ID", "Model EV", "Exit Touch EV"],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()

plot_table(top_5_model, "Top 5 by Model EV")
plot_table(bottom_5_model, "Bottom 5 by Model EV")


 












